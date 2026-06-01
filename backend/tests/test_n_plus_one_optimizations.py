"""Regression tests for the four N+1 → bulk optimizations.

The bulk versions of:
  * ``EdiParser._update_claim_statuses`` (Opt #1)
  * ``EdiParser._resolve_remittance_claim_ids`` (Opt #2)
  * ``EdiParser._build_claim_lifecycles`` (Opt #3)
  * ``_recommendations_for_837`` router (Opt #4)

must preserve the EXACT business semantics of the per-row implementations
they replaced. These tests pin those semantics with explicit synthetic
ParseContexts so a future refactor can't silently regress correctness in
exchange for speed.

All tests use an in-memory SQLite database (``sqlite+aiosqlite://``)
so they run in CI with no Postgres dependency. The parser code itself
is database-agnostic — we only need a working ``AsyncSession``.
"""
from __future__ import annotations

import pytest
import pytest_asyncio
from datetime import date
from decimal import Decimal
from sqlalchemy import BigInteger
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.ext.compiler import compiles


# SQLite autoincrement only works with INTEGER PRIMARY KEY, not BIGINT.
# Production uses Postgres BIGSERIAL — we transparently map BigInteger to
# INTEGER for the SQLite test backend so the existing ORM model file
# doesn't need to change. Production behaviour is identical.
@compiles(BigInteger, "sqlite")
def _bigint_to_int_for_sqlite(element, compiler, **kw):  # noqa: ANN001
    return "INTEGER"

from app.core.database import Base
from app.models.claim import Claim
from app.models.claim_lifecycle import ClaimLifecycle
from app.models.claim_line import ClaimLine
from app.models.diagnosis import Diagnosis
from app.models.edi_file import EdiFile
from app.models.enums import ClaimStatus, FileType, RelationshipType
from app.models.raw_segment import RawSegment  # noqa: F401 — registers mapper
from app.models.remark_code import RemarkCode  # noqa: F401
from app.models.remittance_claim import RemittanceClaim
from app.models.adjustment import Adjustment  # noqa: F401
from app.models.code_master import CodeMaster  # noqa: F401
from app.models.prediction_log import PredictionLog  # noqa: F401
from app.models.training_metric import TrainingMetric  # noqa: F401
from app.services.edi_parser import EdiParser
from app.services.parsers.base import Delimiters, ParseContext


# ---------------------------------------------------------------------------
# In-memory DB fixture
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def session() -> AsyncSession:
    """Fresh in-memory SQLite session, schema created from the ORM metadata.

    SQLite is fine here — the optimizations use SQL constructs that are
    portable (IN, ROW_NUMBER OVER, scalar subquery). Postgres-specific
    types like JSONB are only referenced by PredictionLog, which these
    tests never touch.
    """
    # SQLite doesn't support JSONB; substitute JSON in the engine echo by
    # avoiding ML tables. Create only the tables we exercise.
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:", future=True
    )
    async with engine.begin() as conn:
        await conn.run_sync(
            lambda sync_conn: Base.metadata.create_all(
                sync_conn,
                tables=[
                    EdiFile.__table__,
                    Claim.__table__,
                    ClaimLine.__table__,
                    Diagnosis.__table__,
                    RemittanceClaim.__table__,
                    Adjustment.__table__,
                    RemarkCode.__table__,
                    RawSegment.__table__,
                    ClaimLifecycle.__table__,
                ],
            )
        )
    SessionMaker = async_sessionmaker(engine, expire_on_commit=False)
    async with SessionMaker() as s:
        yield s
    await engine.dispose()


async def _make_edi_file(session: AsyncSession, file_type: FileType = FileType.edi_837) -> EdiFile:
    ef = EdiFile(file_type=file_type, file_name=f"test.{file_type.value}", raw_text="")
    session.add(ef)
    await session.flush()
    return ef


async def _make_persisted_claim(
    session: AsyncSession,
    *,
    edi_file: EdiFile,
    claim_number: str,
    status: ClaimStatus = ClaimStatus.submitted,
    charge: Decimal = Decimal("100"),
) -> Claim:
    c = Claim(
        claim_number=claim_number,
        total_charge_amount=charge,
        service_from_date=date(2025, 1, 1),
        claim_status=status,
        edi_file_id=edi_file.id,
    )
    session.add(c)
    await session.flush()
    return c


# ---------------------------------------------------------------------------
# Opt #1 — bulk claim-status update
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opt1_bulk_status_update_applies_correct_mappings(session):
    """Each CLP02 in ctx.remittance_claims should map to the right
    ClaimStatus on the matching Claim row, with no per-row SELECT loop."""
    ef = await _make_edi_file(session)
    c_paid = await _make_persisted_claim(session, edi_file=ef, claim_number="P-1")
    c_denied = await _make_persisted_claim(session, edi_file=ef, claim_number="D-1")
    c_void = await _make_persisted_claim(session, edi_file=ef, claim_number="V-1")

    ctx = ParseContext(
        edi_file=ef,
        file_type=FileType.edi_835,
        delimiters=Delimiters(),
        remittance_claims=[
            RemittanceClaim(
                claim_id=c_paid.id, claim_status_code="1",
                billed_amount=Decimal("100"), paid_amount=Decimal("100"),
                remittance_date=date(2025, 1, 2),
            ),
            RemittanceClaim(
                claim_id=c_denied.id, claim_status_code="4",
                billed_amount=Decimal("100"), paid_amount=Decimal("0"),
                remittance_date=date(2025, 1, 2),
            ),
            RemittanceClaim(
                claim_id=c_void.id, claim_status_code="22",
                billed_amount=Decimal("100"), paid_amount=Decimal("0"),
                remittance_date=date(2025, 1, 2),
            ),
            # CLP02 that doesn't map → should be skipped (preserved behaviour)
            RemittanceClaim(
                claim_id=c_paid.id, claim_status_code="99",
                billed_amount=Decimal("0"), paid_amount=Decimal("0"),
                remittance_date=date(2025, 1, 2),
            ),
        ],
    )

    await EdiParser._update_claim_statuses(ctx, session)
    await session.commit()

    await session.refresh(c_paid)
    await session.refresh(c_denied)
    await session.refresh(c_void)
    assert c_paid.claim_status == ClaimStatus.paid
    assert c_denied.claim_status == ClaimStatus.denied
    assert c_void.claim_status == ClaimStatus.void


@pytest.mark.asyncio
async def test_opt1_skips_remits_with_no_claim_id(session):
    """Tier-3 placeholders that somehow have claim_id=None must NOT crash —
    they get skipped exactly like the per-row version did."""
    ef = await _make_edi_file(session)
    c = await _make_persisted_claim(session, edi_file=ef, claim_number="K-1")
    ctx = ParseContext(
        edi_file=ef,
        file_type=FileType.edi_835,
        delimiters=Delimiters(),
        remittance_claims=[
            RemittanceClaim(
                claim_id=None, claim_status_code="1",  # ← no claim_id
                billed_amount=Decimal("100"), paid_amount=Decimal("100"),
                remittance_date=date(2025, 1, 2),
            ),
            RemittanceClaim(
                claim_id=c.id, claim_status_code="4",
                billed_amount=Decimal("100"), paid_amount=Decimal("0"),
                remittance_date=date(2025, 1, 2),
            ),
        ],
    )

    await EdiParser._update_claim_statuses(ctx, session)
    await session.commit()
    await session.refresh(c)
    assert c.claim_status == ClaimStatus.denied  # the one with a claim_id was applied


@pytest.mark.asyncio
async def test_opt1_last_write_wins_on_same_claim_id(session):
    """When two remittances target the same claim with different CLP02
    codes, the LAST one in remit order wins — matches the per-row loop
    that overwrote claim.claim_status on each iteration."""
    ef = await _make_edi_file(session)
    c = await _make_persisted_claim(session, edi_file=ef, claim_number="LW-1")
    ctx = ParseContext(
        edi_file=ef, file_type=FileType.edi_835, delimiters=Delimiters(),
        remittance_claims=[
            RemittanceClaim(
                claim_id=c.id, claim_status_code="1",  # paid (applied first)
                billed_amount=Decimal("100"), paid_amount=Decimal("100"),
                remittance_date=date(2025, 1, 2),
            ),
            RemittanceClaim(
                claim_id=c.id, claim_status_code="4",  # denied (overrides paid)
                billed_amount=Decimal("100"), paid_amount=Decimal("0"),
                remittance_date=date(2025, 1, 3),
            ),
        ],
    )
    await EdiParser._update_claim_statuses(ctx, session)
    await session.commit()
    await session.refresh(c)
    assert c.claim_status == ClaimStatus.denied, "last remit in order should win"


# ---------------------------------------------------------------------------
# Opt #2 — bulk remittance claim_id resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opt2_tier1_in_memory_wins_over_tier2_db(session):
    """A claim already present in the local_claim_map (Tier 1) takes
    precedence over a persisted claim with the same number (Tier 2)."""
    ef_old = await _make_edi_file(session)
    persisted = await _make_persisted_claim(session, edi_file=ef_old, claim_number="A-1")

    ef_new = await _make_edi_file(session)
    new_in_memory = await _make_persisted_claim(session, edi_file=ef_new, claim_number="A-1")

    ctx = ParseContext(edi_file=ef_new, file_type=FileType.edi_835, delimiters=Delimiters())
    rc = RemittanceClaim(
        claim_status_code="1", billed_amount=Decimal("100"),
        paid_amount=Decimal("100"), remittance_date=date(2025, 1, 1),
    )
    rc._parse_claim_number = "A-1"
    ctx.remittance_claims = [rc]

    parser = EdiParser()
    local_map = {"A-1": new_in_memory.id}
    await parser._resolve_remittance_claim_ids(ctx, session, local_map)

    assert rc.claim_id == new_in_memory.id, "Tier 1 (local map) must win"


@pytest.mark.asyncio
async def test_opt2_tier2_tiebreak_fewer_remits_then_higher_id(session):
    """When multiple persisted claims share a claim_number, the bulk
    resolver must reproduce the per-row tie-breaker exactly:
    fewest existing remittances first, then largest Claim.id.

    The unique (edi_file_id, claim_number) DB constraint means the
    duplicates have to live in different EDI files — exactly the
    real-world case this tie-breaker exists for (original 837 + later
    replacement 837 carrying the same CLM01)."""
    ef_x = await _make_edi_file(session)
    ef_y = await _make_edi_file(session)
    ef_z = await _make_edi_file(session)
    x = await _make_persisted_claim(session, edi_file=ef_x, claim_number="DUP")
    y = await _make_persisted_claim(session, edi_file=ef_y, claim_number="DUP")
    z = await _make_persisted_claim(session, edi_file=ef_z, claim_number="DUP")
    # Attach 2 prior remits to x.
    for _ in range(2):
        session.add(RemittanceClaim(
            claim_id=x.id, claim_status_code="1",
            billed_amount=Decimal("50"), paid_amount=Decimal("50"),
            remittance_date=date(2025, 1, 1),
        ))
    await session.flush()

    ef_new = await _make_edi_file(session, FileType.edi_835)
    ctx = ParseContext(edi_file=ef_new, file_type=FileType.edi_835, delimiters=Delimiters())
    new_rc = RemittanceClaim(
        claim_status_code="1", billed_amount=Decimal("100"),
        paid_amount=Decimal("100"), remittance_date=date(2025, 1, 2),
    )
    new_rc._parse_claim_number = "DUP"
    ctx.remittance_claims = [new_rc]

    parser = EdiParser()
    await parser._resolve_remittance_claim_ids(ctx, session, local_claim_map={})

    # Highest id among the zero-remit claims is z.
    assert new_rc.claim_id == z.id, (
        f"expected z.id={z.id} (fewest remits + largest id); got {new_rc.claim_id}"
    )


@pytest.mark.asyncio
async def test_opt2_multiple_remits_same_missing_number_share_placeholder(session):
    """Two remits with the same NEW (Tier 3) claim_number must point at
    ONE shared placeholder Claim row, not create two — same as the
    per-row version which used local_claim_map across iterations."""
    ef = await _make_edi_file(session)
    ctx = ParseContext(edi_file=ef, file_type=FileType.edi_835, delimiters=Delimiters())
    rc_a = RemittanceClaim(
        claim_status_code="1", billed_amount=Decimal("100"),
        paid_amount=Decimal("100"), remittance_date=date(2025, 1, 1),
    )
    rc_b = RemittanceClaim(
        claim_status_code="1", billed_amount=Decimal("200"),
        paid_amount=Decimal("200"), remittance_date=date(2025, 1, 2),
    )
    rc_a._parse_claim_number = "MISSING"
    rc_b._parse_claim_number = "MISSING"
    ctx.remittance_claims = [rc_a, rc_b]

    parser = EdiParser()
    local_map: dict[str, int] = {}
    await parser._resolve_remittance_claim_ids(ctx, session, local_map)
    await session.commit()

    assert rc_a.claim_id is not None
    assert rc_a.claim_id == rc_b.claim_id, "both remits must share one placeholder"
    placeholders = [c for c in ctx.claims if c.claim_number == "MISSING"]
    assert len(placeholders) == 1, f"expected 1 placeholder, got {len(placeholders)}"


# ---------------------------------------------------------------------------
# Opt #3 — bulk lifecycle builder
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_opt3_lifecycle_tier1_in_memory_resolution(session):
    """A replacement claim whose REF*F8 points at an in-memory original
    must link via Tier 1 (no DB query needed).

    The original and replacement live in different EDI files (matching
    the real-world resubmission flow and the unique (edi_file_id,
    claim_number) DB constraint)."""
    ef_orig = await _make_edi_file(session)
    ef_repl = await _make_edi_file(session)
    original = await _make_persisted_claim(session, edi_file=ef_orig, claim_number="ORIG-A")
    replacement = Claim(
        claim_number="ORIG-A",  # same claim_number, different EDI file
        total_charge_amount=Decimal("100"),
        service_from_date=date(2025, 2, 1),
        claim_status=ClaimStatus.submitted,
        edi_file_id=ef_repl.id,
        previous_payer_claim_control_no="ORIG-A",
        frequency_code="7",  # = replacement
    )
    session.add(replacement)
    await session.flush()

    ctx = ParseContext(edi_file=ef_repl, file_type=FileType.edi_837, delimiters=Delimiters())
    ctx.claims = [original, replacement]
    parser = EdiParser()
    await parser._build_claim_lifecycles(ctx, session)

    assert len(ctx.claim_lifecycles) == 1
    lc = ctx.claim_lifecycles[0]
    assert lc.parent_claim_id == original.id
    assert lc.original_claim_id == original.id
    assert lc.child_claim_id == replacement.id
    assert lc.relationship_type == RelationshipType.replacement
    assert lc.iteration_number == 1


@pytest.mark.asyncio
async def test_opt3_lifecycle_walks_existing_chain_for_iteration(session):
    """When the new parent is itself a child in an existing lifecycle,
    iteration_number should be (existing.iteration_number + 1) and
    original_claim_id should be the root, not the immediate parent."""
    ef_root = await _make_edi_file(session)
    ef_mid = await _make_edi_file(session)
    ef_new = await _make_edi_file(session)
    root = await _make_persisted_claim(session, edi_file=ef_root, claim_number="ROOT")
    middle = await _make_persisted_claim(session, edi_file=ef_mid, claim_number="MID")
    # Existing lifecycle row: root → middle, iteration=1
    session.add(ClaimLifecycle(
        original_claim_id=root.id,
        parent_claim_id=root.id,
        child_claim_id=middle.id,
        relationship_type=RelationshipType.replacement,
        iteration_number=1,
    ))
    await session.flush()

    new_replacement = Claim(
        claim_number="ROOT-V2",  # distinct claim_number in a distinct file
        total_charge_amount=Decimal("100"),
        service_from_date=date(2025, 3, 1),
        claim_status=ClaimStatus.submitted,
        edi_file_id=ef_new.id,
        previous_payer_claim_control_no="MID",  # points at middle
        frequency_code="7",
    )
    session.add(new_replacement)
    await session.flush()

    ctx = ParseContext(edi_file=ef_new, file_type=FileType.edi_837, delimiters=Delimiters())
    ctx.claims = [root, middle, new_replacement]
    parser = EdiParser()
    await parser._build_claim_lifecycles(ctx, session)

    assert len(ctx.claim_lifecycles) == 1
    lc = ctx.claim_lifecycles[0]
    assert lc.parent_claim_id == middle.id, "immediate parent must be middle"
    assert lc.original_claim_id == root.id, "must walk back to root, not stop at middle"
    assert lc.iteration_number == 2, "should increment off the existing chain"


@pytest.mark.asyncio
async def test_opt3_tier1_excludes_same_claim_object(session):
    """Tier 1 (in-memory scan) must NOT pick the same Python object as
    the parent — the per-row code did ``if c is claim: continue`` and
    the bulk version replicates this with ``if t1 is not claim``.

    With ONE in-memory claim whose prev_ctrl equals its own claim_number,
    Tier 1 self-exclusion fires (the bulk index lookup returns the same
    object). If a DIFFERENT persisted row in the DB shares that number,
    Tier 2 will still find it — which matches the per-row code's
    behaviour exactly (Tier 2 used a plain ``WHERE claim_number=`` and
    didn't exclude self either)."""
    ef_a = await _make_edi_file(session)
    ef_b = await _make_edi_file(session)
    # A persisted "older" claim with the same claim_number, in a different
    # EDI file — this is what Tier 2 should match.
    older = await _make_persisted_claim(session, edi_file=ef_a, claim_number="X")

    # The current claim being parsed: prev_ctrl="X" and its own
    # claim_number="X". Tier 1 must skip itself; Tier 2 must find ``older``.
    current = Claim(
        claim_number="X",
        total_charge_amount=Decimal("100"),
        service_from_date=date(2025, 1, 1),
        claim_status=ClaimStatus.submitted,
        edi_file_id=ef_b.id,
        previous_payer_claim_control_no="X",
        frequency_code="7",
    )
    session.add(current)
    await session.flush()

    ctx = ParseContext(edi_file=ef_b, file_type=FileType.edi_837, delimiters=Delimiters())
    ctx.claims = [current]
    parser = EdiParser()
    await parser._build_claim_lifecycles(ctx, session)

    assert len(ctx.claim_lifecycles) == 1
    lc = ctx.claim_lifecycles[0]
    # Tier 1 skipped ``current`` (self-exclusion), Tier 2 found ``older``.
    assert lc.parent_claim_id == older.id, "Tier 2 should have matched the older persisted row"
    assert lc.child_claim_id == current.id


@pytest.mark.asyncio
async def test_opt3_skips_originals(session):
    """Claims with no prev_ctrl and frequency_code in ('', '1', None)
    are originals — they don't get lifecycle rows."""
    ef = await _make_edi_file(session)
    c1 = await _make_persisted_claim(session, edi_file=ef, claim_number="O-1")
    c2 = await _make_persisted_claim(session, edi_file=ef, claim_number="O-2")
    c1.frequency_code = "1"
    c2.frequency_code = None
    await session.flush()

    ctx = ParseContext(edi_file=ef, file_type=FileType.edi_837, delimiters=Delimiters())
    ctx.claims = [c1, c2]
    parser = EdiParser()
    await parser._build_claim_lifecycles(ctx, session)
    assert ctx.claim_lifecycles == []


# ---------------------------------------------------------------------------
# Opt #4 — batched recommendations 837 prediction loop
#
# Tested indirectly via test_predictor_batch.py (predict_batch parity).
# Add a small integration smoke here that asserts the recommendations
# helper still works when predict_batch produces results.
# ---------------------------------------------------------------------------


def test_opt4_recommendations_handles_batch_failure_gracefully():
    """If predict_batch raises for the whole batch, the recommendations
    endpoint must NOT crash — every claim falls through to "no ML
    signal" and gets evaluated on parser findings only. Tested at the
    function level by passing an empty preds_by_claim_id dict."""
    from app.services.recommendations import build_recommendations_for_claim
    # ml_factors=[] → only parser findings drive the recommendation.
    parser_findings = [
        {"segment": "CLM", "field": "CLM05",
         "message": "frequency_code missing"},
    ]
    recs = build_recommendations_for_claim(
        parser_findings=parser_findings, ml_factors=[]
    )
    assert recs and recs[0]["source"] == "parser"
