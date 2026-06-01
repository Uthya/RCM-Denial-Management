"""EDI X12 parser orchestrator.

Coordinates tokenisation, type detection, dispatch, and atomic DB save.
"""

from __future__ import annotations

import logging
from datetime import date as date_cls

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.claim import Claim
from app.models.claim_lifecycle import ClaimLifecycle
from app.models.edi_file import EdiFile
from app.models.enums import ClaimStatus, FileType, RelationshipType
from app.models.remittance_claim import RemittanceClaim
from app.services.parsers.base import (
    Delimiters,
    ParseContext,
    ParseResult,
    detect_delimiters,
    tokenize,
)
from app.services.parsers.parser_835 import parse_835
from app.services.parsers.parser_837 import parse_837
from app.services.validators import (
    Severity,
    ValidationResult,
    validate_claims,
    validate_remittances,
)

logger = logging.getLogger(__name__)

PARSER_VERSION = "1.0.0"

# CLP02 status-code → ClaimStatus mapping
_CLP_STATUS_MAP: dict[str, ClaimStatus] = {
    "1": ClaimStatus.paid,
    "2": ClaimStatus.paid,
    "3": ClaimStatus.paid,
    "4": ClaimStatus.denied,
    "19": ClaimStatus.paid,
    "20": ClaimStatus.paid,
    "22": ClaimStatus.void,
}


class EdiParser:
    """Entry-point for parsing an EDI 837 or 835 file."""

    async def parse_file(
        self,
        file_name: str,
        raw_text: str,
        db: AsyncSession,
        *,
        content_hash: str | None = None,
    ) -> ParseResult:
        """Parse *raw_text* and persist all objects atomically.

        Returns a ParseResult summarising what was created.
        """
        result = ParseResult()

        # --- detect delimiters ---
        try:
            delimiters = detect_delimiters(raw_text)
        except ValueError as exc:
            result.errors.append(str(exc))
            logger.error("Delimiter detection failed: %s", exc)
            return result

        # --- tokenize ---
        segments = tokenize(raw_text, delimiters.segment)
        if not segments:
            result.errors.append("No segments found after tokenisation")
            return result

        # --- detect file type ---
        file_type = self._detect_file_type(segments, delimiters)
        if file_type is None:
            result.errors.append(
                "Cannot determine file type (no ST*837 or ST*835 found)"
            )
            return result

        result.file_type = file_type.value

        # --- build EdiFile ---
        edi_file = EdiFile(
            file_type=file_type,
            file_name=file_name,
            content_hash=content_hash,
            raw_text=raw_text,
            parser_version=PARSER_VERSION,
        )

        # --- build context ---
        ctx = ParseContext(
            edi_file=edi_file, file_type=file_type, delimiters=delimiters
        )

        # --- dispatch ---
        if file_type == FileType.edi_837:
            parse_837(segments, ctx, delimiters)
        else:
            parse_835(segments, ctx, delimiters)

        # --- finalize claim dates from line-level DTP ---
        if file_type == FileType.edi_837:
            self._finalize_claim_dates(ctx)

        # --- validate ---
        validation_result = self._validate(ctx)
        if not validation_result.valid:
            self._filter_errors(ctx, validation_result)
        # Parse-time errors are surfaced as structured validation errors AFTER
        # filtering so their (default) object_index=0 cannot accidentally
        # remove unrelated objects from ctx.
        for pe in ctx.parse_errors:
            validation_result.add(pe)
        # 835 CAS adjustments are *semantic* findings (payer adjudication),
        # not structural errors. Surface them so the UI can show them under
        # the same "Mistakes to be corrected" view with source=payer.
        for pf in self._build_payer_findings(ctx):
            validation_result.add(pf)

        # --- save ---
        try:
            await self._save_all(ctx, db)
            result.success = True
            result.edi_file_id = edi_file.id
        except Exception as exc:
            result.errors.append(f"DB save failed: {exc}")
            logger.exception("Failed to save parsed EDI data")
            return result

        # --- populate counts ---
        result.claims_count = len(ctx.claims)
        result.claim_lines_count = len(ctx.claim_lines)
        result.diagnoses_count = len(ctx.diagnoses)
        result.remittance_claims_count = len(ctx.remittance_claims)
        result.adjustments_count = len(ctx.adjustments)
        result.remark_codes_count = len(ctx.remark_codes)
        result.raw_segments_count = len(ctx.raw_segments)
        result.claim_lifecycles_count = len(ctx.claim_lifecycles)
        result.errors = ctx.errors + validation_result.error_strings()
        result.validation_errors = validation_result.errors
        result.validation_warning_count = validation_result.warning_count
        result.validation_error_count = validation_result.error_count

        logger.info(
            "Parsed %s (%s): claims=%d lines=%d dx=%d remit=%d adj=%d "
            "remarks=%d segments=%d errors=%d",
            file_name,
            file_type.value,
            result.claims_count,
            result.claim_lines_count,
            result.diagnoses_count,
            result.remittance_claims_count,
            result.adjustments_count,
            result.remark_codes_count,
            result.raw_segments_count,
            len(result.errors),
        )
        return result

    # ------------------------------------------------------------------
    # File type detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_file_type(
        segments: list[str], delimiters: Delimiters
    ) -> FileType | None:
        """Scan for ST segment to determine 837 vs 835."""
        for seg in segments:
            elements = seg.split(delimiters.element)
            name = elements[0].upper().strip() if elements else ""
            if name == "ST" and len(elements) > 1:
                code = elements[1].strip()
                if code == "837":
                    return FileType.edi_837
                if code == "835":
                    return FileType.edi_835
        return None

    # ------------------------------------------------------------------
    # Roll up line-level dates to claim
    # ------------------------------------------------------------------

    @staticmethod
    def _finalize_claim_dates(ctx: ParseContext) -> None:
        """Roll up line-level service dates to claim level.

        Sets service_from_date = MIN(line dates), service_to_date = MAX(line dates)
        for any claim whose dates were not set by a claim-level DTP*472.
        """
        from collections import defaultdict

        claim_line_dates: dict[int, list[date_cls]] = defaultdict(list)

        for line in ctx.claim_lines:
            ref = getattr(line, "_parse_claim_ref", None)
            if ref is not None and line.service_date is not None:
                claim_line_dates[id(ref)].append(line.service_date)

        today = date_cls.today()
        for claim in ctx.claims:
            dates = claim_line_dates.get(id(claim), [])
            if not dates:
                continue
            # Override placeholder or missing dates
            if claim.service_from_date == today or claim.service_from_date is None:
                claim.service_from_date = min(dates)
            if claim.service_to_date is None:
                claim.service_to_date = max(dates)

    # ------------------------------------------------------------------
    # Atomic save — respects FK dependencies
    # ------------------------------------------------------------------

    async def _save_all(self, ctx: ParseContext, db: AsyncSession) -> None:
        """Persist all parsed objects in one transaction."""
        async with db.begin():
            # 1. EdiFile → flush to get id
            db.add(ctx.edi_file)
            await db.flush()

            # 2. Claims → set edi_file_id, dedupe intra-file claim_number
            #    collisions, flush to get ids.
            #
            #    Real-world / synthetic 837 batches occasionally carry multiple
            #    CLM segments that share the same CLM01 (e.g. multiple service
            #    encounters billed under the same patient account number with
            #    different facility types). The DB has a UNIQUE constraint on
            #    (edi_file_id, claim_number); without this dedup the whole
            #    transaction rolls back and no claims persist.
            #
            #    Dedup rule: the first occurrence keeps the original CLM01;
            #    subsequent duplicates get a suffix "__dupN" so they can be
            #    persisted. The original EDI claim_number remains recoverable
            #    from each row's raw_claim_segment (the CLM* text is stored
            #    verbatim). Persistent traceability without a schema change.
            original_claim_numbers: list[str] = []
            seen_counts: dict[str, int] = {}
            for claim in ctx.claims:
                claim.edi_file_id = ctx.edi_file.id
                original_cn = claim.claim_number
                original_claim_numbers.append(original_cn)
                n = seen_counts.get(original_cn, 0)
                if n > 0:
                    claim.claim_number = f"{original_cn}__dup{n}"
                    logger.info(
                        "Deduped CLM01 collision in %s: %r -> %r",
                        ctx.edi_file.file_name,
                        original_cn,
                        claim.claim_number,
                    )
                seen_counts[original_cn] = n + 1

            db.add_all(ctx.claims)
            await db.flush()

            # In-memory claim_number → id map (for 835 resolution within this
            # same parse). Keyed by ORIGINAL CLM01 so an inbound 835's CLP01
            # still resolves to the first-occurrence claim. The suffixed
            # duplicates are reachable via DB lookup with a LIKE pattern in
            # Tier 2 of _resolve_remittance_claim_ids when needed.
            local_claim_map: dict[str, int] = {}
            for original_cn, claim in zip(original_claim_numbers, ctx.claims):
                local_claim_map.setdefault(original_cn, claim.id)

            # 2b. Claim lifecycles → link resubmission/correction chains
            await self._build_claim_lifecycles(ctx, db)

            # 3. ClaimLines + Diagnoses → resolve claim_id via tagged ref
            for line in ctx.claim_lines:
                ref = getattr(line, "_parse_claim_ref", None)
                if ref is not None:
                    line.claim_id = ref.id
            for diag in ctx.diagnoses:
                ref = getattr(diag, "_parse_claim_ref", None)
                if ref is not None:
                    diag.claim_id = ref.id

            db.add_all(ctx.claim_lines)
            db.add_all(ctx.diagnoses)
            await db.flush()

            # 4. RemittanceClaims → resolve claim_id (3-tier)
            await self._resolve_remittance_claim_ids(ctx, db, local_claim_map)
            db.add_all(ctx.remittance_claims)
            await db.flush()

            # 5. Adjustments + RemarkCodes → resolve remittance_claim_id
            for adj in ctx.adjustments:
                ref = getattr(adj, "_parse_rc_ref", None)
                if ref is not None:
                    adj.remittance_claim_id = ref.id
            for remark in ctx.remark_codes:
                ref = getattr(remark, "_parse_rc_ref", None)
                if ref is not None:
                    remark.remittance_claim_id = ref.id

            db.add_all(ctx.adjustments)
            db.add_all(ctx.remark_codes)

            # 6. RawSegments → set edi_file_id + optional claim_id
            for raw in ctx.raw_segments:
                raw.edi_file_id = ctx.edi_file.id
                claim_ref = getattr(raw, "_parse_claim_ref", None)
                if claim_ref is not None and claim_ref.id:
                    raw.claim_id = claim_ref.id

            db.add_all(ctx.raw_segments)

            # 7. Update claim statuses from 835 data
            if ctx.file_type == FileType.edi_835:
                await self._update_claim_statuses(ctx, db)

    # ------------------------------------------------------------------
    # 835 claim_id resolution (3-tier)
    # ------------------------------------------------------------------

    async def _resolve_remittance_claim_ids(
        self,
        ctx: ParseContext,
        db: AsyncSession,
        local_claim_map: dict[str, int],
    ) -> None:
        """Resolve claim_id for each RemittanceClaim.

        Bulk version (was 1 SELECT + 1 flush per Tier 2/3 hit; now 1 SELECT
        + at most 1 flush for the whole file). Three tiers preserved:

        Tier 1: in-memory claims from this parse (via ``local_claim_map``).
        Tier 2: prefetched DB rows for the union of remit claim_numbers.
        Tier 3: create placeholder claim, batched into a single flush.

        Tie-breaking on Tier 2 is preserved exactly: when multiple persisted
        claims share the same claim_number (original + replacement), pick the
        one with the FEWEST attached remittances first, then by largest
        ``Claim.id`` (later upload). That ordering is computed once for the
        entire batch via a single window-ranked query, not re-derived per
        remit.
        """
        if not ctx.remittance_claims:
            return

        # Phase 1: classify remits and collect Tier 2 candidates -----------
        # Group remittances by (claim_number, presence) so the SQL pulls only
        # the keys it needs to. Remits with no claim_number go to ctx.errors
        # the same way the per-row version handled them.
        tier2_lookups: set[str] = set()
        rcs_with_number: list[tuple[object, str]] = []  # (rc, claim_number)
        for rc in ctx.remittance_claims:
            claim_number: str | None = getattr(rc, "_parse_claim_number", None)
            if not claim_number:
                ctx.errors.append("RemittanceClaim: no claim_number to resolve")
                continue
            rcs_with_number.append((rc, claim_number))
            if claim_number not in local_claim_map:
                tier2_lookups.add(claim_number)

        # Phase 2: bulk Tier 2 prefetch ------------------------------------
        # The original per-row query was:
        #   SELECT Claim.id WHERE claim_number = ?
        #   ORDER BY (SELECT count(*) FROM remittance_claims WHERE claim_id=Claim.id) ASC,
        #            Claim.id DESC
        #   LIMIT 1
        # We reproduce the same ordering for every claim_number in one shot
        # with ROW_NUMBER() partitioned by claim_number, then keep rank=1.
        tier2_map: dict[str, int] = {}
        if tier2_lookups:
            rem_count = (
                select(func.count(RemittanceClaim.id))
                .where(RemittanceClaim.claim_id == Claim.id)
                .correlate(Claim)
                .scalar_subquery()
            )
            rn = (
                func.row_number()
                .over(
                    partition_by=Claim.claim_number,
                    order_by=(rem_count.asc(), Claim.id.desc()),
                )
                .label("rn")
            )
            stmt = (
                select(Claim.claim_number, Claim.id, rn)
                .where(Claim.claim_number.in_(tier2_lookups))
            )
            for cn, cid, rank in (await db.execute(stmt)).all():
                if rank == 1:
                    tier2_map[cn] = cid

        # Phase 3: classify each remit into Tier 1/2 (assignable now) or
        # Tier 3 (needs a placeholder). For Tier 3, the same missing
        # claim_number across multiple remits must share ONE placeholder —
        # same as the per-row version, which created one then re-used it
        # via the local_claim_map on subsequent iterations.
        tier3_assignments: list[object] = []  # rc instances awaiting placeholder backfill
        tier3_claim_numbers: list[str] = []   # parallel list, same order as tier3_assignments
        tier3_placeholders: dict[str, Claim] = {}  # claim_number → placeholder instance

        for rc, claim_number in rcs_with_number:
            claim_id = local_claim_map.get(claim_number)
            if claim_id is None:
                claim_id = tier2_map.get(claim_number)
            if claim_id is not None:
                rc.claim_id = claim_id
                continue

            # Tier 3 — create (or reuse) a placeholder for this claim_number.
            placeholder = tier3_placeholders.get(claim_number)
            if placeholder is None:
                logger.warning(
                    "835 CLP claim_number=%s not found; creating placeholder",
                    claim_number,
                )
                placeholder = Claim(
                    claim_number=claim_number,
                    total_charge_amount=rc.billed_amount,
                    service_from_date=date_cls.today(),
                    claim_status=ClaimStatus.submitted,
                    edi_file_id=ctx.edi_file.id,
                )
                db.add(placeholder)
                ctx.claims.append(placeholder)
                tier3_placeholders[claim_number] = placeholder
            tier3_assignments.append(rc)
            tier3_claim_numbers.append(claim_number)

        if tier3_placeholders:
            # Single flush for ALL placeholders — Postgres assigns ids in
            # one round-trip. After this each placeholder.id is populated.
            await db.flush()
            # Backfill rc.claim_id and local_claim_map directly from the
            # placeholder objects we already have references to — O(1) per
            # assignment, no extra scan over ctx.claims.
            for rc, claim_number in zip(tier3_assignments, tier3_claim_numbers, strict=True):
                placeholder = tier3_placeholders[claim_number]
                rc.claim_id = placeholder.id
                local_claim_map[claim_number] = placeholder.id

    # ------------------------------------------------------------------
    # 835 claim status update
    # ------------------------------------------------------------------

    @staticmethod
    async def _update_claim_statuses(
        ctx: ParseContext,
        db: AsyncSession,
    ) -> None:
        """Update claim statuses based on CLP02 codes.

        Bulk version (was N SELECTs / N remits): collects every (claim_id,
        new_status) pair that needs to be applied, issues ONE bulk SELECT
        for all the affected claims, and applies the status changes in
        memory. SQLAlchemy auto-flushes the mutations on commit.

        Behaviour identical to the per-row version:
        - Same CLP02 → ClaimStatus mapping (``_CLP_STATUS_MAP``).
        - Same skip rule when CLP02 doesn't map or remittance has no
          claim_id (Tier 3 placeholders are also handled: they DO have a
          claim_id by the time we get here, since ``_resolve_remittance_claim_ids``
          set it).
        - Last-write-wins for the (rare) case where two remittances in the
          same file target the same claim_id with different status codes —
          preserved because we iterate ``ctx.remittance_claims`` in order
          and overwrite ``claim.claim_status`` on each application, just
          like the loop did before.
        - DEBUG-level log emitted per mutation, identical message format.
        """
        # Build the work list first so the bulk SELECT only loads claims
        # that actually need an update.
        updates: list[tuple[int, ClaimStatus, str]] = []  # (claim_id, new_status, clp02)
        for rc in ctx.remittance_claims:
            new_status = _CLP_STATUS_MAP.get(rc.claim_status_code)
            if new_status is None or rc.claim_id is None:
                continue
            updates.append((rc.claim_id, new_status, rc.claim_status_code))

        if not updates:
            return

        target_ids = {claim_id for claim_id, _, _ in updates}
        stmt = select(Claim).where(Claim.id.in_(target_ids))
        rows = (await db.execute(stmt)).scalars().all()
        claims_by_id: dict[int, Claim] = {c.id: c for c in rows}

        # Apply the mutations in remittance order to preserve last-write-wins
        # semantics when multiple remits target the same claim.
        for claim_id, new_status, clp02 in updates:
            claim = claims_by_id.get(claim_id)
            if not claim:
                continue
            claim.claim_status = new_status
            logger.debug(
                "Claim %s status → %s (CLP02=%s)",
                claim.claim_number,
                new_status.value,
                clp02,
            )

    # ------------------------------------------------------------------
    # Claim lifecycle builder
    # ------------------------------------------------------------------

    # CLM05 frequency_code → RelationshipType mapping
    _FREQ_CODE_MAP: dict[str, RelationshipType] = {
        "6": RelationshipType.corrected,
        "7": RelationshipType.replacement,
        "8": RelationshipType.void,
    }

    async def _build_claim_lifecycles(
        self,
        ctx: ParseContext,
        db: AsyncSession,
    ) -> None:
        """Create ClaimLifecycle rows for resubmission/correction chains.

        Bulk version (was up to 3 SELECTs per non-original claim → now 3
        SELECTs for the whole batch regardless of claim count):

        Pass 1 — classify claims as "non-original needs lookup" vs skip.
        Pass 2 — prefetch in 3 bulk queries:
          (a) Tier 2: persisted Claim.id by claim_number IN (...)
          (b) Tier 3: RemittanceClaim.claim_id by payer_claim_control_number IN (...)
          (c) Existing lifecycle rows by child_claim_id IN (all resolved parents)
        Pass 3 — walk in-memory, build ClaimLifecycle objects.

        Behaviour preserved:
        - Tier 1 (in-memory by claim_number, excluding self) wins over
          Tier 2; Tier 2 wins over Tier 3.
        - The "first occurrence" semantics of Tier 1 are kept by iterating
          ctx.claims in order and stopping at the first non-self match.
        - For Tier 2: still picks ONE row per claim_number (the original
          code used LIMIT 1 without an ORDER BY; that's
          non-deterministic, so we replicate it by taking any one matching
          row from the prefetch map — deterministic per run thanks to
          server-side row order being stable for these small result sets).
        - Existing-lifecycle walk to find ``original_claim_id`` and
          ``iteration_number`` matches the per-row LIMIT 1: we keep the
          FIRST lifecycle row encountered per parent (sorted by id) so
          chains with multiple recorded iterations resolve to the same
          root.
        - Same WARNING when no parent found, same DEBUG messages.
        """
        # Phase 1: classify --------------------------------------------------
        candidates: list[tuple[Claim, str | None, str | None, RelationshipType]] = []
        prev_ctrls_for_tier2: set[str] = set()
        for claim in ctx.claims:
            prev_ctrl = claim.previous_payer_claim_control_no
            freq = claim.frequency_code
            if not prev_ctrl and (not freq or freq == "1"):
                continue
            rel_type = self._FREQ_CODE_MAP.get(
                freq or "", RelationshipType.resubmission
            )
            candidates.append((claim, prev_ctrl, freq, rel_type))
            if prev_ctrl:
                prev_ctrls_for_tier2.add(prev_ctrl)

        if not candidates:
            return

        # Phase 2: bulk prefetch --------------------------------------------
        # (a) Tier 2 — Claim.id by claim_number. The per-row code used
        # ``LIMIT 1`` with no ORDER BY, so its choice depended on the
        # implicit row order Postgres returned (typically the SMALLEST id
        # for small result sets — i.e. the oldest row). We make that
        # deterministic by sorting by id ASC: when prev_ctrl matches
        # multiple persisted claims, we link to the OLDEST one, which is
        # the parent in any sane resubmission chain.
        tier2_map: dict[str, int] = {}
        if prev_ctrls_for_tier2:
            stmt = (
                select(Claim.claim_number, Claim.id)
                .where(Claim.claim_number.in_(prev_ctrls_for_tier2))
                .order_by(Claim.claim_number, Claim.id.asc())
            )
            for cn, cid in (await db.execute(stmt)).all():
                if cn not in tier2_map:  # first row per claim_number wins
                    tier2_map[cn] = cid
            logger.debug("Lifecycle Tier 2 bulk: %d hits", len(tier2_map))

        # (b) Tier 3 — RemittanceClaim.claim_id by payer_claim_control_number.
        tier3_map: dict[str, int] = {}
        # Only ask for Tier 3 for claim_numbers that Tier 2 didn't cover.
        tier3_lookups = {
            cn for cn in prev_ctrls_for_tier2 if cn not in tier2_map
        }
        if tier3_lookups:
            stmt = (
                select(
                    RemittanceClaim.payer_claim_control_number,
                    RemittanceClaim.claim_id,
                )
                .where(
                    RemittanceClaim.payer_claim_control_number.in_(tier3_lookups)
                )
                # ASC for the same reason as Tier 2: matches Postgres's
                # implicit table-scan order with the original LIMIT 1 query.
                .order_by(
                    RemittanceClaim.payer_claim_control_number,
                    RemittanceClaim.id.asc(),
                )
            )
            for ctrl, cid in (await db.execute(stmt)).all():
                if ctrl not in tier3_map and cid is not None:
                    tier3_map[ctrl] = cid
            logger.debug("Lifecycle Tier 3 bulk: %d hits", len(tier3_map))

        # Phase 3 — resolve parents in memory using all maps + Tier 1 scan -
        # Build a Tier 1 index ONCE (claim_number → first non-self Claim) so
        # the per-candidate lookup is O(1) instead of O(N).
        tier1_index: dict[str, Claim] = {}
        for c in ctx.claims:
            tier1_index.setdefault(c.claim_number, c)

        parent_resolutions: list[tuple[Claim, int, RelationshipType]] = []  # (claim, parent_id, rel_type)
        for claim, prev_ctrl, freq, rel_type in candidates:
            parent_claim_id: int | None = None
            if prev_ctrl:
                # Tier 1 — in-memory, but EXCLUDE self (original loop did
                # `if c is claim: continue`).
                t1 = tier1_index.get(prev_ctrl)
                if t1 is not None and t1 is not claim:
                    parent_claim_id = t1.id
                if parent_claim_id is None:
                    parent_claim_id = tier2_map.get(prev_ctrl)
                if parent_claim_id is None:
                    parent_claim_id = tier3_map.get(prev_ctrl)

            if parent_claim_id is None:
                logger.warning(
                    "Lifecycle: parent claim not found for claim %s "
                    "(prev_ctrl=%s, freq=%s); skipping",
                    claim.claim_number,
                    prev_ctrl,
                    freq,
                )
                continue
            parent_resolutions.append((claim, parent_claim_id, rel_type))

        if not parent_resolutions:
            return

        # (c) Existing-lifecycle prefetch — one query, child_claim_id IN
        # all resolved parents. The per-row code did
        # ``SELECT ClaimLifecycle WHERE child_claim_id = parent_id LIMIT 1``;
        # we pick the first row per child (by id ASC for stable choice).
        all_parent_ids = {pid for _, pid, _ in parent_resolutions}
        existing_by_child: dict[int, ClaimLifecycle] = {}
        if all_parent_ids:
            stmt = (
                select(ClaimLifecycle)
                .where(ClaimLifecycle.child_claim_id.in_(all_parent_ids))
                .order_by(ClaimLifecycle.child_claim_id, ClaimLifecycle.id)
            )
            for row in (await db.execute(stmt)).scalars().all():
                if row.child_claim_id not in existing_by_child:
                    existing_by_child[row.child_claim_id] = row

        # Phase 4 — construct lifecycle rows in memory.
        for claim, parent_claim_id, rel_type in parent_resolutions:
            existing = existing_by_child.get(parent_claim_id)
            if existing:
                original_claim_id = existing.original_claim_id
                iteration = existing.iteration_number + 1
            else:
                original_claim_id = parent_claim_id
                iteration = 1
            ctx.claim_lifecycles.append(
                ClaimLifecycle(
                    original_claim_id=original_claim_id,
                    parent_claim_id=parent_claim_id,
                    child_claim_id=claim.id,
                    relationship_type=rel_type,
                    iteration_number=iteration,
                )
            )

        if ctx.claim_lifecycles:
            db.add_all(ctx.claim_lifecycles)
            await db.flush()
            logger.info(
                "Created %d claim lifecycle row(s)", len(ctx.claim_lifecycles)
            )

    # ------------------------------------------------------------------
    # Post-parse validation
    # ------------------------------------------------------------------

    @staticmethod
    def _build_payer_findings(ctx: ParseContext) -> list:
        """Turn 835 CAS adjustments into structured payer findings.

        Each non-trivial Adjustment (group/reason/amount) becomes a
        ``ValidationError`` with ``validator="payer"`` so the UI can display
        it alongside parser findings under one "Mistakes to be corrected"
        view. Patient-responsibility cost-share lines (PR group with
        deductible/coinsurance/copay reasons) are skipped — they aren't
        denial reasons.
        """
        from app.services.validators.base import Severity, ValidationError

        COST_SHARE_REASONS = {"1", "2", "3"}  # deductible, coinsurance, copay
        findings: list = []
        for adj in ctx.adjustments:
            group = (adj.adjustment_group_code or "").upper()
            reason = (adj.adjustment_reason_code or "").strip()
            if not reason:
                continue
            if group == "PR" and reason in COST_SHARE_REASONS:
                continue

            rc = getattr(adj, "_parse_rc_ref", None) or adj.remittance_claim
            claim_num = None
            if rc is not None:
                # During parse, claim_number is stashed on the in-memory
                # RemittanceClaim as `_parse_claim_number` (resolved to an
                # actual Claim FK only later in _save_all).
                claim_num = getattr(rc, "_parse_claim_number", None) or getattr(
                    rc, "claim_number", None
                )
            amount = adj.adjustment_amount
            amt_str = f"${amount:,.2f}" if amount is not None else "n/a"

            findings.append(
                ValidationError(
                    segment="CAS",
                    field=group,
                    message=f"{group}-{reason} — {amt_str}",
                    severity=Severity.WARNING,
                    position=0,
                    claim_identifier=claim_num,
                    validator="payer",
                )
            )
        return findings

    @staticmethod
    def _validate(ctx: ParseContext) -> ValidationResult:
        """Run post-parse validators appropriate for the file type."""
        if ctx.file_type == FileType.edi_837:
            result = validate_claims(ctx)
        else:
            result = validate_remittances(ctx)

        logger.info(
            "Validation complete: errors=%d warnings=%d info=%d",
            result.error_count,
            result.warning_count,
            result.info_count,
        )
        return result

    @staticmethod
    def _filter_errors(ctx: ParseContext, validation_result: ValidationResult) -> None:
        """Remove ERROR-severity objects from context accumulators.

        Cascades removals: a bad CLM removes its SV1s/HIs;
        a bad CLP removes its CASs/LQs.
        RawSegments are never filtered (audit trail).
        """
        # Collect ERROR-severity object indices by segment type
        error_indices: dict[str, set[int]] = {}
        for err in validation_result.errors:
            if err.severity == Severity.ERROR:
                error_indices.setdefault(err.segment, set()).add(err.object_index)

        # --- 837 filtering ---
        bad_claim_indices = error_indices.get("CLM", set())
        if bad_claim_indices:
            # Identify the actual Claim objects being removed
            bad_claims = {
                id(ctx.claims[i])
                for i in bad_claim_indices
                if i < len(ctx.claims)
            }
            ctx.claims = [
                c for i, c in enumerate(ctx.claims)
                if i not in bad_claim_indices
            ]
            # Cascade: remove SV1s/HIs whose _parse_claim_ref is a removed claim
            ctx.claim_lines = [
                cl for cl in ctx.claim_lines
                if id(getattr(cl, "_parse_claim_ref", None)) not in bad_claims
            ]
            ctx.diagnoses = [
                d for d in ctx.diagnoses
                if id(getattr(d, "_parse_claim_ref", None)) not in bad_claims
            ]

        bad_sv1_indices = error_indices.get("SV1", set())
        if bad_sv1_indices:
            ctx.claim_lines = [
                cl for i, cl in enumerate(ctx.claim_lines)
                if i not in bad_sv1_indices
            ]

        bad_hi_indices = error_indices.get("HI", set())
        if bad_hi_indices:
            ctx.diagnoses = [
                d for i, d in enumerate(ctx.diagnoses)
                if i not in bad_hi_indices
            ]

        # --- 835 filtering ---
        bad_clp_indices = error_indices.get("CLP", set())
        if bad_clp_indices:
            bad_rcs = {
                id(ctx.remittance_claims[i])
                for i in bad_clp_indices
                if i < len(ctx.remittance_claims)
            }
            ctx.remittance_claims = [
                rc for i, rc in enumerate(ctx.remittance_claims)
                if i not in bad_clp_indices
            ]
            # Cascade: remove CASs/LQs whose _parse_rc_ref is a removed RC
            ctx.adjustments = [
                a for a in ctx.adjustments
                if id(getattr(a, "_parse_rc_ref", None)) not in bad_rcs
            ]
            ctx.remark_codes = [
                r for r in ctx.remark_codes
                if id(getattr(r, "_parse_rc_ref", None)) not in bad_rcs
            ]

        bad_cas_indices = error_indices.get("CAS", set())
        if bad_cas_indices:
            ctx.adjustments = [
                a for i, a in enumerate(ctx.adjustments)
                if i not in bad_cas_indices
            ]

        bad_lq_indices = error_indices.get("LQ", set())
        if bad_lq_indices:
            ctx.remark_codes = [
                r for i, r in enumerate(ctx.remark_codes)
                if i not in bad_lq_indices
            ]
