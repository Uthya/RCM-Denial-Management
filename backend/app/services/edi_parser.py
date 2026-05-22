"""EDI X12 parser orchestrator.

Coordinates tokenisation, type detection, dispatch, and atomic DB save.
"""

from __future__ import annotations

import logging
from datetime import date as date_cls

from sqlalchemy import select
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

            # 2. Claims → set edi_file_id, flush to get ids
            for claim in ctx.claims:
                claim.edi_file_id = ctx.edi_file.id
            db.add_all(ctx.claims)
            await db.flush()

            # In-memory claim_number → id map (for 835 resolution)
            local_claim_map: dict[str, int] = {
                c.claim_number: c.id for c in ctx.claims
            }

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

        Tier 1: in-memory claims from this parse.
        Tier 2: query DB for previously ingested claim by claim_number.
        Tier 3: create placeholder claim + log warning.
        """
        for rc in ctx.remittance_claims:
            claim_number: str | None = getattr(rc, "_parse_claim_number", None)
            if not claim_number:
                ctx.errors.append(
                    f"RemittanceClaim: no claim_number to resolve"
                )
                continue

            # Tier 1
            claim_id = local_claim_map.get(claim_number)

            # Tier 2
            if claim_id is None:
                stmt = (
                    select(Claim.id)
                    .where(Claim.claim_number == claim_number)
                    .limit(1)
                )
                row = (await db.execute(stmt)).scalar_one_or_none()
                if row is not None:
                    claim_id = row

            # Tier 3
            if claim_id is None:
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
                await db.flush()
                claim_id = placeholder.id
                local_claim_map[claim_number] = claim_id
                ctx.claims.append(placeholder)

            rc.claim_id = claim_id

    # ------------------------------------------------------------------
    # 835 claim status update
    # ------------------------------------------------------------------

    @staticmethod
    async def _update_claim_statuses(
        ctx: ParseContext,
        db: AsyncSession,
    ) -> None:
        """Update claim statuses based on CLP02 codes."""
        for rc in ctx.remittance_claims:
            new_status = _CLP_STATUS_MAP.get(rc.claim_status_code)
            if new_status is None or rc.claim_id is None:
                continue

            stmt = select(Claim).where(Claim.id == rc.claim_id)
            result = await db.execute(stmt)
            claim = result.scalar_one_or_none()
            if claim:
                claim.claim_status = new_status
                logger.debug(
                    "Claim %s status → %s (CLP02=%s)",
                    claim.claim_number,
                    new_status.value,
                    rc.claim_status_code,
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

        Called after claims are flushed (so they have ids).  For each
        claim with a non-original frequency_code or a
        previous_payer_claim_control_no, find the parent claim, walk
        existing lifecycle rows to the root, and create a new row.
        """
        for claim in ctx.claims:
            prev_ctrl = claim.previous_payer_claim_control_no
            freq = claim.frequency_code

            # Skip original submissions with no back-reference
            if not prev_ctrl and (not freq or freq == "1"):
                continue

            rel_type = self._FREQ_CODE_MAP.get(
                freq or "", RelationshipType.resubmission
            )

            # Find parent claim by claim_number or payer_claim_control_number
            # REF*F8 may contain either the original claim_number or the
            # payer-assigned control number from the 835 (CLP07).
            parent_claim_id: int | None = None
            if prev_ctrl:
                # Tier 1: in-memory by claim_number
                for c in ctx.claims:
                    if c is claim:
                        continue
                    if c.claim_number == prev_ctrl:
                        parent_claim_id = c.id
                        break

                # Tier 2: DB by claim_number
                if parent_claim_id is None:
                    stmt = (
                        select(Claim.id)
                        .where(Claim.claim_number == prev_ctrl)
                        .limit(1)
                    )
                    row = (await db.execute(stmt)).scalar_one_or_none()
                    logger.debug(
                        "Lifecycle Tier 2 (claim_number=%s): result=%s",
                        prev_ctrl, row,
                    )
                    if row is not None:
                        parent_claim_id = row

                # Tier 3: DB via remittance_claims.payer_claim_control_number
                if parent_claim_id is None:
                    stmt = (
                        select(RemittanceClaim.claim_id)
                        .where(
                            RemittanceClaim.payer_claim_control_number == prev_ctrl
                        )
                        .limit(1)
                    )
                    row = (await db.execute(stmt)).scalar_one_or_none()
                    logger.debug(
                        "Lifecycle Tier 3 (payer_ctrl=%s): result=%s",
                        prev_ctrl, row,
                    )
                    if row is not None:
                        parent_claim_id = row

            if parent_claim_id is None:
                logger.warning(
                    "Lifecycle: parent claim not found for claim %s "
                    "(prev_ctrl=%s, freq=%s); skipping",
                    claim.claim_number,
                    prev_ctrl,
                    freq,
                )
                continue

            # Walk existing lifecycles to find the original (root) claim
            original_claim_id = parent_claim_id
            iteration = 1

            # Check if parent is itself a child in an existing lifecycle row
            stmt = (
                select(ClaimLifecycle)
                .where(ClaimLifecycle.child_claim_id == parent_claim_id)
                .limit(1)
            )
            existing = (await db.execute(stmt)).scalar_one_or_none()
            if existing:
                original_claim_id = existing.original_claim_id
                iteration = existing.iteration_number + 1

            lifecycle = ClaimLifecycle(
                original_claim_id=original_claim_id,
                parent_claim_id=parent_claim_id,
                child_claim_id=claim.id,
                relationship_type=rel_type,
                iteration_number=iteration,
            )
            ctx.claim_lifecycles.append(lifecycle)

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
