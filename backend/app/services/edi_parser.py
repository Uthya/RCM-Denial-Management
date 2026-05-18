"""EDI X12 parser orchestrator.

Coordinates tokenisation, type detection, dispatch, and atomic DB save.
"""

from __future__ import annotations

import logging
from datetime import date as date_cls

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.claim import Claim
from app.models.edi_file import EdiFile
from app.models.enums import ClaimStatus, FileType
from app.services.parsers.base import (
    Delimiters,
    ParseContext,
    ParseResult,
    detect_delimiters,
    tokenize,
)
from app.services.parsers.parser_835 import parse_835
from app.services.parsers.parser_837 import parse_837

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
        result.errors = ctx.errors

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
