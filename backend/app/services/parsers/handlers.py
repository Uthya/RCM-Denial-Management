"""Segment handler functions for EDI X12 837/835 parsing.

Every handler is a pure function: no DB access, no side-effects beyond
mutating the ParseContext that is explicitly passed in.
"""

from __future__ import annotations

import logging
from datetime import date

from app.models.adjustment import Adjustment
from app.models.claim import Claim
from app.models.claim_line import ClaimLine
from app.models.diagnosis import Diagnosis
from app.models.enums import ClaimStatus
from app.models.raw_segment import RawSegment
from app.models.remark_code import RemarkCode
from app.models.remittance_claim import RemittanceClaim
from app.services.parsers.base import (
    Delimiters,
    ParseContext,
    safe_date,
    safe_decimal,
    safe_element,
    split_composite,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ISA — updates edi_file on context
# ---------------------------------------------------------------------------

def handle_isa(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> None:
    ctx.edi_file.sender_id = safe_element(elements, 6).strip()
    ctx.edi_file.receiver_id = safe_element(elements, 8).strip()
    ctx.edi_file.interchange_control_no = safe_element(elements, 13).strip()
    logger.info(
        "ISA parsed: sender=%s receiver=%s control=%s pos=%d",
        ctx.edi_file.sender_id,
        ctx.edi_file.receiver_id,
        ctx.edi_file.interchange_control_no,
        ctx.segment_position,
    )


# ---------------------------------------------------------------------------
# CLM — creates a new Claim
# ---------------------------------------------------------------------------

def handle_clm(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> Claim:
    claim_number = safe_element(elements, 1)
    if not claim_number:
        raise ValueError(f"CLM at pos {ctx.segment_position}: missing claim_number")

    total_charge = safe_decimal(safe_element(elements, 2))
    if total_charge is None:
        raise ValueError(
            f"CLM at pos {ctx.segment_position}: invalid total_charge_amount"
        )

    # CLM05 is a composite: facility_type:?:frequency_code
    clm05 = safe_element(elements, 5)
    parts = split_composite(clm05, delimiters.component)
    facility_type_code = parts[0] if parts else None
    frequency_code = parts[2] if len(parts) > 2 else None

    claim = Claim(
        claim_number=claim_number,
        total_charge_amount=total_charge,
        facility_type_code=facility_type_code,
        frequency_code=frequency_code,
        payer_name=ctx.current_payer_name,
        patient_member_id=ctx.current_patient_member_id,
        service_from_date=date.today(),  # placeholder, overridden by DTP*472
        claim_status=ClaimStatus.submitted,
        raw_claim_segment=raw_text,
    )

    # Reset per-claim state
    ctx.current_claim = claim
    ctx.current_line_number = 0
    ctx.current_diagnosis_sequence = 0
    ctx.current_service_date = None
    ctx.claims.append(claim)

    logger.debug(
        "CLM %s: charge=%s pos=%d", claim_number, total_charge, ctx.segment_position
    )
    return claim


# ---------------------------------------------------------------------------
# SV1 — creates a ClaimLine
# ---------------------------------------------------------------------------

def handle_sv1(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> ClaimLine:
    if ctx.current_claim is None:
        raise ValueError(
            f"SV1 at pos {ctx.segment_position}: no current claim in context"
        )

    # SV101 composite: HC:procedure_code:mod1:mod2:...
    sv101 = safe_element(elements, 1)
    parts = split_composite(sv101, delimiters.component)
    procedure_code = parts[1] if len(parts) > 1 else ""
    modifier1 = parts[2] if len(parts) > 2 and parts[2] else None
    modifier2 = parts[3] if len(parts) > 3 and parts[3] else None

    if not procedure_code:
        raise ValueError(
            f"SV1 at pos {ctx.segment_position}: missing procedure_code "
            f"(claim={ctx.current_claim.claim_number})"
        )

    billed_amount = safe_decimal(safe_element(elements, 2))
    if billed_amount is None:
        raise ValueError(
            f"SV1 at pos {ctx.segment_position}: invalid billed_amount "
            f"(claim={ctx.current_claim.claim_number})"
        )

    units = safe_decimal(safe_element(elements, 4)) or 1
    place_of_service = safe_element(elements, 5) or None

    ctx.current_line_number += 1

    line = ClaimLine(
        line_number=ctx.current_line_number,
        procedure_code=procedure_code,
        modifier1=modifier1,
        modifier2=modifier2,
        billed_amount=billed_amount,
        units=units,
        place_of_service=place_of_service,
        raw_sv1_segment=raw_text,
    )
    # Tag parent for FK resolution after flush
    line._parse_claim_ref = ctx.current_claim  # type: ignore[attr-defined]
    ctx.claim_lines.append(line)

    logger.debug(
        "SV1 %s line=%d: proc=%s billed=%s pos=%d",
        ctx.current_claim.claim_number,
        ctx.current_line_number,
        procedure_code,
        billed_amount,
        ctx.segment_position,
    )
    return line


# ---------------------------------------------------------------------------
# HI — creates Diagnosis records (multiple per segment)
# ---------------------------------------------------------------------------

def handle_hi(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> list[Diagnosis]:
    if ctx.current_claim is None:
        raise ValueError(
            f"HI at pos {ctx.segment_position}: no current claim in context"
        )

    diagnoses: list[Diagnosis] = []
    # HI*ABK:J0690*ABF:J0691*...  — elements[1..N] are composites
    for i in range(1, len(elements)):
        composite = safe_element(elements, i)
        if not composite:
            continue
        parts = split_composite(composite, delimiters.component)
        dx_type = parts[0] if parts else ""
        dx_code = parts[1] if len(parts) > 1 else ""

        if not dx_code:
            logger.warning(
                "HI at pos %d: empty diagnosis_code in composite %d "
                "(claim=%s)",
                ctx.segment_position,
                i,
                ctx.current_claim.claim_number,
            )
            continue

        ctx.current_diagnosis_sequence += 1

        diag = Diagnosis(
            diagnosis_code=dx_code,
            diagnosis_type=dx_type,
            sequence_number=ctx.current_diagnosis_sequence,
            raw_hi_segment=raw_text,
        )
        # Tag parent for FK resolution after flush
        diag._parse_claim_ref = ctx.current_claim  # type: ignore[attr-defined]
        diagnoses.append(diag)
        ctx.diagnoses.append(diag)

    logger.debug(
        "HI %s: %d diagnoses pos=%d",
        ctx.current_claim.claim_number,
        len(diagnoses),
        ctx.segment_position,
    )
    return diagnoses


# ---------------------------------------------------------------------------
# DTP — date segment; ownership depends on last_segment_type
# ---------------------------------------------------------------------------

def handle_dtp(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> None:
    qualifier = safe_element(elements, 1)
    date_value = safe_date(safe_element(elements, 3))

    if qualifier == "472" and date_value:
        if ctx.last_segment_type in ("CLM", "HI") and ctx.current_claim:
            # DTP*472 after CLM or HI → claim-level service date
            ctx.current_claim.service_from_date = date_value
            ctx.current_claim.service_to_date = date_value
            logger.debug(
                "DTP*472 → claim %s service_date=%s pos=%d",
                ctx.current_claim.claim_number,
                date_value,
                ctx.segment_position,
            )
        elif ctx.last_segment_type == "SV1" and ctx.claim_lines:
            ctx.claim_lines[-1].service_date = date_value
            logger.debug(
                "DTP*472 → claim_line service_date=%s pos=%d",
                date_value,
                ctx.segment_position,
            )
    elif qualifier == "472" and not date_value:
        claim_num = ctx.current_claim.claim_number if ctx.current_claim else "?"
        logger.warning(
            "DTP*472 at pos %d: could not parse date (claim=%s, raw=%s)",
            ctx.segment_position,
            claim_num,
            raw_text,
        )


# ---------------------------------------------------------------------------
# DTM — date/time reference (835); ownership depends on last_segment_type
# ---------------------------------------------------------------------------

def handle_dtm(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> None:
    qualifier = safe_element(elements, 1)
    date_value = safe_date(safe_element(elements, 2))

    if qualifier == "050" and date_value:
        ctx.current_remittance_date = date_value
        if ctx.current_remittance_claim:
            ctx.current_remittance_claim.remittance_date = date_value
        logger.debug(
            "DTM*050 → remittance_date=%s pos=%d",
            date_value,
            ctx.segment_position,
        )


# ---------------------------------------------------------------------------
# N1 — entity name; captures payer info
# ---------------------------------------------------------------------------

def handle_n1(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> None:
    qualifier = safe_element(elements, 1)
    name = safe_element(elements, 2)

    if qualifier == "PR":  # Payer
        ctx.current_payer_name = name
        logger.debug("N1*PR payer_name=%s pos=%d", name, ctx.segment_position)


# ---------------------------------------------------------------------------
# NM1 — individual/org name; captures subscriber/payer
# ---------------------------------------------------------------------------

def handle_nm1(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> None:
    qualifier = safe_element(elements, 1)

    if qualifier == "PR":  # Payer
        # NM1*PR*2*PAYER_NAME
        name = safe_element(elements, 3)
        if name:
            ctx.current_payer_name = name
            logger.debug("NM1*PR payer=%s pos=%d", name, ctx.segment_position)

    elif qualifier == "IL":  # Insured/subscriber
        member_id = safe_element(elements, 9)
        if member_id:
            ctx.current_patient_member_id = member_id
            logger.debug(
                "NM1*IL member_id=%s pos=%d", member_id, ctx.segment_position
            )

    elif qualifier == "QC":  # Patient
        member_id = safe_element(elements, 9)
        if member_id:
            ctx.current_patient_member_id = member_id
            logger.debug(
                "NM1*QC member_id=%s pos=%d", member_id, ctx.segment_position
            )


# ---------------------------------------------------------------------------
# REF — reference identification
# ---------------------------------------------------------------------------

def handle_ref(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> None:
    qualifier = safe_element(elements, 1)
    value = safe_element(elements, 2)

    # REF*1W = member ID (used in some 837 variants)
    if qualifier == "1W" and value:
        ctx.current_patient_member_id = value
        logger.debug("REF*1W member_id=%s pos=%d", value, ctx.segment_position)

    # REF*F8 = original payer claim control number (corrected/replacement claims)
    elif qualifier == "F8" and value and ctx.current_claim:
        ctx.current_claim.previous_payer_claim_control_no = value
        logger.debug("REF*F8 prev_payer_ctrl=%s pos=%d", value, ctx.segment_position)


# ---------------------------------------------------------------------------
# CLP — remittance claim (835)
# ---------------------------------------------------------------------------

def handle_clp(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> RemittanceClaim:
    claim_number = safe_element(elements, 1)
    if not claim_number:
        raise ValueError(
            f"CLP at pos {ctx.segment_position}: missing claim_number"
        )

    status_code = safe_element(elements, 2)
    billed = safe_decimal(safe_element(elements, 3))
    paid = safe_decimal(safe_element(elements, 4))
    payer_control = safe_element(elements, 7) or None

    if billed is None:
        raise ValueError(
            f"CLP at pos {ctx.segment_position}: invalid billed_amount "
            f"(claim={claim_number})"
        )

    rc = RemittanceClaim(
        claim_status_code=status_code,
        billed_amount=billed,
        paid_amount=paid if paid is not None else 0,
        payer_claim_control_number=payer_control,
        remittance_date=ctx.current_remittance_date or date.today(),
        raw_clp_segment=raw_text,
    )
    # Stash claim_number for FK resolution later
    rc._parse_claim_number = claim_number  # type: ignore[attr-defined]

    ctx.current_remittance_claim = rc
    ctx.remittance_claims.append(rc)

    logger.debug(
        "CLP %s: status=%s billed=%s paid=%s pos=%d",
        claim_number,
        status_code,
        billed,
        paid,
        ctx.segment_position,
    )
    return rc


# ---------------------------------------------------------------------------
# CAS — claim adjustment (up to 6 triplets per segment)
# ---------------------------------------------------------------------------

def handle_cas(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> list[Adjustment]:
    if ctx.current_remittance_claim is None:
        raise ValueError(
            f"CAS at pos {ctx.segment_position}: no current remittance claim"
        )

    group_code = safe_element(elements, 1)
    if not group_code:
        raise ValueError(
            f"CAS at pos {ctx.segment_position}: missing adjustment_group_code"
        )

    adjustments: list[Adjustment] = []
    # Triplets start at index 2: reason, amount, quantity (optional)
    # Indices: 2/3, 5/6, 8/9, 11/12, 14/15, 17/18
    for start in range(2, len(elements), 3):
        reason = safe_element(elements, start)
        if not reason:
            break

        amount = safe_decimal(safe_element(elements, start + 1))
        if amount is None:
            logger.warning(
                "CAS at pos %d: invalid amount for reason=%s, skipping triplet",
                ctx.segment_position,
                reason,
            )
            continue

        adj = Adjustment(
            adjustment_group_code=group_code,
            adjustment_reason_code=reason,
            adjustment_amount=amount,
            raw_cas_segment=raw_text,
        )
        # Tag parent for FK resolution after flush
        adj._parse_rc_ref = ctx.current_remittance_claim  # type: ignore[attr-defined]
        adjustments.append(adj)
        ctx.adjustments.append(adj)

    logger.debug(
        "CAS group=%s: %d adjustments pos=%d",
        group_code,
        len(adjustments),
        ctx.segment_position,
    )
    return adjustments


# ---------------------------------------------------------------------------
# LQ — remark code (835)
# ---------------------------------------------------------------------------

def handle_lq(
    elements: list[str],
    raw_text: str,
    ctx: ParseContext,
    delimiters: Delimiters,
) -> RemarkCode:
    if ctx.current_remittance_claim is None:
        raise ValueError(
            f"LQ at pos {ctx.segment_position}: no current remittance claim"
        )

    code = safe_element(elements, 2)
    if not code:
        # Some LQ segments use element 1 as the code qualifier, 2 as code
        code = safe_element(elements, 1)
    if not code:
        raise ValueError(
            f"LQ at pos {ctx.segment_position}: missing remark_code"
        )

    rc = RemarkCode(
        remark_code=code,
        raw_lq_segment=raw_text,
    )
    # Tag parent for FK resolution after flush
    rc._parse_rc_ref = ctx.current_remittance_claim  # type: ignore[attr-defined]
    ctx.remark_codes.append(rc)

    logger.debug("LQ remark=%s pos=%d", code, ctx.segment_position)
    return rc
