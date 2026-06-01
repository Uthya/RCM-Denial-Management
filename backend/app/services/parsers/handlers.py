"""Segment handler functions for EDI X12 837/835 parsing.

Every handler is a pure function: no DB access, no side-effects beyond
mutating the ParseContext that is explicitly passed in.
"""

from __future__ import annotations

import logging
from datetime import date
from decimal import Decimal, InvalidOperation

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
    safe_date_range,
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
        # Billing provider is set at the submitter level (Loop 2010AA, NM1*85),
        # which precedes CLM in spec order — pull it from context. Rendering
        # provider (NM1*82) appears after CLM in Loop 2310B and is stamped on
        # the claim directly by handle_nm1.
        billing_provider_npi=ctx.current_billing_provider_npi,
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
    format_qual = safe_element(elements, 2)  # D8 or RD8
    raw_date = safe_element(elements, 3)
    from_date, to_date = safe_date_range(raw_date, format_qual)

    if qualifier == "472" and from_date:
        if ctx.last_segment_type in ("CLM", "HI") and ctx.current_claim:
            # DTP*472 after CLM or HI → claim-level service date
            ctx.current_claim.service_from_date = from_date
            ctx.current_claim.service_to_date = to_date
            logger.debug(
                "DTP*472 → claim %s service_from=%s service_to=%s pos=%d",
                ctx.current_claim.claim_number,
                from_date,
                to_date,
                ctx.segment_position,
            )
        elif ctx.last_segment_type == "SV1" and ctx.claim_lines:
            ctx.claim_lines[-1].service_date = from_date
            logger.debug(
                "DTP*472 → claim_line service_date=%s pos=%d",
                from_date,
                ctx.segment_position,
            )
    elif qualifier == "472" and not from_date:
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

    elif qualifier == "85":  # Billing provider (Loop 2010AA)
        # NM1*85*2*BILLING NAME****XX*1234567890
        # element 8 is the ID qualifier ("XX" for NPI), element 9 is the NPI.
        # Some 837s use other qualifiers (24=EIN) — only capture XX so the
        # column is consistently a 10-digit NPI and the target encoder isn't
        # poisoned with mixed identifier types.
        id_qualifier = safe_element(elements, 8)
        provider_id = safe_element(elements, 9)
        if provider_id and id_qualifier == "XX":
            # Park on the context until CLM creates a claim. NM1*85 typically
            # appears BEFORE CLM in 837P loop order (2010AA precedes 2300), so
            # capture it on ctx and let CLM consume.
            ctx.current_billing_provider_npi = provider_id
            # Also stamp the active claim if NM1*85 lands mid-claim (rare but
            # spec-permitted via 2310 loops); the CLM-time capture below covers
            # the standard case.
            if ctx.current_claim is not None:
                ctx.current_claim.billing_provider_npi = provider_id
            logger.debug(
                "NM1*85 billing_provider_npi=%s pos=%d",
                provider_id,
                ctx.segment_position,
            )

    elif qualifier == "82":  # Rendering provider (Loop 2310B)
        # NM1*82*1*LAST*FIRST****XX*1234567890
        id_qualifier = safe_element(elements, 8)
        provider_id = safe_element(elements, 9)
        if provider_id and id_qualifier == "XX":
            # Rendering provider is scoped to the current claim's encounter.
            if ctx.current_claim is not None:
                ctx.current_claim.rendering_provider_npi = provider_id
            logger.debug(
                "NM1*82 rendering_provider_npi=%s pos=%d",
                provider_id,
                ctx.segment_position,
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

    # REF*G1 / REF*G3 = prior authorization / predetermination of benefits.
    # Either qualifier predicts authorization-driven denial categories
    # (CARC 15, 95, 197, 198) — captured into a single column.
    elif qualifier in ("G1", "G3") and value and ctx.current_claim:
        # First wins: a claim with both G1 and G3 keeps the auth number from
        # whichever segment appeared first; both still encode the same
        # has_prior_authorization feature later.
        if not ctx.current_claim.authorization_number:
            ctx.current_claim.authorization_number = value
        logger.debug(
            "REF*%s authorization_number=%s pos=%d",
            qualifier,
            value,
            ctx.segment_position,
        )

    # REF*9F = referral number (Loop 2300). Predicts CARC 165 (referral
    # absent / exceeded) and the broader CARC 38/242 PCP-routing denials.
    elif qualifier == "9F" and value and ctx.current_claim:
        ctx.current_claim.referral_number = value
        logger.debug(
            "REF*9F referral_number=%s pos=%d",
            value,
            ctx.segment_position,
        )


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

    triplets, warnings = _parse_cas_triplets(elements[2:], ctx.segment_position)
    for w in warnings:
        logger.warning(w)

    adjustments: list[Adjustment] = []
    for reason, amount, quantity in triplets:
        adj = Adjustment(
            adjustment_group_code=group_code,
            adjustment_reason_code=reason,
            adjustment_amount=amount,
            quantity=quantity,
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
# CAS triplet parser
# ---------------------------------------------------------------------------

def _parse_cas_triplets(
    elements_post_group: list[str], segment_pos: int
) -> tuple[list[tuple[str, Decimal, Decimal | None]], list[str]]:
    """Parse CAS triplets, tolerant of both X12 spec and compact payer forms.

    X12 005010 CAS is a fixed-stride-3 segment: each adjustment occupies
    exactly three positions (reason, amount, quantity), and the quantity slot
    may be empty. Trailing empty positions may be truncated by the sender, so
    spec-compliant element counts after CAS01 are 2, 3, 5, 6, 8, 9, … — i.e.
    any count with ``count % 3 != 1``.

    Non-spec "compact" payers drop empty quantity slots entirely and emit
    reason+amount pairs only (counts 2, 4, 6, 8, …). Both forms must be
    handled or denial dollar amounts and CARC codes get misaligned across
    triplets.

    Strategy:
    1. If element count clearly fits compact only (``count % 3 == 1``), use
       stride-2.
    2. Otherwise try stride-3 first; fall back to stride-2 only if stride-3
       leaves a clearly broken triplet behind (more triplets parse under
       stride-2 than stride-3).
    3. Track and return warnings for any non-spec / partial parses so the
       caller can log them.

    Parameters
    ----------
    elements_post_group : list[str]
        The segment elements AFTER CAS01 (the group code) — i.e. everything
        from the first reason onward. Empty strings are preserved.
    segment_pos : int
        File position of the CAS segment, used only for warning messages.

    Returns
    -------
    triplets : list[(reason, amount, quantity_or_None)]
    warnings : list[str]
    """

    def _attempt(stride: int) -> tuple[list[tuple[str, Decimal, Decimal | None]], bool]:
        """Walk the element stream with a given stride. ``complete`` is True
        iff we stopped on a natural end (empty/trailing) rather than on a
        malformed reason/amount pair."""
        triplets_inner: list[tuple[str, Decimal, Decimal | None]] = []
        complete = True
        n = len(elements_post_group)
        i = 0
        while i < n:
            reason = (elements_post_group[i] or "").strip()
            if not reason:
                break  # natural end — trailing empties are OK
            if i + 1 >= n:
                complete = False
                break
            amount_str = (elements_post_group[i + 1] or "").strip()
            if not amount_str:
                complete = False
                break
            try:
                amount = Decimal(amount_str)
            except (InvalidOperation, ValueError):
                complete = False
                break
            quantity: Decimal | None = None
            if stride == 3 and i + 2 < n:
                q_str = (elements_post_group[i + 2] or "").strip()
                if q_str:
                    try:
                        quantity = Decimal(q_str)
                    except (InvalidOperation, ValueError):
                        quantity = None  # tolerate bad quantity, keep triplet
            triplets_inner.append((reason, amount, quantity))
            i += stride
        return triplets_inner, complete

    warnings: list[str] = []
    cleaned = [(e or "").strip() for e in elements_post_group]
    if not any(cleaned):
        return [], warnings

    # Determine the meaningful element count: strip pure trailing empties
    # (spec permits truncating empty positions at the end of the segment)
    # but preserve any *internal* empty positions — those are quantity
    # placeholders and are a strong signal of spec stride-3 form.
    trimmed = list(cleaned)
    while trimmed and not trimmed[-1]:
        trimmed.pop()
    has_internal_empty = any(not v for v in trimmed)
    effective_count = len(trimmed)

    # No internal empties + count incompatible with spec stride-3
    # (effective_count % 3 == 1) → must be compact stride-2 form.
    if not has_internal_empty and effective_count % 3 == 1:
        triplets_2, complete_2 = _attempt(stride=2)
        if complete_2 and triplets_2:
            warnings.append(
                f"CAS at pos {segment_pos}: non-spec compact form "
                f"({effective_count} non-empty elements, no quantity captured)"
            )
            return triplets_2, warnings

    # Try spec stride-3 first.
    triplets_3, complete_3 = _attempt(stride=3)
    if complete_3 and triplets_3:
        return triplets_3, warnings

    # Stride-3 incomplete — try stride-2 as fallback.
    triplets_2, complete_2 = _attempt(stride=2)
    if complete_2 and len(triplets_2) > len(triplets_3):
        warnings.append(
            f"CAS at pos {segment_pos}: stride-3 incomplete after {len(triplets_3)} "
            f"triplet(s); falling back to compact stride-2 ({len(triplets_2)} triplets)"
        )
        return triplets_2, warnings

    if triplets_3:
        warnings.append(
            f"CAS at pos {segment_pos}: partial stride-3 parse — "
            f"{len(triplets_3)} triplet(s) captured; remaining elements may be malformed"
        )
        return triplets_3, warnings
    if triplets_2:
        warnings.append(
            f"CAS at pos {segment_pos}: partial stride-2 parse — "
            f"{len(triplets_2)} triplet(s) captured"
        )
        return triplets_2, warnings

    warnings.append(
        f"CAS at pos {segment_pos}: no valid triplets parsed from "
        f"{effective_count} non-empty element(s)"
    )
    return [], warnings


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
