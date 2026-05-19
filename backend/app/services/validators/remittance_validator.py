"""Post-parse validation for 835 (remittance) data."""

from __future__ import annotations

from app.services.parsers.base import ParseContext
from app.services.validators.base import (
    Severity,
    ValidationError,
    ValidationResult,
    validate_date_range,
    validate_non_negative_decimal,
    validate_required_string,
)

_VALIDATOR = "remittance_validator"


def validate_remittances(ctx: ParseContext) -> ValidationResult:
    """Run all 835 validations against a fully-populated ParseContext."""
    result = ValidationResult()
    _validate_remittance_claims(ctx, result)
    _validate_adjustments(ctx, result)
    _validate_remark_codes(ctx, result)
    return result


# ---------------------------------------------------------------------------
# Per-remittance-claim (CLP) checks
# ---------------------------------------------------------------------------

def _validate_remittance_claims(ctx: ParseContext, result: ValidationResult) -> None:
    for idx, rc in enumerate(ctx.remittance_claims):
        claim_number = getattr(rc, "_parse_claim_number", None)

        # _parse_claim_number required (defense-in-depth)
        err = validate_required_string(
            claim_number, "_parse_claim_number", "CLP",
            idx, idx, claim_identifier=claim_number, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # claim_status_code required
        err = validate_required_string(
            rc.claim_status_code, "claim_status_code", "CLP",
            idx, idx, claim_identifier=claim_number, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # billed_amount required + non-negative
        err = validate_non_negative_decimal(
            rc.billed_amount, "billed_amount", "CLP",
            idx, idx, required=True,
            claim_identifier=claim_number, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # paid_amount non-negative (not required)
        err = validate_non_negative_decimal(
            rc.paid_amount, "paid_amount", "CLP",
            idx, idx, required=False,
            claim_identifier=claim_number, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # remittance_date required + range
        err = validate_date_range(
            rc.remittance_date, "remittance_date", "CLP",
            idx, idx, required=True,
            claim_identifier=claim_number, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # paid_amount > billed_amount
        if (
            rc.paid_amount is not None
            and rc.billed_amount is not None
            and rc.paid_amount > rc.billed_amount
        ):
            result.add(ValidationError(
                segment="CLP",
                field="paid_amount",
                message=(
                    f"paid_amount ({rc.paid_amount}) exceeds "
                    f"billed_amount ({rc.billed_amount})"
                ),
                severity=Severity.WARNING,
                position=idx,
                object_index=idx,
                claim_identifier=claim_number,
                validator=_VALIDATOR,
            ))


# ---------------------------------------------------------------------------
# Per-adjustment (CAS) checks
# ---------------------------------------------------------------------------

def _validate_adjustments(ctx: ParseContext, result: ValidationResult) -> None:
    for idx, adj in enumerate(ctx.adjustments):
        ref = getattr(adj, "_parse_rc_ref", None)
        claim_number = None
        if ref is not None:
            claim_number = getattr(ref, "_parse_claim_number", None)

        # orphan CAS
        if ref is None:
            result.add(ValidationError(
                segment="CAS",
                field="_parse_rc_ref",
                message="Orphan adjustment — no parent remittance claim reference",
                severity=Severity.ERROR,
                position=idx,
                object_index=idx,
                claim_identifier=claim_number,
                validator=_VALIDATOR,
            ))

        # adjustment_group_code required (defense-in-depth)
        err = validate_required_string(
            adj.adjustment_group_code, "adjustment_group_code", "CAS",
            idx, idx, claim_identifier=claim_number, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # adjustment_reason_code required
        err = validate_required_string(
            adj.adjustment_reason_code, "adjustment_reason_code", "CAS",
            idx, idx, claim_identifier=claim_number, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # adjustment_amount required
        err = validate_non_negative_decimal(
            adj.adjustment_amount, "adjustment_amount", "CAS",
            idx, idx, required=True,
            claim_identifier=claim_number, validator=_VALIDATOR,
        )
        if err:
            result.add(err)


# ---------------------------------------------------------------------------
# Per-remark-code (LQ) checks
# ---------------------------------------------------------------------------

def _validate_remark_codes(ctx: ParseContext, result: ValidationResult) -> None:
    for idx, remark in enumerate(ctx.remark_codes):
        ref = getattr(remark, "_parse_rc_ref", None)
        claim_number = None
        if ref is not None:
            claim_number = getattr(ref, "_parse_claim_number", None)

        # orphan LQ
        if ref is None:
            result.add(ValidationError(
                segment="LQ",
                field="_parse_rc_ref",
                message="Orphan remark code — no parent remittance claim reference",
                severity=Severity.ERROR,
                position=idx,
                object_index=idx,
                claim_identifier=claim_number,
                validator=_VALIDATOR,
            ))

        # remark_code required (defense-in-depth)
        err = validate_required_string(
            remark.remark_code, "remark_code", "LQ",
            idx, idx, claim_identifier=claim_number, validator=_VALIDATOR,
        )
        if err:
            result.add(err)
