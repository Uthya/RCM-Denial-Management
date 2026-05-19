"""Post-parse validation for 837 (claim) data."""

from __future__ import annotations

from collections import Counter
from decimal import Decimal

from app.services.parsers.base import ParseContext
from app.services.validators.base import (
    Severity,
    ValidationError,
    ValidationResult,
    validate_date_range,
    validate_non_negative_decimal,
    validate_required_string,
)

_VALIDATOR = "claim_validator"


def validate_claims(ctx: ParseContext) -> ValidationResult:
    """Run all 837 validations against a fully-populated ParseContext."""
    result = ValidationResult()
    _validate_claims(ctx, result)
    _validate_duplicate_claims(ctx, result)
    _validate_claim_lines(ctx, result)
    _validate_diagnoses(ctx, result)
    return result


# ---------------------------------------------------------------------------
# Per-claim (CLM) checks
# ---------------------------------------------------------------------------

def _validate_claims(ctx: ParseContext, result: ValidationResult) -> None:
    for idx, claim in enumerate(ctx.claims):
        claim_id = getattr(claim, "claim_number", None)

        # claim_number required (defense-in-depth)
        err = validate_required_string(
            claim_id, "claim_number", "CLM", idx, idx,
            claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # total_charge_amount required + non-negative
        err = validate_non_negative_decimal(
            claim.total_charge_amount, "total_charge_amount", "CLM",
            idx, idx, required=True,
            claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # zero total_charge_amount warning
        if (
            claim.total_charge_amount is not None
            and claim.total_charge_amount == Decimal(0)
        ):
            result.add(ValidationError(
                segment="CLM",
                field="total_charge_amount",
                message="total_charge_amount is zero",
                severity=Severity.WARNING,
                position=idx,
                object_index=idx,
                claim_identifier=claim_id,
                validator=_VALIDATOR,
            ))

        # service_from_date required + range
        err = validate_date_range(
            claim.service_from_date, "service_from_date", "CLM",
            idx, idx, required=True,
            claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # service_to_date range (not required)
        err = validate_date_range(
            claim.service_to_date, "service_to_date", "CLM",
            idx, idx, required=False,
            claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)


# ---------------------------------------------------------------------------
# Duplicate claim detection
# ---------------------------------------------------------------------------

def _validate_duplicate_claims(ctx: ParseContext, result: ValidationResult) -> None:
    counts = Counter(
        c.claim_number for c in ctx.claims if c.claim_number
    )
    seen: dict[str, int] = {}
    for idx, claim in enumerate(ctx.claims):
        cn = claim.claim_number
        if not cn or counts[cn] < 2:
            continue
        if cn in seen:
            result.add(ValidationError(
                segment="CLM",
                field="claim_number",
                message=f"Duplicate claim_number '{cn}' (also at index {seen[cn]})",
                severity=Severity.WARNING,
                position=idx,
                object_index=idx,
                claim_identifier=cn,
                validator=_VALIDATOR,
            ))
        else:
            seen[cn] = idx


# ---------------------------------------------------------------------------
# Per-claim-line (SV1) checks
# ---------------------------------------------------------------------------

def _validate_claim_lines(ctx: ParseContext, result: ValidationResult) -> None:
    for idx, line in enumerate(ctx.claim_lines):
        ref = getattr(line, "_parse_claim_ref", None)
        claim_id = ref.claim_number if ref else None

        # orphan SV1
        if ref is None:
            result.add(ValidationError(
                segment="SV1",
                field="_parse_claim_ref",
                message="Orphan claim line — no parent claim reference",
                severity=Severity.ERROR,
                position=idx,
                object_index=idx,
                claim_identifier=claim_id,
                validator=_VALIDATOR,
            ))

        # procedure_code required (defense-in-depth)
        err = validate_required_string(
            line.procedure_code, "procedure_code", "SV1",
            idx, idx, claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # billed_amount required + non-negative
        err = validate_non_negative_decimal(
            line.billed_amount, "billed_amount", "SV1",
            idx, idx, required=True,
            claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # units non-negative
        err = validate_non_negative_decimal(
            line.units, "units", "SV1",
            idx, idx, required=False,
            claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # service_date range
        err = validate_date_range(
            line.service_date, "service_date", "SV1",
            idx, idx, required=False,
            claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)


# ---------------------------------------------------------------------------
# Per-diagnosis (HI) checks
# ---------------------------------------------------------------------------

def _validate_diagnoses(ctx: ParseContext, result: ValidationResult) -> None:
    for idx, diag in enumerate(ctx.diagnoses):
        ref = getattr(diag, "_parse_claim_ref", None)
        claim_id = ref.claim_number if ref else None

        # orphan HI
        if ref is None:
            result.add(ValidationError(
                segment="HI",
                field="_parse_claim_ref",
                message="Orphan diagnosis — no parent claim reference",
                severity=Severity.ERROR,
                position=idx,
                object_index=idx,
                claim_identifier=claim_id,
                validator=_VALIDATOR,
            ))

        # diagnosis_code required
        err = validate_required_string(
            diag.diagnosis_code, "diagnosis_code", "HI",
            idx, idx, claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)

        # diagnosis_type required
        err = validate_required_string(
            diag.diagnosis_type, "diagnosis_type", "HI",
            idx, idx, claim_identifier=claim_id, validator=_VALIDATOR,
        )
        if err:
            result.add(err)
