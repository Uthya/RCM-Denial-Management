"""Core validation types and shared helpers for post-parse validation."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal


class Severity(str, enum.Enum):
    """Validation error severity level."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class ValidationError:
    """A single validation issue found during post-parse inspection."""

    segment: str
    field: str
    message: str
    severity: Severity
    position: int = 0
    object_index: int = 0
    claim_identifier: str | None = None
    validator: str = ""

    def to_string(self) -> str:
        """Backward-compatible string format for the errors list."""
        prefix = f"[{self.severity.value}]"
        loc = f"{self.segment}.{self.field}" if self.field else self.segment
        claim_part = f" claim={self.claim_identifier}" if self.claim_identifier else ""
        return f"{prefix} {loc}: {self.message}{claim_part}"


@dataclass
class ValidationResult:
    """Accumulates validation errors from one or more validators."""

    valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)

    def add(self, error: ValidationError) -> None:
        """Append an error and update validity flag."""
        self.errors.append(error)
        if error.severity == Severity.ERROR:
            self.valid = False

    def merge(self, other: ValidationResult) -> None:
        """Merge another result into this one (preserves insertion order)."""
        self.errors.extend(other.errors)
        if not other.valid:
            self.valid = False

    def error_strings(self) -> list[str]:
        """Return all errors as formatted strings."""
        return [e.to_string() for e in self.errors]

    @property
    def error_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == Severity.WARNING)

    @property
    def info_count(self) -> int:
        return sum(1 for e in self.errors if e.severity == Severity.INFO)

    @property
    def has_errors(self) -> bool:
        return self.error_count > 0


# ---------------------------------------------------------------------------
# Shared helper validators (return ValidationError | None)
# ---------------------------------------------------------------------------

_DATE_MIN = date(2000, 1, 1)
_DATE_FUTURE_THRESHOLD = timedelta(days=365)


def validate_required_string(
    value: str | None,
    field_name: str,
    segment: str,
    position: int,
    object_index: int,
    claim_identifier: str | None = None,
    validator: str = "",
) -> ValidationError | None:
    """Return an ERROR if *value* is missing or blank."""
    if not value or not str(value).strip():
        return ValidationError(
            segment=segment,
            field=field_name,
            message=f"{field_name} is required",
            severity=Severity.ERROR,
            position=position,
            object_index=object_index,
            claim_identifier=claim_identifier,
            validator=validator,
        )
    return None


def validate_non_negative_decimal(
    value: Decimal | None,
    field_name: str,
    segment: str,
    position: int,
    object_index: int,
    required: bool = False,
    claim_identifier: str | None = None,
    validator: str = "",
) -> ValidationError | None:
    """Return ERROR if required and missing, WARNING if negative."""
    if value is None:
        if required:
            return ValidationError(
                segment=segment,
                field=field_name,
                message=f"{field_name} is required",
                severity=Severity.ERROR,
                position=position,
                object_index=object_index,
                claim_identifier=claim_identifier,
                validator=validator,
            )
        return None
    if value < 0:
        return ValidationError(
            segment=segment,
            field=field_name,
            message=f"{field_name} is negative ({value})",
            severity=Severity.WARNING,
            position=position,
            object_index=object_index,
            claim_identifier=claim_identifier,
            validator=validator,
        )
    return None


def validate_date_range(
    value: date | None,
    field_name: str,
    segment: str,
    position: int,
    object_index: int,
    required: bool = False,
    claim_identifier: str | None = None,
    validator: str = "",
) -> ValidationError | None:
    """Return ERROR if required and missing, WARNING if out of plausible range."""
    if value is None:
        if required:
            return ValidationError(
                segment=segment,
                field=field_name,
                message=f"{field_name} is required",
                severity=Severity.ERROR,
                position=position,
                object_index=object_index,
                claim_identifier=claim_identifier,
                validator=validator,
            )
        return None
    today = date.today()
    if value < _DATE_MIN:
        return ValidationError(
            segment=segment,
            field=field_name,
            message=f"{field_name} is before {_DATE_MIN} ({value})",
            severity=Severity.WARNING,
            position=position,
            object_index=object_index,
            claim_identifier=claim_identifier,
            validator=validator,
        )
    if value > today + _DATE_FUTURE_THRESHOLD:
        return ValidationError(
            segment=segment,
            field=field_name,
            message=f"{field_name} is more than 1 year in the future ({value})",
            severity=Severity.WARNING,
            position=position,
            object_index=object_index,
            claim_identifier=claim_identifier,
            validator=validator,
        )
    return None
