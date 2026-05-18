"""EDI X12 parser base: delimiters, context, result, and utility functions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation

from app.models.adjustment import Adjustment
from app.models.claim import Claim
from app.models.claim_line import ClaimLine
from app.models.diagnosis import Diagnosis
from app.models.edi_file import EdiFile
from app.models.enums import FileType
from app.models.raw_segment import RawSegment
from app.models.remark_code import RemarkCode
from app.models.remittance_claim import RemittanceClaim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Delimiters
# ---------------------------------------------------------------------------

@dataclass
class Delimiters:
    """EDI X12 delimiters parsed from the ISA segment."""

    element: str = "*"
    component: str = ":"
    segment: str = "~"


def detect_delimiters(raw_text: str) -> Delimiters:
    """Parse delimiters from the ISA header (always 106 fixed-width chars).

    ISA[3]   → element separator
    ISA[104] → component separator
    ISA[105] → segment terminator
    """
    # Find start of ISA
    isa_pos = raw_text.find("ISA")
    if isa_pos == -1:
        raise ValueError("No ISA segment found in EDI file")

    isa_block = raw_text[isa_pos:]
    if len(isa_block) < 106:
        raise ValueError(
            f"ISA segment too short ({len(isa_block)} chars, need 106)"
        )

    return Delimiters(
        element=isa_block[3],
        component=isa_block[104],
        segment=isa_block[105],
    )


def tokenize(raw_text: str, segment_terminator: str) -> list[str]:
    """Split raw EDI text into segment strings, stripping whitespace."""
    segments = raw_text.split(segment_terminator)
    return [seg.strip() for seg in segments if seg.strip()]


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_element(elements: list[str], index: int, default: str = "") -> str:
    """Return element at *index* or *default* if out of range / blank."""
    if index < len(elements) and elements[index].strip():
        return elements[index].strip()
    return default


def safe_decimal(value: str) -> Decimal | None:
    """Convert string to Decimal, returning None on failure."""
    value = value.strip() if value else ""
    if not value:
        return None
    try:
        return Decimal(value)
    except InvalidOperation:
        return None


def safe_date(date_str: str) -> date | None:
    """Parse CCYYMMDD string into a date, returning None on failure."""
    date_str = date_str.strip() if date_str else ""
    if not date_str or len(date_str) != 8:
        return None
    try:
        return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))
    except (ValueError, IndexError):
        return None


def split_composite(element: str, separator: str) -> list[str]:
    """Split a composite element by the component separator."""
    return element.split(separator) if element else []


# ---------------------------------------------------------------------------
# Parse context — mutable state carried through dispatch
# ---------------------------------------------------------------------------

@dataclass
class ParseContext:
    """Mutable state passed through all segment handlers during a parse."""

    edi_file: EdiFile
    file_type: FileType | None = None
    delimiters: Delimiters = field(default_factory=Delimiters)

    # Hierarchy state
    current_claim: Claim | None = None
    current_remittance_claim: RemittanceClaim | None = None
    current_line_number: int = 0
    current_diagnosis_sequence: int = 0
    last_segment_type: str = ""

    # Contextual fields (set by N1/NM1/REF, consumed by CLM/CLP)
    current_payer_name: str | None = None
    current_patient_member_id: str | None = None
    current_service_date: date | None = None
    current_remittance_date: date | None = None

    # Accumulators
    claims: list[Claim] = field(default_factory=list)
    claim_lines: list[ClaimLine] = field(default_factory=list)
    diagnoses: list[Diagnosis] = field(default_factory=list)
    remittance_claims: list[RemittanceClaim] = field(default_factory=list)
    adjustments: list[Adjustment] = field(default_factory=list)
    remark_codes: list[RemarkCode] = field(default_factory=list)
    raw_segments: list[RawSegment] = field(default_factory=list)

    # Diagnostics
    segment_position: int = 0
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Parse result — returned to callers
# ---------------------------------------------------------------------------

@dataclass
class ParseResult:
    """Summary returned after parsing an EDI file."""

    edi_file_id: int | None = None
    file_type: str = ""
    claims_count: int = 0
    claim_lines_count: int = 0
    diagnoses_count: int = 0
    remittance_claims_count: int = 0
    adjustments_count: int = 0
    remark_codes_count: int = 0
    raw_segments_count: int = 0
    errors: list[str] = field(default_factory=list)
    success: bool = False
