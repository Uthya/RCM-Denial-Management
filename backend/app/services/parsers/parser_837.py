"""837 (claim submission) dispatch logic.

Walks segments sequentially, delegating to handler functions.
Stores every segment as a RawSegment for audit.
"""

from __future__ import annotations

import logging

from app.models.raw_segment import RawSegment
from app.services.parsers.base import (
    Delimiters,
    ParseContext,
    parse_exception_to_validation_error,
    safe_element,
)
from app.services.parsers.handlers import (
    handle_clm,
    handle_dtp,
    handle_hi,
    handle_isa,
    handle_n1,
    handle_nm1,
    handle_ref,
    handle_sv1,
)

logger = logging.getLogger(__name__)

# Structural segments that update last_segment_type for DTP ownership
_STRUCTURAL_SEGMENTS = {"CLM", "SV1", "HI"}


def parse_837(
    segments: list[str],
    ctx: ParseContext,
    delimiters: Delimiters,
) -> None:
    """Dispatch 837 segments to appropriate handlers."""

    for pos, raw_seg in enumerate(segments, start=1):
        ctx.segment_position = pos
        elements = raw_seg.split(delimiters.element)
        seg_name = elements[0].upper().strip() if elements else ""

        # Archive every segment
        raw = RawSegment(
            segment_name=seg_name,
            segment_position=pos,
            raw_segment_text=raw_seg,
        )

        try:
            if seg_name == "ISA":
                handle_isa(elements, raw_seg, ctx, delimiters)

            elif seg_name == "CLM":
                handle_clm(elements, raw_seg, ctx, delimiters)
                ctx.last_segment_type = "CLM"

            elif seg_name == "SV1":
                handle_sv1(elements, raw_seg, ctx, delimiters)
                ctx.last_segment_type = "SV1"

            elif seg_name == "HI":
                handle_hi(elements, raw_seg, ctx, delimiters)
                ctx.last_segment_type = "HI"

            elif seg_name == "DTP":
                handle_dtp(elements, raw_seg, ctx, delimiters)

            elif seg_name == "N1":
                handle_n1(elements, raw_seg, ctx, delimiters)

            elif seg_name == "NM1":
                handle_nm1(elements, raw_seg, ctx, delimiters)

            elif seg_name == "REF":
                handle_ref(elements, raw_seg, ctx, delimiters)

            elif seg_name in ("GS", "GE", "ST", "SE", "IEA"):
                # Envelope segments — skip silently
                pass

            else:
                logger.info(
                    "Unhandled segment %s at position %d, skipped", seg_name, pos
                )

        except (ValueError, IndexError) as exc:
            error_msg = f"{seg_name} at pos {pos}: {exc}"
            raw.parse_error = str(exc)
            ctx.parse_errors.append(
                parse_exception_to_validation_error(
                    exc,
                    seg_name,
                    pos,
                    current_claim_number=(
                        ctx.current_claim.claim_number
                        if ctx.current_claim
                        else None
                    ),
                )
            )
            logger.error(
                "Parse error in 837: %s (raw=%s)", error_msg, raw_seg
            )

        # Associate raw segment with current claim if available
        if ctx.current_claim:
            raw._parse_claim_ref = ctx.current_claim  # type: ignore[attr-defined]

        ctx.raw_segments.append(raw)
