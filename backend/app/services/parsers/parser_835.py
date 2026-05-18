"""835 (remittance/payment) dispatch logic.

Walks segments sequentially, delegating to handler functions.
Stores every segment as a RawSegment for audit.
"""

from __future__ import annotations

import logging

from app.models.raw_segment import RawSegment
from app.services.parsers.base import Delimiters, ParseContext
from app.services.parsers.handlers import (
    handle_cas,
    handle_clp,
    handle_dtm,
    handle_isa,
    handle_lq,
    handle_nm1,
    handle_ref,
)

logger = logging.getLogger(__name__)


def parse_835(
    segments: list[str],
    ctx: ParseContext,
    delimiters: Delimiters,
) -> None:
    """Dispatch 835 segments to appropriate handlers."""

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

            elif seg_name == "CLP":
                handle_clp(elements, raw_seg, ctx, delimiters)
                ctx.last_segment_type = "CLP"

            elif seg_name == "CAS":
                handle_cas(elements, raw_seg, ctx, delimiters)

            elif seg_name == "LQ":
                handle_lq(elements, raw_seg, ctx, delimiters)

            elif seg_name == "DTM":
                handle_dtm(elements, raw_seg, ctx, delimiters)

            elif seg_name == "NM1":
                handle_nm1(elements, raw_seg, ctx, delimiters)

            elif seg_name == "REF":
                handle_ref(elements, raw_seg, ctx, delimiters)

            elif seg_name in ("GS", "GE", "ST", "SE", "IEA", "N1", "PLB",
                              "BPR", "TRN", "TS3", "TS2"):
                # Envelope / payment-level segments — skip silently
                pass

            else:
                logger.info(
                    "Unhandled segment %s at position %d, skipped", seg_name, pos
                )

        except (ValueError, IndexError) as exc:
            error_msg = f"{seg_name} at pos {pos}: {exc}"
            raw.parse_error = str(exc)
            ctx.errors.append(error_msg)
            logger.error(
                "Parse error in 835: %s (raw=%s)", error_msg, raw_seg
            )

        # Associate raw segment with current remittance claim's claim ref
        if ctx.current_remittance_claim:
            raw._parse_rc_ref = ctx.current_remittance_claim  # type: ignore[attr-defined]

        ctx.raw_segments.append(raw)
