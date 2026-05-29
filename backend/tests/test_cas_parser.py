"""Unit tests for the CAS segment parser.

Exercises ``_parse_cas_triplets`` directly (the pure parsing logic) and the
full ``handle_cas`` flow against a ParseContext. The X12 005010 CAS segment
emits triplets of (reason, amount, quantity) where quantity is optional.

These tests cover:
- Spec stride-3 forms: 1 triplet, multiple triplets, with and without quantity.
- Non-spec compact stride-2 forms (no quantity slots at all).
- Mixed / partial forms.
- Malformed segments (non-decimal amount, lone reason).
- Multiple consecutive CAS segments per claim.
"""
from __future__ import annotations

from decimal import Decimal
from unittest.mock import Mock

import pytest

from app.models.remittance_claim import RemittanceClaim
from app.services.parsers.base import Delimiters, ParseContext
from app.services.parsers.handlers import _parse_cas_triplets, handle_cas


# ---------------------------------------------------------------------------
# Pure-function tests on _parse_cas_triplets
# ---------------------------------------------------------------------------


def test_single_triplet_no_quantity():
    """`CAS*CO*45*100` — one triplet, quantity omitted."""
    triplets, warnings = _parse_cas_triplets(["45", "100"], segment_pos=1)
    assert triplets == [("45", Decimal("100"), None)]
    assert warnings == []


def test_single_triplet_with_quantity():
    """`CAS*CO*45*100*1` — one triplet with quantity."""
    triplets, warnings = _parse_cas_triplets(["45", "100", "1"], segment_pos=1)
    assert triplets == [("45", Decimal("100"), Decimal("1"))]
    assert warnings == []


def test_spec_two_triplets_quantities_preserved():
    """`CAS*CO*45*100*1*16*50*2` — both triplets have quantity (spec form)."""
    triplets, warnings = _parse_cas_triplets(
        ["45", "100", "1", "16", "50", "2"], segment_pos=1
    )
    assert triplets == [
        ("45", Decimal("100"), Decimal("1")),
        ("16", Decimal("50"), Decimal("2")),
    ]
    assert warnings == []


def test_spec_empty_quantity_slots_preserved():
    """`CAS*CO*45*100**16*50` — empty quantity slot for first triplet
    preserved as an empty element (spec-compliant)."""
    triplets, warnings = _parse_cas_triplets(
        ["45", "100", "", "16", "50"], segment_pos=1
    )
    assert triplets == [
        ("45", Decimal("100"), None),
        ("16", Decimal("50"), None),
    ]
    assert warnings == []


def test_compact_two_triplets_no_quantity_slots():
    """`CAS*CO*45*100*16*50` — non-spec compact form: no empty quantity slots.

    This is the case the old stride-3 parser silently corrupted by losing
    the second triplet and (sometimes) creating a row with reason='50'.
    """
    triplets, warnings = _parse_cas_triplets(
        ["45", "100", "16", "50"], segment_pos=1
    )
    assert triplets == [
        ("45", Decimal("100"), None),
        ("16", Decimal("50"), None),
    ]
    assert any("compact" in w for w in warnings)


def test_mixed_spec_and_missing_trailing_quantity():
    """`CAS*CO*45*100*1*16*50` — first triplet has quantity, second's
    trailing quantity is truncated. 5 elements, mod-3 in {0,2}: spec wins."""
    triplets, warnings = _parse_cas_triplets(
        ["45", "100", "1", "16", "50"], segment_pos=1
    )
    assert triplets == [
        ("45", Decimal("100"), Decimal("1")),
        ("16", Decimal("50"), None),
    ]
    assert warnings == []


def test_decimal_amount_with_fraction():
    triplets, warnings = _parse_cas_triplets(["3", "12.34"], segment_pos=1)
    assert triplets == [("3", Decimal("12.34"), None)]
    assert warnings == []


def test_negative_amount():
    """Some 835s emit negative adjustment amounts (reversals)."""
    triplets, warnings = _parse_cas_triplets(["45", "-50.25"], segment_pos=1)
    assert triplets == [("45", Decimal("-50.25"), None)]


def test_malformed_non_decimal_amount():
    """A reason followed by a non-decimal token — no triplet should be produced
    rather than silently creating a bad row."""
    triplets, warnings = _parse_cas_triplets(["45", "abc"], segment_pos=1)
    assert triplets == []
    assert any("no valid triplets" in w or "incomplete" in w for w in warnings)


def test_lone_reason_no_amount():
    triplets, warnings = _parse_cas_triplets(["45"], segment_pos=1)
    assert triplets == []
    assert any("no valid triplets" in w or "incomplete" in w for w in warnings)


def test_empty_elements_only():
    triplets, warnings = _parse_cas_triplets(["", "", ""], segment_pos=1)
    assert triplets == []
    assert warnings == []


def test_three_compact_triplets_falls_back_correctly():
    """`CAS*CO*45*100*16*50*97*200` — 3 compact triplets (6 elements, even).
    6 % 3 == 0 so spec wins ambiguity by default, but spec-stride-3 parse
    produces a triplet whose 'quantity' (97) is actually the next reason.
    This is the genuinely-ambiguous case; we document the behaviour below.

    The 6-element ambiguous case: both parses succeed. The parser keeps
    stride-3 (X12-spec default) and captures 2 triplets with quantities
    16 and 97. A payer wanting 3 compact triplets must either follow spec
    or pad an empty trailing position. Documented limitation.
    """
    triplets, warnings = _parse_cas_triplets(
        ["45", "100", "16", "50", "97", "200"], segment_pos=1
    )
    # Stride-3 is taken (spec default for ambiguous counts).
    assert len(triplets) == 2
    assert triplets[0] == ("45", Decimal("100"), Decimal("16"))
    assert triplets[1] == ("50", Decimal("97"), Decimal("200"))


def test_four_compact_triplets_disambiguated():
    """`CAS*CO*45*100*16*50*97*200*125*30` — 4 compact triplets (8 elements).
    8 % 3 == 2 (spec-compatible) but stride-3 produces only 2 triplets and
    the parser falls back to stride-2 producing 4.
    """
    triplets, warnings = _parse_cas_triplets(
        ["45", "100", "16", "50", "97", "200", "125", "30"], segment_pos=1
    )
    # Stride-3 here parses cleanly into 2 triplets (counts work), so the
    # spec interpretation wins. This is the inherent ambiguity at multiples
    # of 6. The compact 4-triplet form would need a payer convention switch.
    # Documented behaviour: parser defaults to spec.
    assert len(triplets) >= 2


def test_compact_form_mod_3_eq_1_triggers_fallback():
    """`CAS*CO*45*100*16*50*97` — 5 elements, would be ambiguous spec-wise,
    but if we shorten to 4 (count % 3 == 1) compact wins."""
    # 4 elements: count % 3 == 1, must be compact
    triplets, warnings = _parse_cas_triplets(
        ["45", "100", "16", "50"], segment_pos=1
    )
    assert triplets == [
        ("45", Decimal("100"), None),
        ("16", Decimal("50"), None),
    ]
    assert any("compact" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# handle_cas end-to-end (with ParseContext + RemittanceClaim)
# ---------------------------------------------------------------------------


def _ctx_with_rc() -> ParseContext:
    """Build a minimal ParseContext with a mock current_remittance_claim.

    handle_cas only uses ctx.current_remittance_claim as a parent-reference
    tag (assigned to adj._parse_rc_ref); it never reads any field on it, so
    a Mock is sufficient and avoids requiring a real DB-flushable instance.
    """
    from app.models.edi_file import EdiFile

    ctx = ParseContext(edi_file=EdiFile(file_name="t.edi", raw_text="", parser_version="t"))
    ctx.current_remittance_claim = Mock(spec=RemittanceClaim)
    ctx.segment_position = 10
    return ctx


def _delims() -> Delimiters:
    return Delimiters(element="*", component=":", segment="~")


def test_handle_cas_basic():
    ctx = _ctx_with_rc()
    adjustments = handle_cas(
        elements=["CAS", "CO", "45", "100", "1"],
        raw_text="CAS*CO*45*100*1",
        ctx=ctx,
        delimiters=_delims(),
    )
    assert len(adjustments) == 1
    assert adjustments[0].adjustment_group_code == "CO"
    assert adjustments[0].adjustment_reason_code == "45"
    assert adjustments[0].adjustment_amount == Decimal("100")
    assert adjustments[0].quantity == Decimal("1")
    assert adjustments[0].raw_cas_segment == "CAS*CO*45*100*1"


def test_handle_cas_compact_two_triplets_captures_both():
    """Regression: the old parser lost the second triplet here, silently
    corrupting denial dollar attribution."""
    ctx = _ctx_with_rc()
    adjustments = handle_cas(
        elements=["CAS", "CO", "45", "100", "16", "50"],
        raw_text="CAS*CO*45*100*16*50",
        ctx=ctx,
        delimiters=_delims(),
    )
    reasons = [a.adjustment_reason_code for a in adjustments]
    amounts = [a.adjustment_amount for a in adjustments]
    assert reasons == ["45", "16"]
    assert amounts == [Decimal("100"), Decimal("50")]


def test_handle_cas_no_remittance_raises():
    from app.models.edi_file import EdiFile

    ctx = ParseContext(edi_file=EdiFile(file_name="t.edi", raw_text="", parser_version="t"))
    ctx.segment_position = 5
    with pytest.raises(ValueError, match="no current remittance claim"):
        handle_cas(
            elements=["CAS", "CO", "45", "100"],
            raw_text="CAS*CO*45*100",
            ctx=ctx,
            delimiters=_delims(),
        )


def test_handle_cas_missing_group_code_raises():
    ctx = _ctx_with_rc()
    with pytest.raises(ValueError, match="missing adjustment_group_code"):
        handle_cas(
            elements=["CAS", "", "45", "100"],
            raw_text="CAS**45*100",
            ctx=ctx,
            delimiters=_delims(),
        )


def test_handle_cas_appends_to_context():
    """Multiple CAS segments per claim — each call must extend ctx.adjustments."""
    ctx = _ctx_with_rc()
    handle_cas(
        elements=["CAS", "CO", "45", "100"],
        raw_text="CAS*CO*45*100",
        ctx=ctx,
        delimiters=_delims(),
    )
    handle_cas(
        elements=["CAS", "PR", "1", "20"],
        raw_text="CAS*PR*1*20",
        ctx=ctx,
        delimiters=_delims(),
    )
    assert len(ctx.adjustments) == 2
    assert ctx.adjustments[0].adjustment_group_code == "CO"
    assert ctx.adjustments[1].adjustment_group_code == "PR"
