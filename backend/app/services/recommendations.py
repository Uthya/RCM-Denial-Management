"""Recommendation generator.

Turns reasons (parser findings, CARC codes, ML risk factors) into imperative
fix sentences. Pure functions — no DB access here. The router composes these
into a per-claim list.
"""

from __future__ import annotations

from typing import Iterable, Literal


# ---------------------------------------------------------------------------
# Lookup tables (server-side source of truth)
# ---------------------------------------------------------------------------

CARC_DESCRIPTIONS: dict[str, str] = {
    "1": "Deductible amount.",
    "2": "Coinsurance amount.",
    "3": "Co-payment amount.",
    "4": "The procedure code is inconsistent with the modifier used.",
    "5": "The procedure code or bill type is inconsistent with the place of service.",
    "6": "The procedure or revenue code is inconsistent with the patient's age.",
    "9": "The diagnosis is inconsistent with the patient's age.",
    "10": "The diagnosis is inconsistent with the patient's gender.",
    "11": "The diagnosis is inconsistent with the procedure.",
    "15": "Authorization number is missing, invalid, or does not apply.",
    "16": "Claim/service lacks information or has submission/billing error(s).",
    "17": "Requested information was not provided or was insufficient/incomplete.",
    "18": "Exact duplicate claim/service.",
    "22": "Care may be covered by another payer per coordination of benefits.",
    "23": "Impact of prior payer(s) adjudication.",
    "24": "Charges are covered under a capitation agreement / managed care plan.",
    "26": "Expenses incurred prior to coverage.",
    "27": "Expenses incurred after coverage terminated.",
    "29": "Time limit for filing has expired.",
    "31": "Patient cannot be identified as our insured.",
    "32": "Our records indicate that this dependent is not an eligible dependent as defined in the plan.",
    "38": "Services not provided or authorized by designated (network/primary-care) providers.",
    "45": "Charge exceeds fee schedule / maximum allowable.",
    "50": "Non-covered service: not deemed a medical necessity.",
    "54": "Multiple physicians/assistants are not covered in this case.",
    "55": "Procedure/treatment/drug is deemed experimental or investigational by the payer.",
    "58": "Treatment was deemed by the utilization-review organization to have been rendered in an inappropriate or invalid place of service.",
    "95": "Plan procedures not followed.",
    "96": "Non-covered charge(s).",
    "97": "Service is included in another service already adjudicated.",
    "109": "Claim/service not covered by this payer/contractor.",
    "110": "Billing date predates service date.",
    "119": "Benefit maximum for this period or occurrence has been reached.",
    "125": "Submission/billing error(s).",
    "146": "Diagnosis was invalid for the date(s) of service reported.",
    "151": "Payer deems the information submitted does not support this many/frequency of services.",
    "165": "Referral absent or exceeded.",
    "167": "Diagnosis is not covered.",
    "170": "Payment denied when performed/billed by this type of provider.",
    "178": "Patient has not met the required spend-down requirements.",
    "181": "Procedure code was invalid on the date of service.",
    "182": "Procedure modifier was invalid on the date of service.",
    "185": "The rendering provider is not eligible to perform the service billed.",
    "197": "Precertification / authorization / notification absent.",
    "198": "Precertification / authorization exceeded.",
    "204": "Service/equipment/drug is not covered under the patient's plan.",
    "226": "Information requested from the billing/rendering provider was not provided or was insufficient.",
    "227": "Information requested from the patient/insured/responsible party was not provided or was insufficient.",
    "234": "This procedure is not paid separately.",
    "242": "Services not provided by network/primary-care providers.",
    "252": "An attachment / other documentation is required to adjudicate this claim/service.",
    "254": "Claim received by the dental plan, but benefits not available under this plan.",
    "256": "Service not payable per managed-care contract.",
    "273": "Coverage / program guidelines were exceeded.",
}

CARC_FIXES: dict[str, str] = {
    "1": "Verify the deductible amount with the payer; bill the remaining balance to the patient.",
    "2": "Confirm the coinsurance terms and bill the patient for the indicated portion.",
    "3": "Bill the patient for the co-payment portion.",
    "4": "Review the SV1 modifier(s); correct or remove the inconsistent modifier and resubmit.",
    "5": "Verify the place-of-service code (SV1) is appropriate for the procedure and bill type; correct and resubmit.",
    "6": "Confirm the patient's age supports this procedure/revenue code; correct the code or attach documentation justifying the exception.",
    "9": "Verify the patient's date of birth and that the diagnosis matches their age; correct demographics or diagnosis and resubmit.",
    "10": "Verify the patient's gender on file matches; correct demographics or replace with a gender-appropriate diagnosis and resubmit.",
    "11": "Review the diagnosis-procedure combination for medical necessity and resubmit with a compatible diagnosis code.",
    "15": "Obtain authorization from the payer and resubmit with the correct authorization number in the REF segment.",
    "16": "Inspect the paired RARC code for the missing field, complete it, and resubmit a corrected claim.",
    "17": "Provide the missing or incomplete information the payer requested and resubmit.",
    "18": "Check claim history before resubmitting — only resubmit if a correction is required, not as a duplicate.",
    "22": "Submit to the primary payer first, then bill the secondary payer with the prior-payer EOB attached.",
    "23": "Adjust the claim based on the prior payer's payment information and resubmit.",
    "24": "Verify the capitation agreement; do not resubmit if the service falls under the contract.",
    "26": "Verify the patient's coverage start date; if the service truly preceded coverage, bill the patient directly.",
    "27": "Check patient eligibility for the date of service; if not covered, bill the patient directly.",
    "29": "Submit a corrected claim within the payer's timely-filing window and include proof of original submission.",
    "31": "Verify the patient's member ID, name, and date of birth against the payer's records; correct demographics and resubmit.",
    "32": "Verify the dependent's eligibility (relationship, age, student/disability status) with the payer; if ineligible, bill the patient or correct enrollment.",
    "38": "Confirm the rendering provider is in-network for this payer; route through the designated provider or appeal with referral documentation.",
    "45": "Adjust the billed amount to match the payer's allowable fee schedule and resubmit.",
    "50": "Attach medical-necessity documentation (notes, lab results, prior auth) and submit an appeal.",
    "54": "Remove the additional physician/assistant from the claim or resubmit with only the covered provider.",
    "55": "Submit medical-necessity documentation and peer-reviewed literature with an appeal; otherwise bill the patient if non-covered.",
    "58": "Verify the place-of-service is appropriate for this procedure under utilization-review guidelines; correct POS or appeal with documentation.",
    "95": "Review the payer's required procedures (referrals, prior auth, network rules); satisfy and resubmit.",
    "96": "Verify coverage with the payer; if the service is non-covered, bill the patient or write off.",
    "97": "Remove the bundled service from the claim — do not resubmit it as a separate line item.",
    "109": "Confirm the correct payer/contractor for this claim and resubmit to the appropriate party.",
    "110": "Correct the billing date so it does not precede the service date and resubmit.",
    "119": "Verify the benefit period; do not resubmit until next plan year or appeal with documentation.",
    "125": "Review the claim for submission errors (provider IDs, codes, dates) and resubmit a corrected claim.",
    "146": "Verify the diagnosis code was valid on the date of service; replace with a code valid for that DOS and resubmit.",
    "151": "Reduce the number/frequency of services to what the documentation supports, or attach justification and appeal.",
    "165": "Obtain a valid referral from the primary-care provider; resubmit with the referral number in the REF segment.",
    "167": "Replace with a covered diagnosis code or submit medical-necessity documentation with an appeal.",
    "170": "Confirm the provider type is authorized to bill this service; route through an eligible provider or appeal.",
    "178": "Confirm the patient's spend-down has been met before resubmitting; otherwise bill the patient.",
    "181": "Verify the CPT/HCPCS code was active on the date of service; replace with a valid code and resubmit.",
    "182": "Verify the modifier was active on the date of service; replace with a valid modifier and resubmit.",
    "185": "Confirm the rendering provider's credentialing/enrollment for this service; route through an eligible provider or update enrollment.",
    "197": "Obtain prior authorization and resubmit with the authorization number in the REF segment.",
    "198": "Confirm the authorized units/visits and resubmit only the authorized portion.",
    "204": "Verify the patient's plan; if the service is not covered, bill the patient or use a covered alternative.",
    "226": "Provide the requested information to the payer (records, attachments, certifications) and resubmit.",
    "227": "Coordinate with the patient/insured to provide the requested information and resubmit once received.",
    "234": "This service is bundled and not paid separately — remove the line or accept the adjustment; do not resubmit as a standalone service.",
    "242": "Route the claim through a network/primary-care provider, or appeal with referral/out-of-network justification.",
    "252": "Attach the supporting documentation the payer requires (medical records, operative report, etc.) and resubmit.",
    "254": "Verify the patient's dental plan covers this benefit; if not, bill the patient or route to the correct plan.",
    "256": "Review the managed-care contract terms; do not resubmit, or appeal with contract-specific justification.",
    "273": "Confirm coverage/program guidelines; reduce the claim to within the allowed limits or appeal with justification.",
}

SEGMENT_LABELS: dict[str, str] = {
    "CLM": "claim header",
    "SV1": "service-line",
    "SV2": "institutional service-line",
    "SV3": "dental service-line",
    "DTP": "date / time period",
    "NM1": "name / entity",
    "HI": "diagnosis code",
    "REF": "reference identifier",
    "SBR": "subscriber",
    "HL": "hierarchical level",
    "ISA": "interchange envelope",
    "GS": "functional group",
    "ST": "transaction set",
    "CAS": "claim adjustment",
    "AMT": "monetary amount",
    "CLP": "claim payment",
    "PER": "contact information",
    "PRV": "provider",
    "N3": "address line",
    "N4": "city / state / postal code",
    "LX": "line counter",
    "LIN": "item identification",
}

# Keys MUST match the human display names the predictor emits (see
# FEATURE_DISPLAY_NAMES in app/ml/predictor.py) so a flagged factor maps
# directly to an actionable hint here.
ML_FEATURE_HINTS: dict[str, str] = {
    "Procedure Code": "Verify the primary CPT/HCPCS code — the model finds it atypical for this payer/POS; ensure documentation supports it.",
    "Diagnosis Code": "Review the primary diagnosis — it is associated with higher denial risk for this procedure; consider a more specific code or stronger medical-necessity documentation.",
    "Payer": "Submissions to this payer historically show elevated denial rates — double-check payer-specific filing requirements before resubmission.",
    "Charge Amount": "The total charge is unusual for this service-line mix — verify line totals and fee-schedule alignment.",
    "Billed Amount": "The aggregated billed amount is outside the typical range — verify each SV1 billed_amount.",
    "Service Units": "The total service units are higher than typical — verify unit counts on each service line.",
    "Line Count": "The number of service lines is unusual for this claim type — confirm each line is necessary and properly supported.",
    "Diagnosis Count": "The diagnosis count is atypical — confirm each diagnosis is documented and contributes to medical necessity.",
    "Modifier Present": "Modifier usage on this claim contributes to elevated risk — verify each modifier is appropriate for the procedure and payer policy.",
    "Facility Type": "The facility type contributes to elevated risk — verify the bill type matches the service rendered.",
    "Claim Frequency": "The claim frequency code (original/replacement/void) flagged elevated risk — confirm it matches the intended submission type.",
    "Place of Service": "The place-of-service code is atypical for this procedure — verify the POS code is appropriate.",
    "Service Duration": "Service date range is atypical — verify the from/to dates and that coverage applied throughout.",
    "Service Month": "Service month influenced risk — verify date accuracy and confirm patient eligibility at the time of service.",
    "Day of Week": "Service day-of-week influenced risk — verify the service date is correct.",
    "Weekend Service": "Weekend service flagged risk — confirm payer policy covers weekend services for this procedure.",
    "Multiple Service Lines": "Multiple service lines contributed to risk — verify each line is necessary and not duplicative.",
    "High Diagnosis Count": "High diagnosis count flagged risk — confirm each diagnosis is supported by documentation.",
    "High Charge Amount": "Charge amount is high relative to comparable claims — verify amounts and consider attaching supporting documentation.",
    "Missing Payer": "Payer information is missing or incomplete — populate the NM1*PR segment before resubmission.",
    "Missing Diagnosis": "No diagnosis code present — add at least one valid ICD-10 code in the HI segment.",
    "Missing Procedure": "No procedure code present — add a valid CPT/HCPCS code in the SV1 segment.",
    "Missing Place of Service": "Place-of-service code is missing — populate POS in the SV1 segment.",
    "Payer × Procedure History": "Historically, this payer denies this procedure code at an elevated rate — review payer policy and documentation before resubmission.",
    "Payer × Diagnosis History": "Historically, this payer denies this diagnosis at an elevated rate — review payer policy and medical necessity.",
    "Payer × Place of Service History": "Historically, this payer denies this place of service at an elevated rate — verify POS is appropriate and supported.",
    "Payer Claim Volume": "Low historical volume with this payer — verify enrollment, payer ID, and filing channel are correct.",
    "Procedure Claim Volume": "Low historical volume for this procedure — confirm the CPT/HCPCS is correct and supported.",
    "Diagnosis Claim Volume": "Low historical volume for this diagnosis — confirm the ICD-10 is correct and specific.",
    "Rare Payer": "Rare payer for this practice — confirm payer enrollment and filing setup.",
    "Rare Procedure": "Rare procedure code — confirm code accuracy and that documentation supports it.",
}

ADJUSTMENT_GROUP_LABELS: dict[str, str] = {
    "CO": "Contractual Obligation",
    "PR": "Patient Responsibility",
    "OA": "Other Adjustment",
    "PI": "Payer-Initiated Reduction",
    "CR": "Correction & Reversal",
}


# Status code groupings (CLP02 from 835)
DENIED_CLP02 = {"4"}
PAID_CLP02 = {"1", "2", "3", "19", "20"}


# ---------------------------------------------------------------------------
# Generators (one per source)
# ---------------------------------------------------------------------------

Source = Literal["parser", "carc", "model"]


def _parser_fix_from_message(segment: str, field: str | None, message: str) -> str:
    """Turn a parser validation message into an imperative fix sentence."""
    msg = (message or "").lower()
    seg_label = SEGMENT_LABELS.get(segment, segment)

    if not field:
        return f"Review the {seg_label} segment and correct the reported issue before resubmission."

    if "required" in msg or "missing" in msg:
        return f"Add a valid {field} value in the {segment} segment before resubmission."
    if "invalid" in msg:
        return f"Correct the invalid {field} value in the {segment} segment."
    if "duplicate" in msg:
        return f"Resolve the duplicate {field} entry in the {segment} segment before resubmitting."
    if "orphan" in msg:
        return f"Reattach the orphan {segment} segment to its parent claim before resubmitting."
    if "zero" in msg or "non-negative" in msg or "negative" in msg:
        return f"Correct the {field} value in the {segment} segment to a valid non-negative amount."
    if "exceed" in msg:
        return f"Recalculate the {field} value in the {segment} segment to a valid amount."
    return f"Review and correct the {field} value in the {segment} segment."


def recommendation_from_parser_finding(
    segment: str,
    field: str | None,
    message: str,
) -> dict:
    """Build a parser-sourced recommendation dict."""
    location = f"{segment} segment" + (f", field {field}" if field else "")
    return {
        "reason": message or f"{segment} validation issue",
        "fix": _parser_fix_from_message(segment, field, message),
        "source": "parser",
        "location": location,
    }


def recommendation_from_carc(
    reason_code: str,
    group_code: str | None = None,
    paired_rarc: str | None = None,
) -> dict:
    """Build a CARC-sourced recommendation dict."""
    description = CARC_DESCRIPTIONS.get(
        reason_code, f"Adjustment reason CARC {reason_code} (no description on file)."
    )
    fix = CARC_FIXES.get(
        reason_code,
        "Review the adjustment reason with the payer for clarification before resubmission.",
    )
    label = f"{group_code}-{reason_code}" if group_code else f"CARC {reason_code}"
    location = f"{ADJUSTMENT_GROUP_LABELS.get(group_code, 'Adjustment')} group" if group_code else "Adjustment"
    reason_text = f"{label}: {description}"
    if paired_rarc:
        reason_text += f" (paired RARC: {paired_rarc})"
    return {
        "reason": reason_text,
        "fix": fix,
        "source": "carc",
        "location": location,
    }


def recommendation_from_ml_factor(
    feature: str,
    direction: str,
    impact: float | str | None = None,
) -> dict | None:
    """Build an ML-sourced recommendation dict. Returns None for protective factors."""
    if direction != "risk":
        return None

    # The predictor already emits human-readable display names (e.g. "Procedure
    # Code"), so use them verbatim. Fall back to a snake_case-to-space conversion
    # only if an unknown raw column slips through.
    readable = feature if " " in feature else feature.replace("_", " ").title()
    hint = ML_FEATURE_HINTS.get(feature) or ML_FEATURE_HINTS.get(readable)
    if not hint:
        hint = (
            f"Review the '{readable}' field on this claim — the model identified "
            "it as a contributor to elevated denial risk."
        )

    impact_part = f" (impact {impact})" if impact else ""
    return {
        "reason": f"Model flagged '{readable}' as elevated-risk contributor{impact_part}",
        "fix": hint,
        "source": "model",
        "location": "ML prediction",
    }


# ---------------------------------------------------------------------------
# Composers
# ---------------------------------------------------------------------------

def build_recommendations_for_claim(
    *,
    parser_findings: Iterable[dict] | None = None,
    carc_entries: Iterable[dict] | None = None,
    ml_factors: Iterable[dict] | None = None,
    max_ml_recs: int = 3,
) -> list[dict]:
    """Compose recommendations from all three sources, in priority order.

    Each input is an iterable of dicts with the relevant keys:
    - parser_findings: {segment, field, message}
    - carc_entries:    {group_code, reason_code, paired_rarc?}
    - ml_factors:      {feature, direction, impact}

    Duplicate entries within a source (e.g. the same CARC code applied to
    multiple service lines) collapse to a single recommendation.
    """
    out: list[dict] = []
    seen_parser: set[tuple] = set()
    seen_carc: set[tuple] = set()
    seen_ml: set[str] = set()

    # 1. Parser findings — highest priority (deterministic, mechanical fix)
    for finding in parser_findings or ():
        segment = finding.get("segment", "?")
        field = finding.get("field")
        message = finding.get("message", "")
        key = (segment, field or "", message)
        if key in seen_parser:
            continue
        seen_parser.add(key)
        out.append(
            recommendation_from_parser_finding(
                segment=segment, field=field, message=message
            )
        )

    # 2. CARC entries — payer-authoritative; one rec per (group, reason)
    for entry in carc_entries or ():
        reason_code = entry["reason_code"]
        group_code = entry.get("group_code")
        key = (group_code or "", reason_code)
        if key in seen_carc:
            continue
        seen_carc.add(key)
        out.append(
            recommendation_from_carc(
                reason_code=reason_code,
                group_code=group_code,
                paired_rarc=entry.get("paired_rarc"),
            )
        )

    # 3. ML factors — capped at max_ml_recs (avoid clutter from low-impact features)
    ml_added = 0
    for f in ml_factors or ():
        if ml_added >= max_ml_recs:
            break
        feature = f.get("feature", "")
        if feature in seen_ml:
            continue
        rec = recommendation_from_ml_factor(
            feature=feature,
            direction=f.get("direction", ""),
            impact=f.get("impact"),
        )
        if rec is not None:
            seen_ml.add(feature)
            out.append(rec)
            ml_added += 1

    return out
