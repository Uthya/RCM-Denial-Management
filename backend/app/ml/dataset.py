"""Dataset construction for claim-level denial prediction.

Queries the database, applies labeling rules, and produces a labeled
pandas DataFrame ready for feature engineering and model training.

v5.1 (perf): replaced the ORM-iteration build with a single denormalized
SQL query. The previous implementation pulled every Claim + selectinloaded
its claim_lines / diagnoses / remittance_claims, then walked the result
in Python to derive label, aggregates, and "primary" picks. That was 291s
on the 672k-row dataset locally. The bulk SQL version below does all the
aggregation in Postgres and returns the final row shape directly —
benchmarks show ~10-15x speedup with identical output schema and values.
"""

from decimal import Decimal

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# CLP02 status codes from 835 remittance segments.
_DENIED_CLP02 = {"4"}
_PAID_CLP02 = {"1", "2", "3", "19", "20"}


def _label_from_remittances(remittance_claims: list) -> int | None:
    """First-pass denial label derived from remittance CLP02 codes.

    Retained for callers (e.g., prediction_reconciler) that still operate on
    ORM-loaded remittance lists. The bulk SQL builder below replicates this
    rule set-theoretically with ``bool_or(...)``.

    If ANY remittance_claim has CLP02="4" the original submission was denied,
    even if a later resubmission was paid. Returns 1 for denied, 0 for paid,
    None when no actionable CLP02 code is present.
    """
    if not remittance_claims:
        return None
    codes = {rc.claim_status_code for rc in remittance_claims}
    if codes & _DENIED_CLP02:
        return 1  # denied on first pass
    if codes & _PAID_CLP02:
        return 0  # paid on first pass
    return None


# Bulk SQL — replicates the labeling rule, "primary" picks, and per-claim
# aggregates in Postgres so we don't drag millions of rows into Python.
#
# Behaviour preserved from the ORM implementation:
# - Only originals (frequency_code IS NULL or '1'). [WHERE in CTE]
# - Must have at least one adjudicating remit. [JOIN remittance_claims]
# - Label: 1 if any CLP02='4'; 0 if any CLP02 in (1,2,3,19,20) AND no '4';
#   otherwise the claim is excluded (HAVING ... + WHERE denied IS NOT NULL).
# - Primary procedure / DX / POS: the value with the lowest line_number /
#   sequence_number, matching ``sorted(...)[0]`` in the original Python.
# - line_count / diagnosis_count: row counts (zero when no related rows).
# - total_billed_amount / total_units: sums cast to float (matches the
#   ``float(sum(..., Decimal(0)))`` original).
# - has_modifier: any modifier1 IS NOT NULL (matches the ``any(...)``).
# - submission_date: ``claim.created_at::date`` (matches ``claim.created_at.date()``).
#
# Optional ``:since_service_date`` filter mirrors the kwarg on the Python signature.
_BUILD_DATASET_SQL = text("""
    WITH labeled AS (
        SELECT
            c.id AS claim_id,
            c.claim_number,
            CASE
                WHEN bool_or(rc.claim_status_code = '4') THEN 1
                WHEN bool_or(rc.claim_status_code IN ('1','2','3','19','20')) THEN 0
                ELSE NULL
            END AS denied
        FROM claims c
        JOIN remittance_claims rc ON rc.claim_id = c.id
        WHERE (c.frequency_code IS NULL OR c.frequency_code = '1')
          AND (CAST(:since_service_date AS DATE) IS NULL
               OR c.service_from_date >= CAST(:since_service_date AS DATE))
        GROUP BY c.id, c.claim_number
    ),
    line_agg AS (
        SELECT
            cl.claim_id,
            count(*)::int AS line_count,
            sum(cl.billed_amount)::float AS total_billed_amount,
            sum(cl.units)::float AS total_units,
            bool_or(cl.modifier1 IS NOT NULL) AS has_modifier,
            (array_agg(cl.procedure_code  ORDER BY cl.line_number))[1] AS primary_procedure_code,
            (array_agg(cl.place_of_service ORDER BY cl.line_number))[1] AS place_of_service
        FROM claim_lines cl
        GROUP BY cl.claim_id
    ),
    diag_agg AS (
        SELECT
            d.claim_id,
            count(*)::int AS diagnosis_count,
            (array_agg(d.diagnosis_code ORDER BY d.sequence_number))[1] AS primary_diagnosis_code
        FROM diagnoses d
        GROUP BY d.claim_id
    )
    SELECT
        labeled.claim_id,
        labeled.claim_number,
        labeled.denied,
        c.total_charge_amount::float AS total_charge_amount,
        c.payer_name,
        c.facility_type_code,
        c.frequency_code,
        c.service_from_date,
        c.service_to_date,
        COALESCE(line_agg.line_count, 0) AS line_count,
        COALESCE(diag_agg.diagnosis_count, 0) AS diagnosis_count,
        diag_agg.primary_diagnosis_code,
        line_agg.primary_procedure_code,
        COALESCE(line_agg.total_billed_amount, 0.0) AS total_billed_amount,
        COALESCE(line_agg.total_units, 0.0) AS total_units,
        COALESCE(line_agg.has_modifier, FALSE) AS has_modifier,
        line_agg.place_of_service,
        c.authorization_number,
        c.referral_number,
        c.billing_provider_npi,
        c.rendering_provider_npi,
        c.created_at::date AS submission_date
    FROM labeled
    JOIN claims c ON c.id = labeled.claim_id
    LEFT JOIN line_agg ON line_agg.claim_id = labeled.claim_id
    LEFT JOIN diag_agg ON diag_agg.claim_id = labeled.claim_id
    WHERE labeled.denied IS NOT NULL
""")


# Column order matches the legacy DataFrame so downstream code (feature
# engineering, dataset stats, serialization) sees the same shape.
_DATASET_COLUMNS: tuple[str, ...] = (
    "claim_id", "claim_number", "denied",
    "total_charge_amount", "payer_name", "facility_type_code",
    "frequency_code", "service_from_date", "service_to_date",
    "line_count", "diagnosis_count",
    "primary_diagnosis_code", "primary_procedure_code",
    "total_billed_amount", "total_units", "has_modifier",
    "place_of_service",
    "authorization_number", "referral_number",
    "billing_provider_npi", "rendering_provider_npi",
    "submission_date",
)


async def build_dataset(
    db: AsyncSession,
    since_service_date=None,
) -> pd.DataFrame:
    """Query the DB and return a labeled DataFrame for ML training.

    Only includes claims that:
    - Are original submissions (frequency_code IS NULL or '1')
    - Have at least one remittance_claim (proof of adjudication)
    - Optionally: have ``service_from_date >= since_service_date``
      (used by the drift endpoint to scope the pull to a recent window
      rather than reading the whole table over a WAN link).

    Labels are derived from remittance CLP02 codes, not the mutable
    ``claim.claim_status`` field (which gets overwritten by resubmission 835s).

    Returns a DataFrame with one row per claim and a ``denied`` label column.
    """
    result = await db.execute(
        _BUILD_DATASET_SQL,
        {"since_service_date": since_service_date},
    )
    rows = result.mappings().all()
    if not rows:
        # Return an empty frame with the expected columns so downstream
        # code that does ``df.empty`` / column access still works.
        return pd.DataFrame(columns=list(_DATASET_COLUMNS))
    df = pd.DataFrame(rows, columns=list(_DATASET_COLUMNS))
    # Cast denied to int64 — was a Python int before, now defaults to a
    # nullable PG integer that pandas may keep as object. Tests + the
    # trainer's ``y = df['denied']`` rely on a numeric dtype.
    df["denied"] = df["denied"].astype("int64")
    return df


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """Return label distribution stats for a labeled dataset DataFrame."""
    total = len(df)
    if total == 0:
        return {"total": 0, "denied": 0, "paid": 0, "denial_rate": 0.0}

    denied = int(df["denied"].sum())
    paid = total - denied
    return {
        "total": total,
        "denied": denied,
        "paid": paid,
        "denial_rate": round(denied / total, 4),
    }
