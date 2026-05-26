"""Reconcile unresolved prediction_log rows when remittances arrive.

When an 835 file finishes parsing, this service:
  1. Finds unresolved prediction_log rows whose claim now has at least one
     remittance attached.
  2. Applies the SAME labeling rule the trainer uses (dataset.py
     ``_label_from_remittances``) so live-monitoring numbers stay
     comparable to training metrics.
  3. Sets actual_denied / actual_status / resolved_at and links to the
     specific remittance row that decided the label.

Best-effort: any exception is logged and swallowed so a reconcile failure
cannot block the EDI upload response.

The query is bounded by ``prediction_log.resolved_at IS NULL`` so cost
grows with the *pending* prediction backlog, not with total remittance
volume. The index ``ix_prediction_log_unresolved`` covers it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ml.dataset import _DENIED_CLP02, _PAID_CLP02, _label_from_remittances
from app.models.prediction_log import PredictionLog
from app.models.remittance_claim import RemittanceClaim

logger = logging.getLogger(__name__)


def _pick_deciding_remittance(
    remittance_claims: list[RemittanceClaim],
) -> RemittanceClaim | None:
    """Return the remittance row that *caused* the final label.

    Mirrors the trainer's rule: a denial CLP02=4 sticks even if later
    remittances pay, so prefer the first denial. Otherwise pick the first
    row that maps to an actionable paid code.
    """
    if not remittance_claims:
        return None
    sorted_rcs = sorted(
        remittance_claims,
        key=lambda rc: (rc.remittance_date or datetime.min.date(), rc.id),
    )
    for rc in sorted_rcs:
        if rc.claim_status_code in _DENIED_CLP02:
            return rc
    for rc in sorted_rcs:
        if rc.claim_status_code in _PAID_CLP02:
            return rc
    return None


async def reconcile_pending(db: AsyncSession) -> dict:
    """Reconcile every currently-unresolved prediction whose claim has remittances.

    Called after an 835 upload but intentionally NOT scoped to that file —
    a single 835 can affect predictions made days ago, and re-running this
    is safely idempotent (resolved rows are excluded by the WHERE clause).

    Returns a small summary dict and never raises.
    """
    summary: dict = {"reconciled": 0, "skipped": 0, "error": None}
    try:
        # Pull unresolved predictions joined to the claim, eagerly loading
        # remittance_claims so we can apply the labeling rule without an
        # N+1 query.
        stmt = (
            select(PredictionLog)
            .where(
                PredictionLog.resolved_at.is_(None),
                PredictionLog.claim_id.is_not(None),
            )
        )
        unresolved_result = await db.execute(stmt)
        unresolved = unresolved_result.scalars().all()

        if not unresolved:
            return summary

        # Batch-load remittances for the relevant claim_ids.
        claim_ids = list({row.claim_id for row in unresolved if row.claim_id})
        rc_stmt = select(RemittanceClaim).where(
            RemittanceClaim.claim_id.in_(claim_ids)
        )
        rc_result = await db.execute(rc_stmt)
        rcs_by_claim: dict[int, list[RemittanceClaim]] = {}
        for rc in rc_result.scalars().all():
            rcs_by_claim.setdefault(rc.claim_id, []).append(rc)

        now = datetime.now(timezone.utc)
        for row in unresolved:
            rcs = rcs_by_claim.get(row.claim_id, [])
            if not rcs:
                summary["skipped"] += 1
                continue
            label = _label_from_remittances(rcs)
            if label is None:
                summary["skipped"] += 1
                continue
            deciding = _pick_deciding_remittance(rcs)
            row.actual_denied = label
            row.actual_status = "denied" if label == 1 else "paid"
            row.resolved_at = now
            row.resolved_by_remittance_id = deciding.id if deciding else None
            summary["reconciled"] += 1

        await db.commit()
        if summary["reconciled"] or summary["skipped"]:
            logger.info("Reconcile complete: %s", summary)
    except Exception as exc:
        logger.exception("Prediction reconciliation failed")
        try:
            await db.rollback()
        except Exception:
            logger.exception("Rollback after reconcile failure also failed")
        summary["error"] = str(exc)
    return summary
