"""Claims listing and detail endpoints."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.claim import Claim
from app.models.enums import ClaimStatus
from app.models.remittance_claim import RemittanceClaim
from app.schemas.claim import ClaimDetailResponse, ClaimResponse

router = APIRouter()

_SORT_COLUMNS = {
    "claim_number": Claim.claim_number,
    "payer_name": Claim.payer_name,
    "total_charge_amount": Claim.total_charge_amount,
    "claim_status": Claim.claim_status,
    "service_from_date": Claim.service_from_date,
    "created_at": Claim.created_at,
}


class PaginatedClaimsResponse(BaseModel):
    items: list[ClaimResponse]
    total: int


@router.get("/", response_model=PaginatedClaimsResponse)
async def list_claims(
    skip: int = 0,
    limit: int = 100,
    status: str | None = Query(None, description="Filter by claim_status"),
    sort_by: str = Query("created_at", description="Column to sort by"),
    sort_dir: Literal["asc", "desc"] = Query("desc", description="Sort direction"),
    db: AsyncSession = Depends(get_db),
):
    """List claims with pagination, filtering, and sorting."""
    base = select(Claim)

    if status:
        base = base.where(Claim.claim_status == status)

    # Count
    count_stmt = select(func.count()).select_from(base.subquery())
    total = (await db.execute(count_stmt)).scalar()

    # Sort
    col = _SORT_COLUMNS.get(sort_by, Claim.created_at)
    order = col.asc() if sort_dir == "asc" else col.desc()

    stmt = base.order_by(order).offset(skip).limit(limit)
    rows = await db.execute(stmt)
    return PaginatedClaimsResponse(items=rows.scalars().all(), total=total)


@router.get("/{claim_id}", response_model=ClaimDetailResponse)
async def get_claim(
    claim_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get a single claim with all related data."""
    stmt = (
        select(Claim)
        .where(Claim.id == claim_id)
        .options(
            selectinload(Claim.claim_lines),
            selectinload(Claim.diagnoses),
            selectinload(Claim.remittance_claims)
            .selectinload(RemittanceClaim.adjustments),
            selectinload(Claim.remittance_claims)
            .selectinload(RemittanceClaim.remark_codes),
        )
    )
    result = await db.execute(stmt)
    claim = result.scalar_one_or_none()
    if claim is None:
        raise HTTPException(status_code=404, detail="Claim not found")
    return claim
