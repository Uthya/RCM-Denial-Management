"""Claims listing and detail endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.claim import Claim
from app.models.remittance_claim import RemittanceClaim
from app.schemas.claim import ClaimDetailResponse, ClaimResponse

router = APIRouter()


@router.get("/", response_model=list[ClaimResponse])
async def list_claims(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
):
    """List all claims, newest first."""
    stmt = (
        select(Claim)
        .order_by(Claim.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = await db.execute(stmt)
    return rows.scalars().all()


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
