"""EDI file upload and listing endpoints."""

import logging

from fastapi import APIRouter, Depends, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.edi_file import EdiFile
from app.schemas.edi_file import EdiFileResponse, ParseResultResponse
from app.services.edi_parser import EdiParser

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=ParseResultResponse)
async def upload_edi_file(
    file: UploadFile,
    db: AsyncSession = Depends(get_db),
):
    """Upload and parse an EDI 837 or 835 file."""
    raw_bytes = await file.read()
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw_text = raw_bytes.decode("latin-1")

    parser = EdiParser()
    result = await parser.parse_file(file.filename or "unknown.edi", raw_text, db)

    return ParseResultResponse(
        edi_file_id=result.edi_file_id,
        file_type=result.file_type,
        claims_count=result.claims_count,
        claim_lines_count=result.claim_lines_count,
        diagnoses_count=result.diagnoses_count,
        remittance_claims_count=result.remittance_claims_count,
        adjustments_count=result.adjustments_count,
        remark_codes_count=result.remark_codes_count,
        raw_segments_count=result.raw_segments_count,
        errors=result.errors,
        success=result.success,
    )


@router.get("/files", response_model=list[EdiFileResponse])
async def list_edi_files(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """List all uploaded EDI files, newest first."""
    stmt = (
        select(EdiFile)
        .order_by(EdiFile.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    rows = await db.execute(stmt)
    return rows.scalars().all()
