"""EDI file upload and listing endpoints."""

import hashlib
import logging

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.edi_file import EdiFile
from app.schemas.edi_file import (
    EdiFileResponse,
    ParseResultResponse,
    ValidationErrorResponse,
)
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

    # Duplicate file check
    content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    existing = await db.execute(
        select(EdiFile.id).where(EdiFile.content_hash == content_hash)
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=409,
            detail="This file has already been uploaded.",
        )
    # Reset implicit transaction so _save_all can call db.begin()
    await db.rollback()

    parser = EdiParser()
    result = await parser.parse_file(file.filename or "unknown.edi", raw_text, db, content_hash=content_hash)

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
        validation_errors=[
            ValidationErrorResponse(
                segment=ve.segment,
                field=ve.field,
                message=ve.message,
                severity=ve.severity.value,
                position=ve.position,
                claim_identifier=ve.claim_identifier,
                validator=ve.validator,
            )
            for ve in result.validation_errors
        ],
        validation_warning_count=result.validation_warning_count,
        validation_error_count=result.validation_error_count,
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
