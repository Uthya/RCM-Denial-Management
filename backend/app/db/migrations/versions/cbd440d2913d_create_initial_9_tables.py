"""create initial 9 tables

Revision ID: cbd440d2913d
Revises:
Create Date: 2026-05-18

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "cbd440d2913d"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. edi_files
    op.create_table(
        "edi_files",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("file_type", sa.Enum("edi_837", "edi_835", name="filetype"), nullable=False),
        sa.Column("file_name", sa.String(255), nullable=False),
        sa.Column("interchange_control_no", sa.String(20), nullable=True),
        sa.Column("sender_id", sa.String(50), nullable=True),
        sa.Column("receiver_id", sa.String(50), nullable=True),
        sa.Column("raw_text", sa.Text, nullable=True),
    )

    # 2. claims
    op.create_table(
        "claims",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("claim_number", sa.String(50), nullable=False),
        sa.Column("payer_name", sa.String(255), nullable=True),
        sa.Column("patient_member_id", sa.String(80), nullable=True),
        sa.Column("total_charge_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("facility_type_code", sa.String(10), nullable=True),
        sa.Column("frequency_code", sa.String(5), nullable=True),
        sa.Column("service_from_date", sa.Date, nullable=False),
        sa.Column("service_to_date", sa.Date, nullable=True),
        sa.Column(
            "claim_status",
            sa.Enum("submitted", "paid", "denied", "partially_paid", "void", name="claimstatus"),
            nullable=False,
            server_default="submitted",
        ),
        sa.Column("edi_file_id", sa.BigInteger, sa.ForeignKey("edi_files.id"), nullable=True),
        sa.Column("raw_claim_segment", sa.Text, nullable=True),
    )
    op.create_index("ix_claims_claim_number", "claims", ["claim_number"])
    op.create_index("ix_claims_patient_member_id", "claims", ["patient_member_id"])
    op.create_index("ix_claims_claim_status", "claims", ["claim_status"])
    op.create_index("ix_claims_edi_file_id", "claims", ["edi_file_id"])
    op.create_index("ix_claims_edi_file_claim_number", "claims", ["edi_file_id", "claim_number"], unique=True)

    # 3. claim_lines
    op.create_table(
        "claim_lines",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("claim_id", sa.BigInteger, sa.ForeignKey("claims.id"), nullable=False),
        sa.Column("line_number", sa.Integer, nullable=False),
        sa.Column("procedure_code", sa.String(10), nullable=False),
        sa.Column("modifier1", sa.String(5), nullable=True),
        sa.Column("modifier2", sa.String(5), nullable=True),
        sa.Column("units", sa.Numeric(10, 2), nullable=False, server_default="1"),
        sa.Column("billed_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("service_date", sa.Date, nullable=True),
        sa.Column("place_of_service", sa.String(2), nullable=True),
        sa.Column("raw_sv1_segment", sa.Text, nullable=True),
    )
    op.create_index("ix_claim_lines_claim_id", "claim_lines", ["claim_id"])
    op.create_index("ix_claim_lines_claim_id_line_number", "claim_lines", ["claim_id", "line_number"], unique=True)

    # 4. diagnoses
    op.create_table(
        "diagnoses",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("claim_id", sa.BigInteger, sa.ForeignKey("claims.id"), nullable=False),
        sa.Column("diagnosis_code", sa.String(10), nullable=False),
        sa.Column("diagnosis_type", sa.String(10), nullable=False),
        sa.Column("sequence_number", sa.Integer, nullable=False),
        sa.Column("raw_hi_segment", sa.Text, nullable=True),
    )
    op.create_index("ix_diagnoses_claim_id", "diagnoses", ["claim_id"])
    op.create_index("ix_diagnoses_claim_id_sequence", "diagnoses", ["claim_id", "sequence_number"], unique=True)

    # 5. remittance_claims
    op.create_table(
        "remittance_claims",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("claim_id", sa.BigInteger, sa.ForeignKey("claims.id"), nullable=False),
        sa.Column("claim_status_code", sa.String(10), nullable=False),
        sa.Column("billed_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("paid_amount", sa.Numeric(12, 2), nullable=False, server_default="0"),
        sa.Column("payer_claim_control_number", sa.String(50), nullable=True),
        sa.Column("remittance_date", sa.Date, nullable=False),
        sa.Column("raw_clp_segment", sa.Text, nullable=True),
    )
    op.create_index("ix_remittance_claims_claim_id", "remittance_claims", ["claim_id"])

    # 6. adjustments
    op.create_table(
        "adjustments",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("remittance_claim_id", sa.BigInteger, sa.ForeignKey("remittance_claims.id"), nullable=False),
        sa.Column("adjustment_group_code", sa.String(5), nullable=False),
        sa.Column("adjustment_reason_code", sa.String(10), nullable=False),
        sa.Column("adjustment_amount", sa.Numeric(12, 2), nullable=False),
        sa.Column("raw_cas_segment", sa.Text, nullable=True),
    )
    op.create_index("ix_adjustments_remittance_claim_id", "adjustments", ["remittance_claim_id"])
    op.create_index("ix_adjustments_adjustment_reason_code", "adjustments", ["adjustment_reason_code"])

    # 7. remark_codes
    op.create_table(
        "remark_codes",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("remittance_claim_id", sa.BigInteger, sa.ForeignKey("remittance_claims.id"), nullable=False),
        sa.Column("remark_code", sa.String(10), nullable=False),
        sa.Column("raw_lq_segment", sa.Text, nullable=True),
    )
    op.create_index("ix_remark_codes_remittance_claim_id", "remark_codes", ["remittance_claim_id"])

    # 8. raw_segments
    op.create_table(
        "raw_segments",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("edi_file_id", sa.BigInteger, sa.ForeignKey("edi_files.id"), nullable=True),
        sa.Column("claim_id", sa.BigInteger, sa.ForeignKey("claims.id"), nullable=True),
        sa.Column("segment_name", sa.String(10), nullable=False),
        sa.Column("segment_position", sa.Integer, nullable=False),
        sa.Column("raw_segment_text", sa.Text, nullable=False),
        sa.Column("parse_error", sa.Text, nullable=True),
    )
    op.create_index("ix_raw_segments_edi_file_id", "raw_segments", ["edi_file_id"])
    op.create_index("ix_raw_segments_edi_file_position", "raw_segments", ["edi_file_id", "segment_position"])

    # 9. code_masters
    op.create_table(
        "code_masters",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("code_type", sa.Enum("carc", "rarc", "pos", "claim_status", name="codetype"), nullable=False),
        sa.Column("code", sa.String(10), nullable=False),
        sa.Column("description", sa.String(500), nullable=False),
    )
    op.create_index("ix_code_masters_type_code", "code_masters", ["code_type", "code"], unique=True)


def downgrade() -> None:
    op.drop_table("code_masters")
    op.drop_table("raw_segments")
    op.drop_table("remark_codes")
    op.drop_table("adjustments")
    op.drop_table("remittance_claims")
    op.drop_table("diagnoses")
    op.drop_table("claim_lines")
    op.drop_table("claims")
    op.drop_table("edi_files")

    op.execute("DROP TYPE IF EXISTS filetype")
    op.execute("DROP TYPE IF EXISTS claimstatus")
    op.execute("DROP TYPE IF EXISTS codetype")
