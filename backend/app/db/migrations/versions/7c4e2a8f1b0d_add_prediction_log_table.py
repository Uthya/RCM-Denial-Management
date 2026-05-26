"""add_prediction_log_table

Revision ID: 7c4e2a8f1b0d
Revises: 063d5709a41b
Create Date: 2026-05-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = '7c4e2a8f1b0d'
down_revision: Union[str, None] = '063d5709a41b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'prediction_log',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('claim_id', sa.BigInteger(), nullable=True),
        sa.Column('claim_number', sa.String(length=50), nullable=True),
        sa.Column('prediction_id', sa.String(length=36), nullable=False),
        sa.Column('predicted_risk', sa.Float(), nullable=False),
        sa.Column('predicted_label', sa.Integer(), nullable=False),
        sa.Column('risk_level', sa.String(length=10), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=False),
        sa.Column('feature_engineering_version', sa.String(length=50), nullable=False),
        sa.Column('feature_snapshot', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('prediction_time', sa.DateTime(timezone=True), nullable=False),
        sa.Column('actual_denied', sa.Integer(), nullable=True),
        sa.Column('actual_status', sa.String(length=20), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_by_remittance_id', sa.BigInteger(), nullable=True),
        sa.Column(
            'created_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.Column(
            'updated_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(['claim_id'], ['claims.id']),
        sa.ForeignKeyConstraint(['resolved_by_remittance_id'], ['remittance_claims.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('prediction_id', name='uq_prediction_log_prediction_id'),
    )

    op.create_index(
        op.f('ix_prediction_log_prediction_id'),
        'prediction_log',
        ['prediction_id'],
        unique=True,
    )
    op.create_index(
        op.f('ix_prediction_log_claim_id'),
        'prediction_log',
        ['claim_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_prediction_log_claim_number'),
        'prediction_log',
        ['claim_number'],
        unique=False,
    )
    op.create_index(
        op.f('ix_prediction_log_model_version'),
        'prediction_log',
        ['model_version'],
        unique=False,
    )
    op.create_index(
        op.f('ix_prediction_log_prediction_time'),
        'prediction_log',
        ['prediction_time'],
        unique=False,
    )
    op.create_index(
        op.f('ix_prediction_log_actual_denied'),
        'prediction_log',
        ['actual_denied'],
        unique=False,
    )
    op.create_index(
        op.f('ix_prediction_log_resolved_at'),
        'prediction_log',
        ['resolved_at'],
        unique=False,
    )
    op.create_index(
        'ix_prediction_log_unresolved',
        'prediction_log',
        ['claim_id', 'resolved_at'],
        unique=False,
    )
    op.create_index(
        'ix_prediction_log_perf_rollup',
        'prediction_log',
        ['resolved_at', 'prediction_time'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index('ix_prediction_log_perf_rollup', table_name='prediction_log')
    op.drop_index('ix_prediction_log_unresolved', table_name='prediction_log')
    op.drop_index(op.f('ix_prediction_log_resolved_at'), table_name='prediction_log')
    op.drop_index(op.f('ix_prediction_log_actual_denied'), table_name='prediction_log')
    op.drop_index(op.f('ix_prediction_log_prediction_time'), table_name='prediction_log')
    op.drop_index(op.f('ix_prediction_log_model_version'), table_name='prediction_log')
    op.drop_index(op.f('ix_prediction_log_claim_number'), table_name='prediction_log')
    op.drop_index(op.f('ix_prediction_log_claim_id'), table_name='prediction_log')
    op.drop_index(op.f('ix_prediction_log_prediction_id'), table_name='prediction_log')
    op.drop_table('prediction_log')
