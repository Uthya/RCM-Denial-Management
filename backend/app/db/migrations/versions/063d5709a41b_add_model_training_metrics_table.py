"""add_model_training_metrics_table

Revision ID: 063d5709a41b
Revises: ad40975dfb88
Create Date: 2026-05-22 12:27:18.857334

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '063d5709a41b'
down_revision: Union[str, None] = 'ad40975dfb88'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        'model_training_metrics',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('training_id', sa.String(length=36), nullable=False),
        sa.Column('training_timestamp', sa.DateTime(timezone=True), nullable=False),
        sa.Column('total_claims_used', sa.Integer(), nullable=False),
        sa.Column('training_samples', sa.Integer(), nullable=False),
        sa.Column('test_samples', sa.Integer(), nullable=False),
        sa.Column('denied_claims', sa.Integer(), nullable=False),
        sa.Column('paid_claims', sa.Integer(), nullable=False),
        sa.Column('denial_rate', sa.Float(), nullable=False),
        sa.Column('accuracy', sa.Float(), nullable=False),
        sa.Column('precision', sa.Float(), nullable=False),
        sa.Column('recall', sa.Float(), nullable=False),
        sa.Column('f1_score', sa.Float(), nullable=False),
        sa.Column('roc_auc', sa.Float(), nullable=True),
        sa.Column('training_time_seconds', sa.Float(), nullable=False),
        sa.Column('model_version', sa.String(length=50), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=20), nullable=False),
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
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('training_id', name='uq_model_training_metrics_training_id'),
    )
    op.create_index(
        op.f('ix_model_training_metrics_training_id'),
        'model_training_metrics',
        ['training_id'],
        unique=True,
    )
    op.create_index(
        op.f('ix_model_training_metrics_training_timestamp'),
        'model_training_metrics',
        ['training_timestamp'],
        unique=False,
    )
    op.create_index(
        op.f('ix_model_training_metrics_model_version'),
        'model_training_metrics',
        ['model_version'],
        unique=False,
    )
    op.create_index(
        'ix_model_training_metrics_timestamp_desc',
        'model_training_metrics',
        ['training_timestamp'],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        'ix_model_training_metrics_timestamp_desc',
        table_name='model_training_metrics',
    )
    op.drop_index(
        op.f('ix_model_training_metrics_model_version'),
        table_name='model_training_metrics',
    )
    op.drop_index(
        op.f('ix_model_training_metrics_training_timestamp'),
        table_name='model_training_metrics',
    )
    op.drop_index(
        op.f('ix_model_training_metrics_training_id'),
        table_name='model_training_metrics',
    )
    op.drop_table('model_training_metrics')
