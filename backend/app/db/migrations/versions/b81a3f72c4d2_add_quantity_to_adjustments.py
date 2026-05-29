"""add_quantity_to_adjustments

Revision ID: b81a3f72c4d2
Revises: 7c4e2a8f1b0d
Create Date: 2026-05-27 22:00:00.000000

Adds the optional ``quantity`` column to ``adjustments``. The CAS segment in
X12 005010 emits triplets of (reason, amount, quantity) where quantity is
optional — the old parser silently discarded it. The new parser captures it
when present.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b81a3f72c4d2'
down_revision: Union[str, None] = '7c4e2a8f1b0d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'adjustments',
        sa.Column('quantity', sa.Numeric(precision=12, scale=3), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('adjustments', 'quantity')
