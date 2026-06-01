"""add_auth_referral_provider_to_claims

Adds the four v5 feature columns to ``claims``:
  - authorization_number      (REF*G1 / REF*G3)
  - referral_number           (REF*9F)
  - billing_provider_npi      (NM1*85, element 9 when qualifier=XX)
  - rendering_provider_npi    (NM1*82, element 9 when qualifier=XX)

All nullable: existing claims keep working untouched. The values
"not present" carry signal (encoded as ``has_prior_authorization=0`` /
``has_referral=0`` features), so the absence of data is itself a feature
rather than a data-quality problem.

Revision ID: a1b2c3d4e5f6
Revises: b81a3f72c4d2
Create Date: 2026-06-01 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, None] = "b81a3f72c4d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "claims",
        sa.Column("authorization_number", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "claims",
        sa.Column("referral_number", sa.String(length=50), nullable=True),
    )
    op.add_column(
        "claims",
        sa.Column("billing_provider_npi", sa.String(length=20), nullable=True),
    )
    op.add_column(
        "claims",
        sa.Column("rendering_provider_npi", sa.String(length=20), nullable=True),
    )

    # Indices on the NPI columns: target encoding scans these per-fold during
    # training, and downstream OOV reporting filters by NPI. Both columns are
    # high-cardinality strings; the indices keep the WHERE/GROUP BY paths fast
    # without bloating the row.
    op.create_index(
        "ix_claims_billing_provider_npi",
        "claims",
        ["billing_provider_npi"],
    )
    op.create_index(
        "ix_claims_rendering_provider_npi",
        "claims",
        ["rendering_provider_npi"],
    )


def downgrade() -> None:
    op.drop_index("ix_claims_rendering_provider_npi", table_name="claims")
    op.drop_index("ix_claims_billing_provider_npi", table_name="claims")
    op.drop_column("claims", "rendering_provider_npi")
    op.drop_column("claims", "billing_provider_npi")
    op.drop_column("claims", "referral_number")
    op.drop_column("claims", "authorization_number")
