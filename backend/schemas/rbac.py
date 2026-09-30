"""
Pydantic request schemas for INTEL-I user and role management.
"""

from __future__ import annotations

from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
)

from security.rbac import Role


# ============================================================
# CREATE USER
# ============================================================

class StrictRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class UserCreate(StrictRequest):
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=150,
    )

    email: EmailStr

    password: str = Field(
        ...,
        min_length=12,
        max_length=128,
    )

    role: Role

    @field_validator("email")
    @classmethod
    def normalize_email(cls, value: EmailStr) -> str:
        return str(value).strip().lower()

    @field_validator("full_name")
    @classmethod
    def validate_full_name(
        cls,
        value: str,
    ) -> str:
        normalized = " ".join(value.split())

        if len(normalized) < 2:
            raise ValueError(
                "Full name must contain at least 2 characters"
            )

        return normalized


# ============================================================
# UPDATE USER
# ============================================================

class UserUpdate(StrictRequest):
    full_name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=150,
    )

    role: Optional[Role] = None

    is_active: Optional[bool] = None

    @field_validator("full_name")
    @classmethod
    def validate_optional_full_name(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None

        normalized = " ".join(value.split())

        if len(normalized) < 2:
            raise ValueError(
                "Full name must contain at least 2 characters"
            )

        return normalized


# ============================================================
# PASSWORD RESET
# ============================================================

class PasswordReset(StrictRequest):
    password: str = Field(
        ...,
        min_length=12,
        max_length=128,
    )
