from __future__ import annotations

import os

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    Response,
    status,
)
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from auth.auth import (
    get_current_user,
    hash_password,
    validate_password_strength,
)
from db.database import getDB
from db.model import (
    AuditLog,
    RefreshToken,
    User,
)
from schemas.rbac import (
    PasswordReset,
    UserCreate,
    UserUpdate,
)
from security.rbac import (
    ROLE_DEPARTMENT,
    Role,
    allowed_child_roles,
    can_manage,
    normalize_role,
    permissions_for,
    require_super_admin,
)
from security.rateLimit import limiter
from services.authSecurityService import admin_reset_password


router = APIRouter(
    prefix="/api/rbac",
    tags=["RBAC"],
)


# ============================================================
# USER RESPONSE
# ============================================================


def _view(user: User) -> dict:
    return {
        "id": user.id,
        "full_name": user.full_name,
        "email": user.email,
        "role": user.role,
        "department": user.department,
        "manager_id": user.manager_id,
        "is_active": user.is_active,
        "permissions": permissions_for(user.role),
        "created_at": user.created_at,
    }


# ============================================================
# AUDIT LOG
# ============================================================


def _audit(
    db: Session,
    actor: User,
    action: str,
    target: User,
    details: dict | None = None,
    source_ip: str | None = None,
) -> None:
    """
    Existing RBAC audit helper used by ordinary user-management operations.

    Password resets intentionally do not use this helper because the
    production authSecurityService owns that complete transaction and writes
    its own sanitized security audit event.
    """

    db.add(
        AuditLog(
            user_id=actor.id,
            action=action,
            resource_type="user",
            resource_id=str(target.id),
            details=details or {},
            source_ip=source_ip,
        )
    )



def _source_ip(request: Request) -> str | None:
    """
    Return the immediate peer IP.

    Do not trust X-Forwarded-For directly here. Reverse-proxy-aware source-IP
    handling for the new authentication-security routes is implemented in
    routers/auth_security.py with an explicit trusted-proxy allowlist.
    """

    if request.client is None:
        return None

    return str(request.client.host or "")[:64] or None


# ============================================================
# GET MANAGEABLE USER
# ============================================================


def _target(
    db: Session,
    actor: User,
    user_id: int,
) -> User:
    target = (
        db.query(User)
        .filter(User.id == user_id)
        .first()
    )

    if target is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    if not can_manage(actor, target):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot manage this user",
        )

    return target


# ============================================================
# CURRENT USER ROLE INFORMATION
# Every authenticated user can view their own permissions.
# ============================================================


@router.get("/roles")
def get_roles(
    current_user: User = Depends(get_current_user),
):
    return {
        "current_role": current_user.role,
        "assignable_roles": [
            role.value
            for role in allowed_child_roles(current_user)
        ],
        "permissions": permissions_for(current_user.role),
    }


# ============================================================
# LIST USERS
# SUPER ADMIN ONLY
# ============================================================


@router.get("/users")
def list_users(
    db: Session = Depends(getDB),
    actor: User = Depends(require_super_admin()),
):
    users = (
        db.query(User)
        .filter(User.id != actor.id)
        .order_by(User.created_at.desc())
        .all()
    )

    return [_view(user) for user in users]


# ============================================================
# CREATE USER
# SUPER ADMIN ONLY
# ============================================================


@router.post(
    "/users",
    status_code=status.HTTP_201_CREATED,
)
@limiter.limit(os.getenv("RATE_LIMIT_USER_CREATE", "20/hour"))
def create_user(
    request: Request,
    response: Response,
    payload: UserCreate,
    db: Session = Depends(getDB),
    actor: User = Depends(require_super_admin()),
):
    role = normalize_role(payload.role)

    if role == Role.SUPER_ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "Another super administrator cannot be "
                "created through this endpoint"
            ),
        )

    if role not in allowed_child_roles(actor):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You cannot assign this role",
        )

    full_name = payload.full_name.strip()
    email = payload.email.strip().lower()

    if not full_name:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Full name is required",
        )

    existing_user = (
        db.query(User.id)
        .filter(func.lower(User.email) == email)
        .first()
    )

    if existing_user is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists",
        )

    validate_password_strength(payload.password)

    new_user = User(
        full_name=full_name,
        email=email,
        password=hash_password(payload.password),
        role=role.value,
        department=ROLE_DEPARTMENT.get(role),
        manager_id=actor.id,
        is_active=True,
        token_version=0,
    )

    try:
        db.add(new_user)
        db.flush()

        _audit(
            db=db,
            actor=actor,
            action="rbac.user.created",
            target=new_user,
            details={
                "role": role.value,
                "department": new_user.department,
            },
            source_ip=_source_ip(request),
        )

        db.commit()
        db.refresh(new_user)

    except IntegrityError as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already exists",
        ) from exc

    except HTTPException:
        db.rollback()
        raise

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to create user",
        ) from exc

    return _view(new_user)


# ============================================================
# UPDATE USER
# SUPER ADMIN ONLY
# ============================================================


@router.patch("/users/{user_id}")
@limiter.limit(os.getenv("RATE_LIMIT_USER_UPDATE", "60/hour"))
def update_user(
    request: Request,
    response: Response,
    user_id: int,
    payload: UserUpdate,
    db: Session = Depends(getDB),
    actor: User = Depends(require_super_admin()),
):
    if not payload.model_fields_set:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one account field must be provided",
        )

    target = _target(
        db=db,
        actor=actor,
        user_id=user_id,
    )

    if payload.role is not None:
        role = normalize_role(payload.role)

        if role == Role.SUPER_ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "The super administrator role cannot "
                    "be assigned through this endpoint"
                ),
            )

        if role not in allowed_child_roles(actor):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You cannot assign this role",
            )

        target.role = role.value
        target.department = ROLE_DEPARTMENT.get(role)

    if payload.full_name is not None:
        full_name = payload.full_name.strip()

        if not full_name:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Full name cannot be empty",
            )

        target.full_name = full_name

    if payload.is_active is not None:
        target.is_active = bool(payload.is_active)

    # Changing a user account invalidates all existing sessions.
    target.token_version = int(target.token_version or 0) + 1

    try:
        (
            db.query(RefreshToken)
            .filter(RefreshToken.user_id == target.id)
            .update(
                {"revoked": True},
                synchronize_session=False,
            )
        )

        _audit(
            db=db,
            actor=actor,
            action="rbac.user.updated",
            target=target,
            details={
                "role": target.role,
                "department": target.department,
                "is_active": target.is_active,
            },
            source_ip=_source_ip(request),
        )

        db.commit()
        db.refresh(target)

    except HTTPException:
        db.rollback()
        raise

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to update user",
        ) from exc

    return _view(target)


# ============================================================
# RESET PASSWORD
# SUPER ADMIN ONLY
# ============================================================


@router.post(
    "/users/{user_id}/reset-password",
    status_code=status.HTTP_204_NO_CONTENT,
)
@limiter.limit(os.getenv("RATE_LIMIT_PASSWORD_RESET", "20/hour"))
def reset_password(
    request: Request,
    user_id: int,
    payload: PasswordReset,
    db: Session = Depends(getDB),
    actor: User = Depends(require_super_admin()),
):
    """
    Reset a managed user's password using the production security service.

    The external API contract is intentionally preserved:

        POST /api/rbac/users/{user_id}/reset-password
        {"password": "..."}
        -> 204 No Content

    The security service now performs the sensitive mutation atomically:
    - full password policy validation
    - recent password reuse prevention
    - previous password hash added to bounded history
    - new password hashing
    - outstanding forgot-password tokens revoked
    - User.token_version incremented
    - all refresh tokens revoked
    - sanitized AuditLog event written

    The target user's existing MFA enrollment is retained.
    """

    target = _target(
        db=db,
        actor=actor,
        user_id=user_id,
    )

    try:
        admin_reset_password(
            db,
            actor=actor,
            target=target,
            new_password=payload.password,
            source_ip=_source_ip(request),
        )

    except PermissionError:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Unable to reset this user's password",
        )

    except ValueError as exc:
        db.rollback()

        # Password policy/history messages are safe to return to the
        # authenticated Super Admin. Password values themselves are never
        # included in the exception or logs.
        detail = str(exc).strip() or "Invalid password"

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail[:500],
        )

    except HTTPException:
        db.rollback()
        raise

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to reset password",
        ) from exc

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )


# ============================================================
# DELETE USER
# SUPER ADMIN ONLY
# ============================================================


@router.delete(
    "/users/{user_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
@limiter.limit(os.getenv("RATE_LIMIT_USER_DELETE", "10/hour"))
def delete_user(
    request: Request,
    user_id: int,
    db: Session = Depends(getDB),
    actor: User = Depends(require_super_admin()),
):
    target = _target(
        db=db,
        actor=actor,
        user_id=user_id,
    )

    try:
        (
            db.query(RefreshToken)
            .filter(RefreshToken.user_id == target.id)
            .delete(synchronize_session=False)
        )

        _audit(
            db=db,
            actor=actor,
            action="rbac.user.deleted",
            target=target,
            details={
                "email": target.email,
                "role": target.role,
            },
            source_ip=_source_ip(request),
        )

        db.delete(target)
        db.commit()

    except Exception as exc:
        db.rollback()

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to delete user",
        ) from exc

    return Response(
        status_code=status.HTTP_204_NO_CONTENT
    )
