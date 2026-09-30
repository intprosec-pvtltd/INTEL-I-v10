"""
Security and role-access rules for INTEL-I.

Policy:
- Every active authenticated user can operate INTEL-I.
- Only super_admin can manage user accounts.
"""

from __future__ import annotations

from enum import Enum

from fastapi import (
    Depends,
    HTTPException,
    Request,
    status,
)


# ============================================================
# ROLES
# ============================================================

class Role(str, Enum):
    SUPER_ADMIN = "super_admin"

    RTO_ADMIN = "rto_admin"
    RTO_STAFF = "rto_staff"

    CRIME_ADMIN = "crime_admin"
    CRIME_STAFF = "crime_staff"

    CYBER_ADMIN = "cyber_admin"
    CYBER_STAFF = "cyber_staff"


# ============================================================
# DEPARTMENTS
# ============================================================

ROLE_DEPARTMENT = {
    Role.SUPER_ADMIN: None,

    Role.RTO_ADMIN: "rto",
    Role.RTO_STAFF: "rto",

    Role.CRIME_ADMIN: "crime",
    Role.CRIME_STAFF: "crime",

    Role.CYBER_ADMIN: "cyber",
    Role.CYBER_STAFF: "cyber",
}


ADMIN_ROLES = frozenset(
    {
        Role.SUPER_ADMIN,
        Role.RTO_ADMIN,
        Role.CRIME_ADMIN,
        Role.CYBER_ADMIN,
    }
)


NON_SUPER_ADMIN_ROLES = (
    Role.RTO_ADMIN,
    Role.RTO_STAFF,
    Role.CRIME_ADMIN,
    Role.CRIME_STAFF,
    Role.CYBER_ADMIN,
    Role.CYBER_STAFF,
)


# ============================================================
# OPERATIONAL PERMISSIONS
# ============================================================

OPERATIONAL_PERMISSIONS = {
    "dashboard.view",
    "alerts.view",
    "alerts.manage",
    "incidents.view",
    "incidents.manage",
    "gis.view",
    "gis.manage",
    "assistant.use",
    "cameras.view",
    "cameras.manage",
    "streams.view",
    "streams.manage",
    "uploads.view",
    "uploads.manage",
    "zones.view",
    "zones.manage",
    "rules.view",
    "rules.manage",
    "vehicle.view",
    "vehicle.manage",
    "person.view",
    "person.manage",
    "watchlists.view",
    "watchlists.manage",
    "intelligence.view",
    "intelligence.manage",
    "system.view",
}


PERMISSIONS = {
    Role.SUPER_ADMIN: {"*"},

    Role.RTO_ADMIN: set(OPERATIONAL_PERMISSIONS),
    Role.RTO_STAFF: set(OPERATIONAL_PERMISSIONS),

    Role.CRIME_ADMIN: set(OPERATIONAL_PERMISSIONS),
    Role.CRIME_STAFF: set(OPERATIONAL_PERMISSIONS),

    Role.CYBER_ADMIN: set(OPERATIONAL_PERMISSIONS),
    Role.CYBER_STAFF: set(OPERATIONAL_PERMISSIONS),
}


# ============================================================
# ROLE HELPERS
# ============================================================

def normalize_role(
    value: str | Role,
) -> Role:
    if isinstance(value, Role):
        return value

    try:
        return Role(str(value).strip().lower())

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has no valid role",
        ) from exc


def permissions_for(
    role: str | Role,
) -> list[str]:
    normalized_role = normalize_role(role)

    return sorted(
        PERMISSIONS.get(
            normalized_role,
            set(),
        )
    )


def has_permission(
    role: str | Role,
    permission: str,
) -> bool:
    normalized_role = normalize_role(role)

    granted = PERMISSIONS.get(
        normalized_role,
        set(),
    )

    if "*" in granted:
        return True

    if permission in granted:
        return True

    if permission.endswith(".view"):
        manage_permission = (
            permission.removesuffix(".view")
            + ".manage"
        )

        if manage_permission in granted:
            return True

    return False


# ============================================================
# OPERATIONAL ACCESS
# ============================================================

def permission_for_request(
    method: str,
    path: str,
) -> str | None:
    """
    All valid active authenticated roles can operate INTEL-I.

    Account-management routes use require_super_admin()
    separately.
    """

    normalized_path = (
        "/"
        + str(path or "")
        .strip()
        .lower()
        .lstrip("/")
    )

    if normalized_path.startswith("/auth/"):
        return None

    if normalized_path.startswith("/api/rbac/"):
        return None

    return "dashboard.view"


def enforce_request_permission(
    user,
    method: str,
    path: str,
) -> None:
    """
    Called from auth.auth.get_current_user().

    Authentication and active-account validation have already
    occurred. This confirms that the account has a valid role.
    """

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not getattr(user, "is_active", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account disabled",
        )

    role = normalize_role(user.role)

    if role not in PERMISSIONS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account has no valid role",
        )


# ============================================================
# SPECIFIC PERMISSION DEPENDENCY
# ============================================================

def require_permissions(
    *required_permissions: str,
):
    from auth.auth import get_current_user

    required = tuple(
        str(permission).strip()
        for permission in required_permissions
        if str(permission).strip()
    )

    def dependency(
        request: Request,
        current_user=Depends(get_current_user),
    ):
        missing = [
            permission
            for permission in required
            if not has_permission(
                current_user.role,
                permission,
            )
        ]

        if missing:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Insufficient permission: "
                    + ", ".join(missing)
                ),
            )

        request.state.user = current_user
        return current_user

    return dependency


# ============================================================
# SUPER-ADMIN DEPENDENCY
# ============================================================

def require_super_admin():
    from auth.auth import get_current_user

    def dependency(
        request: Request,
        current_user=Depends(get_current_user),
    ):
        if (
            normalize_role(current_user.role)
            != Role.SUPER_ADMIN
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    "Only the super administrator "
                    "can perform this action"
                ),
            )

        request.state.user = current_user
        return current_user

    return dependency


# ============================================================
# ACCOUNT MANAGEMENT
# ============================================================

def can_manage(
    actor,
    target,
) -> bool:
    if actor is None or target is None:
        return False

    if normalize_role(actor.role) != Role.SUPER_ADMIN:
        return False

    if actor.id == target.id:
        return False

    if normalize_role(target.role) == Role.SUPER_ADMIN:
        return False

    return True


def allowed_child_roles(
    actor,
) -> tuple[Role, ...]:
    if actor is None:
        return ()

    if normalize_role(actor.role) != Role.SUPER_ADMIN:
        return ()

    return NON_SUPER_ADMIN_ROLES