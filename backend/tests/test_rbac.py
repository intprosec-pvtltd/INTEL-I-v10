from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from schemas.rbac import UserCreate, UserUpdate
from security.rbac import (
    NON_SUPER_ADMIN_ROLES,
    Role,
    allowed_child_roles,
    can_manage,
    has_permission,
    permission_for_request,
)


def user(user_id, role, manager_id=None):
    return SimpleNamespace(
        id=user_id,
        role=role,
        manager_id=manager_id,
    )


def test_only_super_admin_can_manage_user_accounts():
    root = user(1, Role.SUPER_ADMIN)
    department_admin = user(2, Role.CRIME_ADMIN, 1)
    staff = user(3, Role.CRIME_STAFF, 2)
    another_super_admin = user(4, Role.SUPER_ADMIN)

    assert allowed_child_roles(root) == NON_SUPER_ADMIN_ROLES
    assert allowed_child_roles(department_admin) == ()
    assert can_manage(root, department_admin)
    assert can_manage(root, staff)
    assert not can_manage(root, root)
    assert not can_manage(root, another_super_admin)
    assert not can_manage(department_admin, staff)


def test_user_management_permission_is_super_admin_only():
    assert has_permission(Role.SUPER_ADMIN, "users.manage")

    for role in NON_SUPER_ADMIN_ROLES:
        assert has_permission(role, "dashboard.view")
        assert not has_permission(role, "users.manage")


def test_account_routes_use_dedicated_server_side_dependency():
    assert permission_for_request("GET", "/auth/login") is None
    assert permission_for_request("POST", "/api/rbac/users") is None
    assert permission_for_request("GET", "/cameras") == "dashboard.view"


def test_user_create_normalizes_input_and_rejects_extra_fields():
    payload = UserCreate(
        full_name="  Control   Room Operator  ",
        email="OPERATOR@EXAMPLE.COM",
        password="StrongPassword!123",
        role=Role.CRIME_STAFF,
    )

    assert payload.full_name == "Control Room Operator"
    assert str(payload.email) == "operator@example.com"

    with pytest.raises(ValidationError):
        UserCreate(
            full_name="Operator",
            email="operator@example.com",
            password="StrongPassword!123",
            role=Role.CRIME_STAFF,
            is_active=False,
        )


def test_user_update_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        UserUpdate(email="changed@example.com")


def test_public_signup_endpoint_is_not_present():
    main_source = (Path(__file__).parents[1] / "main.py").read_text(
        encoding="utf-8",
    )

    assert '@app.post("/auth/signup")' not in main_source
    assert "class SignupRequest" not in main_source
