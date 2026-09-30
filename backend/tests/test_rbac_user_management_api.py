from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import BigInteger, create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from auth.auth import create_access_token
from db import advanced_intelligence_model, intelligence_model, watchlist_model  # noqa: F401
from db.database import Base, getDB
from db.model import AuditLog, RefreshToken, User, indian_time
from routers.rbac import router
from security.rateLimit import setup_rate_limiter


@compiles(BigInteger, "sqlite")
def compile_big_integer_for_sqlite(_type, _compiler, **_kwargs):
    return "INTEGER"


def make_token(user: User) -> str:
    return create_access_token(
        {
            "sub": str(user.id),
            "user_id": user.id,
            "role": user.role,
            "ver": int(user.token_version or 0),
        }
    )


def test_user_management_api_is_super_admin_only():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    TestingSession = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    Base.metadata.create_all(bind=engine)

    db = TestingSession()
    super_admin = User(
        full_name="Super Administrator",
        email="super@example.com",
        password="unused-test-hash",
        role="super_admin",
        is_active=True,
        token_version=0,
    )
    staff = User(
        full_name="Crime Operator",
        email="operator@example.com",
        password="unused-test-hash",
        role="crime_staff",
        department="crime",
        is_active=True,
        token_version=0,
    )
    db.add_all([super_admin, staff])
    db.commit()
    db.refresh(super_admin)
    db.refresh(staff)

    app = FastAPI()
    setup_rate_limiter(app)
    app.include_router(router)

    def override_database():
        session = TestingSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[getDB] = override_database
    client = TestClient(app)

    try:
        unauthenticated = client.get("/api/rbac/users")
        assert unauthenticated.status_code == 401

        staff_response = client.get(
            "/api/rbac/users",
            cookies={"access_token": make_token(staff)},
        )
        assert staff_response.status_code == 403

        super_cookies = {"access_token": make_token(super_admin)}
        roles_response = client.get("/api/rbac/roles", cookies=super_cookies)
        assert roles_response.status_code == 200
        assert "super_admin" not in roles_response.json()["assignable_roles"]

        create_response = client.post(
            "/api/rbac/users",
            cookies=super_cookies,
            json={
                "full_name": "RTO Operator",
                "email": "RTO.OPERATOR@EXAMPLE.COM",
                "password": "StrongPassword!123",
                "role": "rto_staff",
            },
        )
        assert create_response.status_code == 201
        assert "x-ratelimit-limit" in create_response.headers
        created = create_response.json()
        assert created["email"] == "rto.operator@example.com"
        assert created["department"] == "rto"
        assert created["manager_id"] == super_admin.id

        duplicate_response = client.post(
            "/api/rbac/users",
            cookies=super_cookies,
            json={
                "full_name": "Duplicate User",
                "email": "rto.operator@example.com",
                "password": "StrongPassword!123",
                "role": "rto_staff",
            },
        )
        assert duplicate_response.status_code == 409

        super_creation_response = client.post(
            "/api/rbac/users",
            cookies=super_cookies,
            json={
                "full_name": "Another Super Admin",
                "email": "root2@example.com",
                "password": "StrongPassword!123",
                "role": "super_admin",
            },
        )
        assert super_creation_response.status_code == 403

        session = TestingSession()
        try:
            created_user = (
                session.query(User)
                .filter(User.email == "rto.operator@example.com")
                .one()
            )
            session.add(
                RefreshToken(
                    user_id=created_user.id,
                    token_jti="test-refresh-token",
                    expires_at=indian_time() + timedelta(days=1),
                    revoked=False,
                )
            )
            session.commit()
            created_user_id = created_user.id
        finally:
            session.close()

        disable_response = client.patch(
            f"/api/rbac/users/{created_user_id}",
            cookies=super_cookies,
            json={"is_active": False},
        )
        assert disable_response.status_code == 200
        assert "x-ratelimit-limit" in disable_response.headers
        assert disable_response.json()["is_active"] is False

        verification = TestingSession()
        try:
            disabled_user = verification.get(User, created_user_id)
            refresh_token = (
                verification.query(RefreshToken)
                .filter(RefreshToken.token_jti == "test-refresh-token")
                .one()
            )
            assert disabled_user.token_version == 1
            assert refresh_token.revoked is True
            assert (
                verification.query(AuditLog)
                .filter(AuditLog.action == "rbac.user.created")
                .count()
                == 1
            )
        finally:
            verification.close()
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()
