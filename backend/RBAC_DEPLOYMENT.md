# INTEL-I RBAC deployment

## Hierarchy

- `super_admin` creates and manages `rto_admin`, `crime_admin`, and `cyber_admin`.
- Each department admin creates and manages only its own staff role.
- Staff cannot create users or elevate roles.
- A department admin cannot manage another department, another admin, or the Super Admin.

## Deploy

1. Back up PostgreSQL.
2. Public signup is not exposed. Keep account creation restricted to the Super Admin page and `/api/rbac/users` API.
3. Run `alembic upgrade head` from the backend directory.
4. If this is a new database, create the first account with `python -m scripts.create_super_admin`.
5. Restart the API. Existing sessions are intentionally invalid after the migration; users must sign in again.
6. Open **User Management** from the navigation bar.

For unattended bootstrap, pass `SUPER_ADMIN_EMAIL`, `SUPER_ADMIN_NAME`, and `SUPER_ADMIN_PASSWORD` only for the one-time command, then remove them from the environment.

## Security behavior

- The API is authoritative; frontend menu filtering is only a usability feature.
- Protected API namespaces are checked against the authenticated role.
- Role, status, and password changes increment `token_version` and revoke every refresh token for that user.
- User-management actions are stored in `audit_logs` without passwords or tokens.
- Public signup has no frontend control or backend endpoint.
