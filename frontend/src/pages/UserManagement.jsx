import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Check,
  CheckCircle2,
  Circle,
  Clipboard,
  Eye,
  EyeOff,
  RefreshCw,
  Search,
  ShieldCheck,
  Sparkles,
  UserCheck,
  UserPlus,
  UserX,
  Users,
} from "lucide-react";
import toast from "react-hot-toast";
import api from "../api/axios";

const EMPTY_FORM = {
  full_name: "",
  email: "",
  password: "",
  confirm_password: "",
  role: "",
};

const ROLE_LABELS = {
  rto_admin: "RTO Admin",
  rto_staff: "RTO Staff",
  crime_admin: "Crime Admin",
  crime_staff: "Crime Staff",
  cyber_admin: "Cyber Admin",
  cyber_staff: "Cyber Staff",
};

const PASSWORD_RULES = [
  { label: "12+ characters", test: (value) => value.length >= 12 },
  { label: "Uppercase letter", test: (value) => /[A-Z]/.test(value) },
  { label: "Lowercase letter", test: (value) => /[a-z]/.test(value) },
  { label: "Number", test: (value) => /\d/.test(value) },
  { label: "Special symbol", test: (value) => /[^\w\s]/.test(value) },
];

const errorMessage = (error, fallback) => {
  const detail = error?.response?.data?.detail;

  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail[0]?.msg) return detail[0].msg;

  return fallback;
};

const isStrongPassword = (password) =>
  PASSWORD_RULES.every((rule) => rule.test(password));

const initials = (name) =>
  String(name || "U")
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "U";

const departmentLabel = (department) => {
  if (!department) return "All";

  return `${department.charAt(0).toUpperCase()}${department.slice(1)}`;
};

const secureRandomIndex = (length) => {
  const value = new Uint32Array(1);
  window.crypto.getRandomValues(value);
  return value[0] % length;
};

const generateSecurePassword = () => {
  const groups = [
    "ABCDEFGHJKLMNPQRSTUVWXYZ",
    "abcdefghijkmnopqrstuvwxyz",
    "23456789",
    "!@#$%&*?",
  ];

  const allCharacters = groups.join("");

  const characters = groups.map(
    (group) => group[secureRandomIndex(group.length)],
  );

  while (characters.length < 16) {
    characters.push(
      allCharacters[secureRandomIndex(allCharacters.length)],
    );
  }

  for (let index = characters.length - 1; index > 0; index -= 1) {
    const swapIndex = secureRandomIndex(index + 1);

    [characters[index], characters[swapIndex]] = [
      characters[swapIndex],
      characters[index],
    ];
  }

  return characters.join("");
};

export default function UserManagement() {
  const [users, setUsers] = useState([]);
  const [roles, setRoles] = useState([]);
  const [form, setForm] = useState(EMPTY_FORM);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [updatingUserId, setUpdatingUserId] = useState(null);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] =
    useState(false);

  const load = useCallback(async ({ silent = false } = {}) => {
    if (!silent) setLoading(true);

    try {
      const [usersResponse, rolesResponse] = await Promise.all([
        api.get("/api/rbac/users"),
        api.get("/api/rbac/roles"),
      ]);

      const nextUsers = Array.isArray(usersResponse.data)
        ? usersResponse.data
        : [];

      const assignableRoles = Array.isArray(
        rolesResponse.data?.assignable_roles,
      )
        ? rolesResponse.data.assignable_roles
        : [];

      setUsers(nextUsers);
      setRoles(assignableRoles);

      setForm((previous) => ({
        ...previous,
        role:
          previous.role && assignableRoles.includes(previous.role)
            ? previous.role
            : assignableRoles[0] || "",
      }));
    } catch (error) {
      toast.error(
        errorMessage(error, "Could not load user accounts"),
      );
      throw error;
    } finally {
      if (!silent) setLoading(false);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load().catch(() => undefined);
  }, [load]);

  const summary = useMemo(() => {
    const active = users.filter((user) => user.is_active).length;

    return {
      total: users.length,
      active,
      disabled: users.length - active,
    };
  }, [users]);

  const filteredUsers = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();

    if (!normalizedQuery) return users;

    return users.filter((user) =>
      [
        user.full_name,
        user.email,
        ROLE_LABELS[user.role] || user.role,
        user.department,
      ].some((value) =>
        String(value || "")
          .toLowerCase()
          .includes(normalizedQuery),
      ),
    );
  }, [query, users]);

  const setField = (event) => {
    const { name, value } = event.target;

    setForm((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const generatePassword = () => {
    try {
      const password = generateSecurePassword();

      setForm((previous) => ({
        ...previous,
        password,
        confirm_password: password,
      }));

      setShowPassword(true);
      setShowConfirmPassword(true);
      toast.success("Secure temporary password generated");
    } catch {
      toast.error("Secure password generation is unavailable");
    }
  };

  const copyPassword = async () => {
    if (!form.password) {
      toast.error("Generate or enter a password first");
      return;
    }

    try {
      await navigator.clipboard.writeText(form.password);
      toast.success("Temporary password copied");
    } catch {
      toast.error("Could not copy the password");
    }
  };

  const createUser = async (event) => {
    event.preventDefault();

    if (creating) return;

    if (!isStrongPassword(form.password)) {
      toast.error("Complete all password requirements");
      return;
    }

    if (form.password !== form.confirm_password) {
      toast.error("Temporary passwords do not match");
      return;
    }

    try {
      setCreating(true);

      await api.post("/api/rbac/users", {
        full_name: form.full_name.trim(),
        email: form.email.trim().toLowerCase(),
        password: form.password,
        role: form.role,
      });

      setForm({
        ...EMPTY_FORM,
        role: roles[0] || "",
      });

      setShowPassword(false);
      setShowConfirmPassword(false);

      await load({ silent: true });

      toast.success("Account created successfully");
    } catch (error) {
      toast.error(
        errorMessage(error, "Account creation failed"),
      );
    } finally {
      setCreating(false);
    }
  };

  const toggleUser = async (user) => {
    const nextActive = !user.is_active;
    const action = nextActive ? "enable" : "disable";

    if (
      !nextActive &&
      !window.confirm(
        `Disable ${user.full_name}? Their active sessions will be revoked immediately.`,
      )
    ) {
      return;
    }

    try {
      setUpdatingUserId(user.id);

      const response = await api.patch(
        `/api/rbac/users/${user.id}`,
        {
          is_active: nextActive,
        },
      );

      setUsers((previous) =>
        previous.map((item) =>
          item.id === user.id ? response.data : item,
        ),
      );

      toast.success(`Account ${action}d successfully`);
    } catch (error) {
      toast.error(
        errorMessage(error, `Could not ${action} account`),
      );
    } finally {
      setUpdatingUserId(null);
    }
  };

  const renderStatus = (user) => (
    <span
      className={
        user.is_active
          ? "inline-flex items-center gap-1.5 rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2.5 py-1 text-xs font-semibold text-emerald-400"
          : "inline-flex items-center gap-1.5 rounded-full border border-red-500/20 bg-red-500/10 px-2.5 py-1 text-xs font-semibold text-red-400"
      }
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          user.is_active ? "bg-emerald-400" : "bg-red-400"
        }`}
      />

      {user.is_active ? "Active" : "Disabled"}
    </span>
  );

  const renderAction = (user) => (
    <button
      type="button"
      onClick={() => toggleUser(user)}
      disabled={updatingUserId === user.id}
      className={`inline-flex min-w-20 items-center justify-center rounded-lg border px-3 py-2 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
        user.is_active
          ? "border-red-500/25 bg-red-500/5 text-red-300 hover:border-red-500/50 hover:bg-red-500/10"
          : "border-emerald-500/25 bg-emerald-500/5 text-emerald-300 hover:border-emerald-500/50 hover:bg-emerald-500/10"
      }`}
    >
      {updatingUserId === user.id
        ? "Updating..."
        : user.is_active
          ? "Disable"
          : "Enable"}
    </button>
  );

  return (
    <section
      className="mx-auto w-full max-w-[1600px] space-y-5 pb-8"
      aria-busy={loading}
    >
      <header className="rounded-2xl border border-slate-800 bg-gradient-to-r from-[#07111d] to-[#0a1726] p-5 shadow-xl shadow-black/10 sm:p-6">
        <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-center">
          <div className="flex items-center gap-4">
            <div className="flex h-13 w-13 shrink-0 items-center justify-center rounded-2xl border border-cyan-500/30 bg-cyan-500/10 shadow-lg shadow-cyan-950/20">
              <ShieldCheck
                className="text-cyan-400"
                size={29}
              />
            </div>

            <div>
              <p className="mb-1 text-xs font-bold tracking-[0.18em] text-cyan-400 uppercase">
                Access Control
              </p>

              <h1 className="text-2xl font-bold text-white sm:text-3xl">
                User Management
              </h1>

              <p className="mt-1 text-sm text-slate-400">
                Create and control authorized INTEL-I operator
                accounts.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <span className="inline-flex items-center gap-2 rounded-xl border border-amber-500/20 bg-amber-500/5 px-3 py-2 text-xs font-semibold text-amber-300">
              <ShieldCheck size={15} />
              Super Administrator only
            </span>

            <button
              type="button"
              onClick={() =>
                load().catch(() => undefined)
              }
              disabled={loading}
              className="inline-flex items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-900/60 px-4 py-2 text-sm font-semibold text-slate-200 transition hover:border-cyan-500/60 hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RefreshCw
                size={16}
                className={loading ? "animate-spin" : ""}
              />
              Refresh
            </button>
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
        {[
          {
            label: "Total accounts",
            value: summary.total,
            icon: Users,
            color: "text-blue-400",
            panel: "border-blue-500/20 bg-blue-500/5",
          },
          {
            label: "Active accounts",
            value: summary.active,
            icon: UserCheck,
            color: "text-emerald-400",
            panel:
              "border-emerald-500/20 bg-emerald-500/5",
          },
          {
            label: "Disabled accounts",
            value: summary.disabled,
            icon: UserX,
            color: "text-red-400",
            panel: "border-red-500/20 bg-red-500/5",
          },
        ].map((item) => {
          const Icon = item.icon;

          return (
            <div
              key={item.label}
              className={`flex items-center gap-3 rounded-2xl border p-4 ${item.panel}`}
            >
              <div className="rounded-xl bg-slate-950/60 p-2.5">
                <Icon
                  size={21}
                  className={item.color}
                />
              </div>

              <div>
                <p className="text-xs font-medium text-slate-400">
                  {item.label}
                </p>

                <p className="mt-0.5 text-2xl font-bold text-white">
                  {item.value}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      <div className="grid items-start gap-5 xl:grid-cols-[460px_minmax(0,1fr)]">
        <form
          onSubmit={createUser}
          className="overflow-hidden rounded-2xl border border-slate-800 bg-[#07111d] shadow-xl shadow-black/10"
        >
          <div className="border-b border-slate-800 bg-slate-950/35 px-5 py-4 sm:px-6">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-cyan-500/10 p-2 text-cyan-400">
                <UserPlus size={19} />
              </div>

              <div>
                <h2 className="font-semibold text-white">
                  Create account
                </h2>

                <p className="text-xs text-slate-500">
                  Add a verified user and assign their role.
                </p>
              </div>
            </div>
          </div>

          <div className="space-y-4 p-5 sm:p-6">
            <label className="block text-sm font-medium text-slate-300">
              Full name

              <input
                required
                name="full_name"
                type="text"
                minLength={2}
                maxLength={150}
                autoComplete="off"
                value={form.full_name}
                onChange={setField}
                disabled={creating}
                placeholder="Enter full name"
                className="mt-2 w-full rounded-xl border border-slate-700 bg-[#030914] px-3.5 py-3 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/10 disabled:opacity-60"
              />
            </label>

            <label className="block text-sm font-medium text-slate-300">
              Email address

              <input
                required
                name="email"
                type="email"
                maxLength={254}
                autoComplete="off"
                value={form.email}
                onChange={setField}
                disabled={creating}
                placeholder="name@department.gov.in"
                className="mt-2 w-full rounded-xl border border-slate-700 bg-[#030914] px-3.5 py-3 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/10 disabled:opacity-60"
              />
            </label>

            <label className="block text-sm font-medium text-slate-300">
              Operational role

              <select
                required
                name="role"
                value={form.role}
                onChange={setField}
                disabled={
                  creating || roles.length === 0
                }
                className="mt-2 w-full rounded-xl border border-slate-700 bg-[#030914] px-3.5 py-3 text-sm text-white outline-none transition focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/10 disabled:opacity-60"
              >
                {roles.map((role) => (
                  <option key={role} value={role}>
                    {ROLE_LABELS[role] || role}
                  </option>
                ))}
              </select>
            </label>

            <div>
              <div className="flex flex-wrap items-center justify-between gap-2">
                <label
                  htmlFor="temporary-password"
                  className="text-sm font-medium text-slate-300"
                >
                  Temporary password
                </label>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={generatePassword}
                    disabled={creating}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/25 bg-cyan-500/5 px-2.5 py-1.5 text-xs font-semibold text-cyan-300 hover:bg-cyan-500/10 disabled:opacity-50"
                  >
                    <Sparkles size={13} />
                    Generate
                  </button>

                  <button
                    type="button"
                    onClick={copyPassword}
                    disabled={
                      creating || !form.password
                    }
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 px-2.5 py-1.5 text-xs font-semibold text-slate-300 hover:border-slate-600 hover:text-white disabled:opacity-40"
                  >
                    <Clipboard size={13} />
                    Copy
                  </button>
                </div>
              </div>

              <div className="relative mt-2">
                <input
                  id="temporary-password"
                  required
                  name="password"
                  type={
                    showPassword ? "text" : "password"
                  }
                  minLength={12}
                  maxLength={128}
                  autoComplete="new-password"
                  value={form.password}
                  onChange={setField}
                  disabled={creating}
                  placeholder="Enter or generate a secure password"
                  className="w-full rounded-xl border border-slate-700 bg-[#030914] px-3.5 py-3 pr-12 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/10 disabled:opacity-60"
                />

                <button
                  type="button"
                  onClick={() =>
                    setShowPassword(
                      (previous) => !previous,
                    )
                  }
                  className="absolute inset-y-0 right-0 flex w-12 items-center justify-center text-slate-400 hover:text-white"
                  aria-label={
                    showPassword
                      ? "Hide password"
                      : "Show password"
                  }
                >
                  {showPassword ? (
                    <EyeOff size={18} />
                  ) : (
                    <Eye size={18} />
                  )}
                </button>
              </div>
            </div>

            <label className="block text-sm font-medium text-slate-300">
              Confirm temporary password

              <div className="relative mt-2">
                <input
                  required
                  name="confirm_password"
                  type={
                    showConfirmPassword
                      ? "text"
                      : "password"
                  }
                  minLength={12}
                  maxLength={128}
                  autoComplete="new-password"
                  value={form.confirm_password}
                  onChange={setField}
                  disabled={creating}
                  placeholder="Re-enter temporary password"
                  className="w-full rounded-xl border border-slate-700 bg-[#030914] px-3.5 py-3 pr-12 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/10 disabled:opacity-60"
                />

                <button
                  type="button"
                  onClick={() =>
                    setShowConfirmPassword(
                      (previous) => !previous,
                    )
                  }
                  className="absolute inset-y-0 right-0 flex w-12 items-center justify-center text-slate-400 hover:text-white"
                  aria-label={
                    showConfirmPassword
                      ? "Hide password"
                      : "Show password"
                  }
                >
                  {showConfirmPassword ? (
                    <EyeOff size={18} />
                  ) : (
                    <Eye size={18} />
                  )}
                </button>
              </div>
            </label>

            <div className="rounded-xl border border-slate-800 bg-slate-950/45 p-3.5">
              <p className="mb-2.5 text-xs font-semibold text-slate-400">
                Password requirements
              </p>

              <div className="grid grid-cols-2 gap-x-3 gap-y-2">
                {PASSWORD_RULES.map((rule) => {
                  const passed = rule.test(form.password);

                  return (
                    <div
                      key={rule.label}
                      className={`flex items-center gap-1.5 text-[11px] ${
                        passed
                          ? "text-emerald-400"
                          : "text-slate-500"
                      }`}
                    >
                      {passed ? (
                        <CheckCircle2 size={13} />
                      ) : (
                        <Circle size={13} />
                      )}

                      {rule.label}
                    </div>
                  );
                })}
              </div>
            </div>

            <p className="text-xs leading-5 text-slate-500">
              Share the temporary password using a secure
              channel. INTEL-I never returns stored passwords
              after account creation.
            </p>

            <button
              type="submit"
              disabled={
                creating || !form.role || loading
              }
              className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 px-4 py-3 font-semibold text-white shadow-lg shadow-blue-950/20 transition hover:from-cyan-500 hover:to-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {creating ? (
                <>
                  <RefreshCw
                    size={17}
                    className="animate-spin"
                  />
                  Creating account...
                </>
              ) : (
                <>
                  <UserPlus size={17} />
                  Create account
                </>
              )}
            </button>
          </div>
        </form>

        <div className="min-w-0 overflow-hidden rounded-2xl border border-slate-800 bg-[#07111d] shadow-xl shadow-black/10">
          <div className="flex flex-col gap-4 border-b border-slate-800 bg-slate-950/35 p-5 sm:flex-row sm:items-center sm:justify-between sm:px-6">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-blue-500/10 p-2 text-blue-400">
                <Users size={19} />
              </div>

              <div>
                <div className="flex items-center gap-2">
                  <h2 className="font-semibold text-white">
                    User accounts
                  </h2>

                  <span className="rounded-full border border-slate-700 bg-slate-800 px-2 py-0.5 text-[11px] font-semibold text-slate-300">
                    {users.length}
                  </span>
                </div>

                <p className="text-xs text-slate-500">
                  Search and control existing user access.
                </p>
              </div>
            </div>

            <label className="relative block w-full sm:w-80">
              <span className="sr-only">Search users</span>

              <Search
                size={16}
                className="pointer-events-none absolute top-1/2 left-3.5 -translate-y-1/2 text-slate-500"
              />

              <input
                type="search"
                value={query}
                onChange={(event) =>
                  setQuery(event.target.value)
                }
                placeholder="Search name, email or role"
                className="w-full rounded-xl border border-slate-700 bg-[#030914] py-2.5 pr-3 pl-10 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-cyan-500 focus:ring-2 focus:ring-cyan-500/10"
              />
            </label>
          </div>

          <div className="hidden md:block">
            <table className="w-full table-fixed text-left text-sm">
              <colgroup>
                <col className="w-[34%]" />
                <col className="w-[21%]" />
                <col className="w-[16%]" />
                <col className="w-[15%]" />
                <col className="w-[14%]" />
              </colgroup>

              <thead className="bg-[#040b16] text-xs text-slate-400 uppercase">
                <tr>
                  <th className="px-5 py-3.5 font-semibold">
                    User
                  </th>

                  <th className="px-4 py-3.5 font-semibold">
                    Role
                  </th>

                  <th className="px-4 py-3.5 font-semibold">
                    Department
                  </th>

                  <th className="px-4 py-3.5 font-semibold">
                    Status
                  </th>

                  <th className="px-5 py-3.5 text-right font-semibold">
                    Action
                  </th>
                </tr>
              </thead>

              <tbody>
                {filteredUsers.map((user) => (
                  <tr
                    key={user.id}
                    className="border-t border-slate-800/90 text-slate-200 transition hover:bg-slate-900/40"
                  >
                    <td className="px-5 py-4">
                      <div className="flex min-w-0 items-center gap-3">
                        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-blue-500/20 bg-blue-500/10 text-xs font-bold text-blue-300">
                          {initials(user.full_name)}
                        </div>

                        <div className="min-w-0">
                          <p className="truncate font-semibold text-white">
                            {user.full_name}
                          </p>

                          <p
                            className="mt-0.5 truncate text-xs text-slate-500"
                            title={user.email}
                          >
                            {user.email}
                          </p>
                        </div>
                      </div>
                    </td>

                    <td className="px-4 py-4">
                      <span className="block truncate text-slate-200">
                        {ROLE_LABELS[user.role] ||
                          user.role}
                      </span>
                    </td>

                    <td className="px-4 py-4 text-slate-400">
                      {departmentLabel(user.department)}
                    </td>

                    <td className="px-4 py-4">
                      {renderStatus(user)}
                    </td>

                    <td className="px-5 py-4 text-right">
                      {renderAction(user)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div className="space-y-3 p-4 md:hidden">
            {filteredUsers.map((user) => (
              <article
                key={user.id}
                className="rounded-xl border border-slate-800 bg-slate-950/35 p-4"
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex min-w-0 items-center gap-3">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-blue-500/20 bg-blue-500/10 text-xs font-bold text-blue-300">
                      {initials(user.full_name)}
                    </div>

                    <div className="min-w-0">
                      <p className="truncate font-semibold text-white">
                        {user.full_name}
                      </p>

                      <p className="truncate text-xs text-slate-500">
                        {user.email}
                      </p>
                    </div>
                  </div>

                  {renderStatus(user)}
                </div>

                <div className="mt-4 grid grid-cols-2 gap-3 rounded-lg bg-slate-950/60 p-3 text-xs">
                  <div>
                    <p className="text-slate-500">Role</p>

                    <p className="mt-1 text-slate-200">
                      {ROLE_LABELS[user.role] ||
                        user.role}
                    </p>
                  </div>

                  <div>
                    <p className="text-slate-500">
                      Department
                    </p>

                    <p className="mt-1 text-slate-200">
                      {departmentLabel(user.department)}
                    </p>
                  </div>
                </div>

                <div className="mt-3 flex justify-end">
                  {renderAction(user)}
                </div>
              </article>
            ))}
          </div>

          {!loading && filteredUsers.length === 0 && (
            <div className="flex min-h-64 flex-col items-center justify-center px-6 py-12 text-center">
              <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-4 text-slate-500">
                <Users size={30} />
              </div>

              <p className="mt-4 font-semibold text-slate-300">
                {query
                  ? "No matching accounts"
                  : "No user accounts yet"}
              </p>

              <p className="mt-1 max-w-sm text-sm text-slate-500">
                {query
                  ? "Try a different name, email address, role or department."
                  : "Create the first managed account using the form."}
              </p>
            </div>
          )}

          {loading && (
            <div className="flex min-h-64 items-center justify-center gap-2 text-sm text-slate-500">
              <RefreshCw
                size={17}
                className="animate-spin"
              />
              Loading user accounts...
            </div>
          )}

          {!loading && filteredUsers.length > 0 && (
            <div className="flex flex-col gap-2 border-t border-slate-800 bg-slate-950/25 px-5 py-3 text-xs text-slate-500 sm:flex-row sm:items-center sm:justify-between">
              <span>
                Showing {filteredUsers.length} of{" "}
                {users.length} accounts
              </span>

              <span className="inline-flex items-center gap-1.5">
                <Check
                  size={13}
                  className="text-emerald-400"
                />
                Account changes revoke existing sessions
              </span>
            </div>
          )}
        </div>
      </div>
    </section>
  );
}