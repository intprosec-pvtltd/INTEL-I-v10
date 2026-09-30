import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  Copy,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  Lock,
  RefreshCw,
  Shield,
  ShieldCheck,
  ShieldOff,
  Smartphone,
  X,
} from "lucide-react";
import toast from "react-hot-toast";
import api from "../api/axios";

const EMPTY_PASSWORD_FORM = {
  currentPassword: "",
  newPassword: "",
  confirmPassword: "",
};

const normalizeError = (error, fallback) => {
  const detail = error?.response?.data?.detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((item) => item?.msg)
      .filter(Boolean)
      .join(", ");
  }

  const message = error?.response?.data?.message;

  if (typeof message === "string" && message.trim()) {
    return message;
  }

  return fallback;
};

const normalizeTotpCode = (value) =>
  String(value || "")
    .replace(/\D/g, "")
    .slice(0, 6);

const SecurityCard = ({
  icon: Icon,
  title,
  description,
  children,
}) => (
  <section className="overflow-hidden rounded-2xl border border-slate-800 bg-[#07111d] shadow-xl shadow-black/10">
    <div className="border-b border-slate-800 px-5 py-5 sm:px-6">
      <div className="flex items-start gap-3">
        <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-blue-500/25 bg-blue-500/10">
          <Icon size={21} className="text-blue-400" />
        </div>

        <div className="min-w-0">
          <h2 className="text-lg font-semibold text-white">
            {title}
          </h2>

          <p className="mt-1 text-sm leading-6 text-slate-400">
            {description}
          </p>
        </div>
      </div>
    </div>

    <div className="p-5 sm:p-6">{children}</div>
  </section>
);

const StatusBadge = ({ enabled }) => (
  <span
    className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-semibold ${
      enabled
        ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300"
        : "border-amber-500/30 bg-amber-500/10 text-amber-300"
    }`}
  >
    <span
      className={`h-2 w-2 rounded-full ${
        enabled ? "bg-emerald-400" : "bg-amber-400"
      }`}
    />

    {enabled ? "MFA Enabled" : "MFA Disabled"}
  </span>
);

const PasswordInput = ({
  id,
  label,
  value,
  onChange,
  show,
  onToggle,
  autoComplete,
  disabled,
}) => (
  <div>
    <label
      htmlFor={id}
      className="mb-2 block text-sm font-medium text-slate-300"
    >
      {label}
    </label>

    <div className="flex h-11 items-center gap-2 rounded-xl border border-slate-700 bg-[#030a12] px-3 focus-within:border-blue-500">
      <Lock size={17} className="shrink-0 text-slate-500" />

      <input
        id={id}
        type={show ? "text" : "password"}
        value={value}
        onChange={onChange}
        autoComplete={autoComplete}
        maxLength={128}
        required
        disabled={disabled}
        className="h-full min-w-0 flex-1 bg-transparent text-sm text-white outline-none placeholder:text-slate-600 disabled:cursor-not-allowed"
      />

      <button
        type="button"
        disabled={disabled}
        onClick={onToggle}
        aria-label={show ? `Hide ${label}` : `Show ${label}`}
        className="text-slate-500 transition hover:text-white disabled:cursor-not-allowed disabled:opacity-50"
      >
        {show ? <EyeOff size={17} /> : <Eye size={17} />}
      </button>
    </div>
  </div>
);

const AccountSecurity = () => {
  const [initialLoading, setInitialLoading] = useState(true);

  const [mfaStatus, setMfaStatus] = useState({
    enabled: false,
  });

  const [setupData, setSetupData] = useState(null);
  const [setupCode, setSetupCode] = useState("");
  const [recoveryCodes, setRecoveryCodes] = useState([]);

  const [setupLoading, setSetupLoading] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [regenerateLoading, setRegenerateLoading] =
    useState(false);
  const [disableLoading, setDisableLoading] = useState(false);

  const [disableCode, setDisableCode] = useState("");
  const [disablePassword, setDisablePassword] = useState("");
  const [regenerateCode, setRegenerateCode] = useState("");
  const [regeneratePassword, setRegeneratePassword] = useState("");
  const [passwordMfaCode, setPasswordMfaCode] = useState("");

  const [passwordForm, setPasswordForm] = useState(
    EMPTY_PASSWORD_FORM,
  );

  const [passwordLoading, setPasswordLoading] =
    useState(false);

  const [passwordVisibility, setPasswordVisibility] =
    useState({
      current: false,
      next: false,
      confirm: false,
    });

  const mfaEnabled = Boolean(mfaStatus?.enabled);

  const provisioningUri =
    setupData?.provisioning_uri ||
    setupData?.otpauth_uri ||
    "";

  const setupSecret =
    setupData?.secret ||
    setupData?.manual_entry_key ||
    "";

  const passwordMismatch = useMemo(
    () =>
      Boolean(
        passwordForm.confirmPassword &&
          passwordForm.newPassword !==
            passwordForm.confirmPassword,
      ),
    [
      passwordForm.confirmPassword,
      passwordForm.newPassword,
    ],
  );

  const loadMfaStatus = useCallback(async () => {
    try {
      const response = await api.get("/auth/mfa/status");

      setMfaStatus({
        ...response.data,
        enabled: Boolean(response.data?.enabled),
      });
    } catch (error) {
      toast.error(
        normalizeError(
          error,
          "Unable to load MFA security status",
        ),
      );

      throw error;
    }
  }, []);

  useEffect(() => {
    let mounted = true;

    const load = async () => {
      try {
        await loadMfaStatus();
      } catch {
        // Toast is handled by loadMfaStatus.
      } finally {
        if (mounted) {
          setInitialLoading(false);
        }
      }
    };

    load();

    return () => {
      mounted = false;
    };
  }, [loadMfaStatus]);

  const beginMfaSetup = async () => {
    if (setupLoading) return;

    try {
      setSetupLoading(true);
      setRecoveryCodes([]);
      setSetupCode("");

      const response = await api.post(
        "/auth/mfa/setup",
        {},
      );

      if (
        !response.data?.secret ||
        !response.data?.provisioning_uri
      ) {
        throw new Error(
          "Server did not return MFA enrollment information",
        );
      }

      setSetupData(response.data);

      toast.success(
        "MFA setup started. Add INTEL-I to your authenticator app.",
      );
    } catch (error) {
      toast.error(
        normalizeError(error, "Unable to start MFA setup"),
      );
    } finally {
      setSetupLoading(false);
    }
  };

  const confirmMfaSetup = async (event) => {
    event.preventDefault();

    if (confirmLoading) return;

    const code = normalizeTotpCode(setupCode);

    if (code.length !== 6) {
      toast.error(
        "Enter the 6-digit code from your authenticator app",
      );
      return;
    }

    try {
      setConfirmLoading(true);

      const response = await api.post(
        "/auth/mfa/setup/confirm",
        {
          totp_code: code,
        },
      );

      const codes = Array.isArray(
        response.data?.recovery_codes,
      )
        ? response.data.recovery_codes
        : [];

      setRecoveryCodes(codes);
      setSetupData(null);
      setSetupCode("");

      await loadMfaStatus();

      toast.success(
        "Multi-factor authentication enabled successfully",
      );
    } catch (error) {
      toast.error(
        normalizeError(
          error,
          "Unable to verify authenticator code",
        ),
      );
    } finally {
      setConfirmLoading(false);
    }
  };

  const regenerateRecoveryCodes = async (event) => {
    event.preventDefault();

    if (regenerateLoading) return;

    const code = normalizeTotpCode(regenerateCode);

    if (!regeneratePassword) {
      toast.error("Enter your current account password");
      return;
    }

    if (code.length !== 6) {
      toast.error(
        "Enter your current 6-digit authenticator code",
      );
      return;
    }

    try {
      setRegenerateLoading(true);

      const response = await api.post(
        "/auth/mfa/recovery/regenerate",
        {
          current_password: regeneratePassword,
          totp_code: code,
        },
      );

      const codes = Array.isArray(
        response.data?.recovery_codes,
      )
        ? response.data.recovery_codes
        : [];

      if (codes.length === 0) {
        throw new Error(
          "Server did not return new recovery codes",
        );
      }

      setRecoveryCodes(codes);
      setRegenerateCode("");
      setRegeneratePassword("");

      toast.success(
        "New recovery codes generated. Previous codes are no longer valid.",
      );
    } catch (error) {
      toast.error(
        normalizeError(
          error,
          "Unable to regenerate recovery codes",
        ),
      );
    } finally {
      setRegenerateLoading(false);
    }
  };

  const disableMfa = async (event) => {
    event.preventDefault();

    if (disableLoading) return;

    const code = normalizeTotpCode(disableCode);

    if (!disablePassword) {
      toast.error("Enter your current account password");
      return;
    }

    if (code.length !== 6) {
      toast.error(
        "Enter your current 6-digit authenticator code",
      );
      return;
    }

    const confirmed = window.confirm(
      "Disable multi-factor authentication for your INTEL-I account?",
    );

    if (!confirmed) return;

    try {
      setDisableLoading(true);

      await api.post("/auth/mfa/disable", {
        current_password: disablePassword,
        mfa_code: code,
      });

      setDisableCode("");
      setDisablePassword("");
      setSetupCode("");
      setSetupData(null);
      setRecoveryCodes([]);

      await loadMfaStatus();

      toast.success(
        "Multi-factor authentication disabled",
      );
    } catch (error) {
      toast.error(
        normalizeError(error, "Unable to disable MFA"),
      );
    } finally {
      setDisableLoading(false);
    }
  };

  const handlePasswordChange = (field, value) => {
    setPasswordForm((previous) => ({
      ...previous,
      [field]: value,
    }));
  };

  const changePassword = async (event) => {
    event.preventDefault();

    if (passwordLoading) return;

    const {
      currentPassword,
      newPassword,
      confirmPassword,
    } = passwordForm;

    if (
      !currentPassword ||
      !newPassword ||
      !confirmPassword
    ) {
      toast.error("Complete all password fields");
      return;
    }

    if (newPassword !== confirmPassword) {
      toast.error("New passwords do not match");
      return;
    }

    if (currentPassword === newPassword) {
      toast.error(
        "Your new password must be different from your current password",
      );
      return;
    }

    const mfaCode = normalizeTotpCode(passwordMfaCode);

    if (mfaEnabled && mfaCode.length !== 6) {
      toast.error(
        "Enter your current 6-digit authenticator code",
      );
      return;
    }

    try {
      setPasswordLoading(true);

      await api.post("/auth/password/change", {
        current_password: currentPassword,
        new_password: newPassword,
        confirm_password: confirmPassword,
        mfa_code: mfaEnabled ? mfaCode : null,
      });

      setPasswordForm(EMPTY_PASSWORD_FORM);
      setPasswordMfaCode("");

      toast.success(
        "Password changed successfully. Please sign in again.",
      );

      window.setTimeout(() => {
        localStorage.removeItem("user");
        localStorage.removeItem("csrf_token");
        localStorage.removeItem("login_expiry");
        window.location.assign("/");
      }, 800);
    } catch (error) {
      toast.error(
        normalizeError(error, "Unable to change password"),
      );
    } finally {
      setPasswordLoading(false);
    }
  };

  const copyText = async (value, label) => {
    try {
      await navigator.clipboard.writeText(value);
      toast.success(`${label} copied`);
    } catch {
      toast.error(`Unable to copy ${label.toLowerCase()}`);
    }
  };

  const copyRecoveryCodes = async () => {
    if (recoveryCodes.length === 0) return;

    await copyText(
      recoveryCodes.join("\n"),
      "Recovery codes",
    );
  };

  if (initialLoading) {
    return (
      <main className="min-h-[calc(100vh-74px)] bg-[#01050a] px-4 py-8 text-white sm:px-6 lg:px-8">
        <div className="mx-auto flex min-h-[420px] max-w-6xl items-center justify-center">
          <div className="flex items-center gap-3 text-slate-400">
            <Loader2
              size={22}
              className="animate-spin text-blue-400"
            />
            Loading account security...
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-[calc(100vh-74px)] bg-[#01050a] px-4 py-7 text-white sm:px-6 lg:px-8">
      <div className="mx-auto w-full max-w-6xl">
        <div className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
          <div>
            <div className="mb-2 flex items-center gap-2 text-sm font-semibold uppercase tracking-[0.18em] text-blue-400">
              <ShieldCheck size={17} />
              Account Protection
            </div>

            <h1 className="text-2xl font-bold text-white sm:text-3xl">
              Security Settings
            </h1>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Manage multi-factor authentication, recovery
              credentials and your INTEL-I account password.
            </p>
          </div>

          <StatusBadge enabled={mfaEnabled} />
        </div>

        <div className="grid gap-6">
          <SecurityCard
            icon={mfaEnabled ? ShieldCheck : Shield}
            title="Multi-Factor Authentication"
            description="Protect your account with a time-based one-time password from an authenticator application."
          >
            {!mfaEnabled && !setupData && (
              <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
                <div className="flex items-start gap-3">
                  <div className="mt-0.5 rounded-lg bg-amber-500/10 p-2">
                    <AlertTriangle
                      size={18}
                      className="text-amber-400"
                    />
                  </div>

                  <div>
                    <p className="font-medium text-white">
                      MFA is not enabled
                    </p>

                    <p className="mt-1 max-w-xl text-sm leading-6 text-slate-400">
                      Enable MFA to require an additional
                      authenticator code after your password when
                      signing in.
                    </p>
                  </div>
                </div>

                <button
                  type="button"
                  disabled={setupLoading}
                  onClick={beginMfaSetup}
                  className="inline-flex h-11 shrink-0 items-center justify-center gap-2 rounded-xl bg-blue-600 px-5 text-sm font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {setupLoading ? (
                    <Loader2
                      size={17}
                      className="animate-spin"
                    />
                  ) : (
                    <Smartphone size={17} />
                  )}

                  {setupLoading
                    ? "Starting..."
                    : "Enable MFA"}
                </button>
              </div>
            )}

            {!mfaEnabled && setupData && (
              <div>
                <div className="rounded-xl border border-blue-500/20 bg-blue-500/5 p-4">
                  <div className="flex items-start gap-3">
                    <Smartphone
                      size={21}
                      className="mt-0.5 shrink-0 text-blue-400"
                    />

                    <div>
                      <p className="font-semibold text-white">
                        Add INTEL-I to your authenticator app
                      </p>

                      <p className="mt-1 text-sm leading-6 text-slate-400">
                        Use Google Authenticator, Microsoft
                        Authenticator or another compatible TOTP
                        application.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="mt-5">
                  <p className="text-sm font-medium text-slate-300">
                    Manual setup key
                  </p>

                  <div className="mt-2 flex items-center gap-2 rounded-xl border border-slate-700 bg-[#030a12] p-3">
                    <code className="min-w-0 flex-1 break-all text-sm font-semibold tracking-wider text-cyan-300">
                      {setupSecret}
                    </code>

                    <button
                      type="button"
                      onClick={() =>
                        copyText(
                          setupSecret,
                          "Authenticator key",
                        )
                      }
                      className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-800 hover:text-white"
                      aria-label="Copy authenticator key"
                    >
                      <Copy size={17} />
                    </button>
                  </div>
                </div>

                {provisioningUri && (
                  <div className="mt-4">
                    <p className="text-sm font-medium text-slate-300">
                      Provisioning URI
                    </p>

                    <div className="mt-2 flex items-center gap-2 rounded-xl border border-slate-800 bg-[#030a12] p-3">
                      <code className="min-w-0 flex-1 break-all text-xs text-slate-500">
                        {provisioningUri}
                      </code>

                      <button
                        type="button"
                        onClick={() =>
                          copyText(
                            provisioningUri,
                            "Provisioning URI",
                          )
                        }
                        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-800 hover:text-white"
                        aria-label="Copy provisioning URI"
                      >
                        <Copy size={17} />
                      </button>
                    </div>
                  </div>
                )}

                <form
                  onSubmit={confirmMfaSetup}
                  className="mt-6"
                >
                  <label
                    htmlFor="mfa-setup-code"
                    className="mb-2 block text-sm font-medium text-slate-300"
                  >
                    6-digit authenticator code
                  </label>

                  <div className="flex flex-col gap-3 sm:flex-row">
                    <input
                      id="mfa-setup-code"
                      type="text"
                      inputMode="numeric"
                      autoComplete="one-time-code"
                      value={setupCode}
                      onChange={(event) =>
                        setSetupCode(
                          normalizeTotpCode(
                            event.target.value,
                          ),
                        )
                      }
                      maxLength={6}
                      placeholder="000000"
                      required
                      disabled={confirmLoading}
                      className="h-11 min-w-0 flex-1 rounded-xl border border-slate-700 bg-[#030a12] px-4 text-center text-lg tracking-[0.35em] text-white outline-none transition focus:border-blue-500 disabled:cursor-not-allowed"
                    />

                    <button
                      type="submit"
                      disabled={confirmLoading}
                      className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-emerald-600 px-5 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {confirmLoading ? (
                        <Loader2
                          size={17}
                          className="animate-spin"
                        />
                      ) : (
                        <Check size={17} />
                      )}

                      Verify & Enable
                    </button>
                  </div>
                </form>
              </div>
            )}

            {mfaEnabled && (
              <div>
                <div className="flex items-start gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
                  <CheckCircle2
                    size={20}
                    className="mt-0.5 shrink-0 text-emerald-400"
                  />

                  <div>
                    <p className="font-semibold text-emerald-300">
                      Your account is protected by MFA
                    </p>

                    <p className="mt-1 text-sm leading-6 text-slate-400">
                      INTEL-I will require your authenticator or
                      recovery code after successful password
                      authentication.
                    </p>
                  </div>
                </div>

                <div className="mt-6 grid gap-5 lg:grid-cols-2">
                  <form
                    onSubmit={regenerateRecoveryCodes}
                    className="rounded-xl border border-slate-800 bg-[#030a12] p-4"
                  >
                    <div className="flex items-center gap-2">
                      <RefreshCw
                        size={18}
                        className="text-blue-400"
                      />

                      <h3 className="font-semibold text-white">
                        Recovery Codes
                      </h3>
                    </div>

                    <p className="mt-2 text-sm leading-6 text-slate-400">
                      Generate a fresh set if your existing recovery
                      codes are lost or may have been exposed.
                    </p>

                    <PasswordInput
                      id="recovery-current-password"
                      label="Current Password"
                      value={regeneratePassword}
                      onChange={(event) =>
                        setRegeneratePassword(event.target.value)
                      }
                      show={false}
                      onToggle={() => {}}
                      autoComplete="current-password"
                      disabled={regenerateLoading}
                    />

                    <input
                      type="text"
                      inputMode="numeric"
                      autoComplete="one-time-code"
                      value={regenerateCode}
                      onChange={(event) =>
                        setRegenerateCode(
                          normalizeTotpCode(
                            event.target.value,
                          ),
                        )
                      }
                      placeholder="Current 6-digit MFA code"
                      maxLength={6}
                      required
                      disabled={regenerateLoading}
                      className="mt-4 h-11 w-full rounded-xl border border-slate-700 bg-[#07111d] px-4 text-sm text-white outline-none transition focus:border-blue-500"
                    />

                    <button
                      type="submit"
                      disabled={regenerateLoading}
                      className="mt-3 inline-flex h-10 w-full items-center justify-center gap-2 rounded-xl border border-blue-500/30 bg-blue-500/10 px-4 text-sm font-semibold text-blue-300 transition hover:bg-blue-500/15 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {regenerateLoading ? (
                        <Loader2
                          size={16}
                          className="animate-spin"
                        />
                      ) : (
                        <RefreshCw size={16} />
                      )}

                      Regenerate Recovery Codes
                    </button>
                  </form>

                  <form
                    onSubmit={disableMfa}
                    className="rounded-xl border border-red-500/15 bg-red-500/[0.03] p-4"
                  >
                    <div className="flex items-center gap-2">
                      <ShieldOff
                        size={18}
                        className="text-red-400"
                      />

                      <h3 className="font-semibold text-white">
                        Disable MFA
                      </h3>
                    </div>

                    <p className="mt-2 text-sm leading-6 text-slate-400">
                      Disabling MFA removes the second authentication
                      factor from your account.
                    </p>

                    <PasswordInput
                      id="disable-mfa-current-password"
                      label="Current Password"
                      value={disablePassword}
                      onChange={(event) =>
                        setDisablePassword(event.target.value)
                      }
                      show={false}
                      onToggle={() => {}}
                      autoComplete="current-password"
                      disabled={disableLoading}
                    />

                    <input
                      type="text"
                      inputMode="numeric"
                      autoComplete="one-time-code"
                      value={disableCode}
                      onChange={(event) =>
                        setDisableCode(
                          normalizeTotpCode(
                            event.target.value,
                          ),
                        )
                      }
                      placeholder="Current 6-digit MFA code"
                      maxLength={6}
                      required
                      disabled={disableLoading}
                      className="mt-4 h-11 w-full rounded-xl border border-slate-700 bg-[#07111d] px-4 text-sm text-white outline-none transition focus:border-red-500"
                    />

                    <button
                      type="submit"
                      disabled={disableLoading}
                      className="mt-3 inline-flex h-10 w-full items-center justify-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-4 text-sm font-semibold text-red-300 transition hover:bg-red-500/15 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {disableLoading ? (
                        <Loader2
                          size={16}
                          className="animate-spin"
                        />
                      ) : (
                        <ShieldOff size={16} />
                      )}

                      Disable MFA
                    </button>
                  </form>
                </div>
              </div>
            )}
          </SecurityCard>

          {recoveryCodes.length > 0 && (
            <section className="overflow-hidden rounded-2xl border border-amber-500/25 bg-amber-500/[0.04]">
              <div className="border-b border-amber-500/15 px-5 py-4 sm:px-6">
                <div className="flex items-center gap-3">
                  <KeyRound
                    size={20}
                    className="text-amber-400"
                  />

                  <div>
                    <h2 className="font-semibold text-white">
                      Save Your Recovery Codes
                    </h2>

                    <p className="mt-1 text-sm text-amber-200/70">
                      These codes are shown only after generation.
                    </p>
                  </div>
                </div>
              </div>

              <div className="p-5 sm:p-6">
                <div className="mb-4 flex items-start gap-3 rounded-xl border border-amber-500/20 bg-amber-500/5 p-4">
                  <AlertTriangle
                    size={19}
                    className="mt-0.5 shrink-0 text-amber-400"
                  />

                  <p className="text-sm leading-6 text-slate-300">
                    Store these codes somewhere secure. Each recovery
                    code can be used only once. INTEL-I does not store
                    the plaintext versions.
                  </p>
                </div>

                <div className="grid gap-2 sm:grid-cols-2">
                  {recoveryCodes.map((code, index) => (
                    <div
                      key={`${code}-${index}`}
                      className="flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-[#030a12] px-4 py-3"
                    >
                      <code className="font-semibold tracking-wider text-slate-200">
                        {code}
                      </code>

                      <Check
                        size={15}
                        className="shrink-0 text-emerald-500"
                      />
                    </div>
                  ))}
                </div>

                <button
                  type="button"
                  onClick={copyRecoveryCodes}
                  className="mt-4 inline-flex h-10 items-center justify-center gap-2 rounded-xl border border-slate-700 bg-slate-900 px-4 text-sm font-semibold text-slate-200 transition hover:border-slate-600 hover:bg-slate-800"
                >
                  <Copy size={16} />
                  Copy All Recovery Codes
                </button>
              </div>
            </section>
          )}

          <SecurityCard
            icon={KeyRound}
            title="Change Password"
            description="Change your own INTEL-I password. Existing authenticated sessions will be revoked after a successful password change."
          >
            <form
              onSubmit={changePassword}
              className="grid gap-5"
            >
              <PasswordInput
                id="current-password"
                label="Current Password"
                value={passwordForm.currentPassword}
                onChange={(event) =>
                  handlePasswordChange(
                    "currentPassword",
                    event.target.value,
                  )
                }
                show={passwordVisibility.current}
                onToggle={() =>
                  setPasswordVisibility((previous) => ({
                    ...previous,
                    current: !previous.current,
                  }))
                }
                autoComplete="current-password"
                disabled={passwordLoading}
              />

              <div className="grid gap-5 md:grid-cols-2">
                <PasswordInput
                  id="new-password"
                  label="New Password"
                  value={passwordForm.newPassword}
                  onChange={(event) =>
                    handlePasswordChange(
                      "newPassword",
                      event.target.value,
                    )
                  }
                  show={passwordVisibility.next}
                  onToggle={() =>
                    setPasswordVisibility(
                      (previous) => ({
                        ...previous,
                        next: !previous.next,
                      }),
                    )
                  }
                  autoComplete="new-password"
                  disabled={passwordLoading}
                />

                <PasswordInput
                  id="confirm-password"
                  label="Confirm New Password"
                  value={passwordForm.confirmPassword}
                  onChange={(event) =>
                    handlePasswordChange(
                      "confirmPassword",
                      event.target.value,
                    )
                  }
                  show={passwordVisibility.confirm}
                  onToggle={() =>
                    setPasswordVisibility(
                      (previous) => ({
                        ...previous,
                        confirm: !previous.confirm,
                      }),
                    )
                  }
                  autoComplete="new-password"
                  disabled={passwordLoading}
                />
              </div>

              {passwordMismatch && (
                <div className="flex items-center gap-2 text-sm text-red-400">
                  <X size={16} />
                  New passwords do not match.
                </div>
              )}

              {mfaEnabled && (
                <div>
                  <label
                    htmlFor="password-change-mfa-code"
                    className="mb-2 block text-sm font-medium text-slate-300"
                  >
                    Current 6-digit MFA code
                  </label>

                  <input
                    id="password-change-mfa-code"
                    type="text"
                    inputMode="numeric"
                    autoComplete="one-time-code"
                    value={passwordMfaCode}
                    onChange={(event) =>
                      setPasswordMfaCode(
                        normalizeTotpCode(event.target.value),
                      )
                    }
                    maxLength={6}
                    placeholder="000000"
                    required
                    disabled={passwordLoading}
                    className="h-11 w-full rounded-xl border border-slate-700 bg-[#030a12] px-4 text-center text-lg tracking-[0.35em] text-white outline-none transition focus:border-blue-500 disabled:cursor-not-allowed"
                  />
                </div>
              )}

              <div className="rounded-xl border border-slate-800 bg-[#030a12] p-4">
                <p className="text-sm font-medium text-slate-300">
                  Password requirements
                </p>

                <p className="mt-2 text-xs leading-6 text-slate-500">
                  Use at least 12 characters with uppercase and
                  lowercase letters, a number and a special
                  character. Previously used passwords may be
                  rejected.
                </p>
              </div>

              <div>
                <button
                  type="submit"
                  disabled={
                    passwordLoading || passwordMismatch
                  }
                  className="inline-flex h-11 items-center justify-center gap-2 rounded-xl bg-blue-600 px-6 text-sm font-semibold text-white transition hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {passwordLoading ? (
                    <Loader2
                      size={17}
                      className="animate-spin"
                    />
                  ) : (
                    <Lock size={17} />
                  )}

                  {passwordLoading
                    ? "Changing Password..."
                    : "Change Password"}
                </button>
              </div>
            </form>
          </SecurityCard>
        </div>
      </div>
    </main>
  );
};

export default AccountSecurity;