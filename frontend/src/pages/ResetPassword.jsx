import { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  Lock,
  ShieldCheck,
} from "lucide-react";
import {
  useNavigate,
  useSearchParams,
} from "react-router-dom";
import toast from "react-hot-toast";
import api, {
  clearAuthStorage,
} from "../api/axios";

const AUTH_EXPIRY_KEY = "login_expiry";

const STATUS_FORM = "form";
const STATUS_SUCCESS = "success";
const STATUS_INVALID = "invalid";

const MAX_PASSWORD_LENGTH = 128;
const MAX_PASSWORD_UTF8_BYTES = 72;

const clearLocalSession = () => {
  clearAuthStorage();
  localStorage.removeItem(AUTH_EXPIRY_KEY);
};

const getUtf8Length = (value) => {
  try {
    return new TextEncoder().encode(
      String(value || ""),
    ).length;
  } catch {
    return String(value || "").length;
  }
};

const getErrorMessage = (
  error,
  fallback,
) => {
  const detail =
    error?.response?.data?.detail;

  if (
    typeof detail === "string" &&
    detail.trim()
  ) {
    return detail;
  }

  if (
    Array.isArray(detail) &&
    detail.length > 0
  ) {
    const messages = detail
      .map((item) => item?.msg)
      .filter(Boolean);

    if (messages.length > 0) {
      return messages.join(", ");
    }
  }

  const message =
    error?.response?.data?.message;

  if (
    typeof message === "string" &&
    message.trim()
  ) {
    return message;
  }

  return fallback;
};

const getResetFailureMessage = (
  error,
) => {
  const status =
    error?.response?.status;

  if (status === 429) {
    return "Too many password reset attempts. Please wait and try again.";
  }

  if (
    status === 400 ||
    status === 401 ||
    status === 403 ||
    status === 404 ||
    status === 410
  ) {
    return (
      getErrorMessage(
        error,
        "This password reset link is invalid, expired, or has already been used.",
      ) ||
      "This password reset link is invalid, expired, or has already been used."
    );
  }

  return getErrorMessage(
    error,
    "Unable to reset your password. Please try again.",
  );
};

const PasswordInput = ({
  id,
  label,
  value,
  onChange,
  show,
  onToggle,
  autoFocus = false,
  disabled = false,
}) => (
  <div className="w-full">
    <label
      htmlFor={id}
      className="mb-2 block text-sm font-medium text-gray-300"
    >
      {label}
    </label>

    <div className="flex h-12 w-full items-center gap-2 overflow-hidden rounded-full border border-[#1f4b7a] bg-transparent pr-4 pl-5 transition focus-within:border-blue-500">
      <Lock
        size={18}
        className="shrink-0 text-gray-400"
      />

      <input
        id={id}
        type={show ? "text" : "password"}
        value={value}
        onChange={onChange}
        placeholder={label}
        autoComplete="new-password"
        maxLength={MAX_PASSWORD_LENGTH}
        autoFocus={autoFocus}
        required
        disabled={disabled}
        className="h-full min-w-0 flex-1 bg-transparent text-white outline-none placeholder:text-gray-500 disabled:cursor-not-allowed"
      />

      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        aria-label={
          show
            ? `Hide ${label}`
            : `Show ${label}`
        }
        aria-pressed={show}
        className="shrink-0 text-gray-400 transition hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {show ? (
          <EyeOff size={18} />
        ) : (
          <Eye size={18} />
        )}
      </button>
    </div>
  </div>
);

const Requirement = ({
  valid,
  children,
}) => (
  <div
    className={`flex items-center gap-2 text-xs ${
      valid
        ? "text-emerald-400"
        : "text-slate-500"
    }`}
  >
    <span
      className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${
        valid
          ? "border-emerald-500/40 bg-emerald-500/10"
          : "border-slate-700"
      }`}
    >
      {valid && (
        <CheckCircle2 size={11} />
      )}
    </span>

    <span>{children}</span>
  </div>
);

const ResetPassword = () => {
  const navigate = useNavigate();

  const [searchParams] =
    useSearchParams();

  /*
   * The backend-generated frontend URL is expected
   * to use:
   *
   * /reset-password?token=<opaque-reset-token>
   *
   * The token stays in memory and the current URL.
   * It is never copied into localStorage/sessionStorage.
   */
  const resetToken = useMemo(
    () =>
      String(
        searchParams.get("token") || "",
      ).trim(),
    [searchParams],
  );

  const [status, setStatus] =
    useState(
      resetToken
        ? STATUS_FORM
        : STATUS_INVALID,
    );

  const [
    newPassword,
    setNewPassword,
  ] = useState("");

  const [
    confirmPassword,
    setConfirmPassword,
  ] = useState("");

  const [
    showNewPassword,
    setShowNewPassword,
  ] = useState(false);

  const [
    showConfirmPassword,
    setShowConfirmPassword,
  ] = useState(false);

  const [loading, setLoading] =
    useState(false);

  const [
    failureMessage,
    setFailureMessage,
  ] = useState("");

  useEffect(() => {
    /*
     * A password-reset page should not retain
     * any previously cached authenticated
     * frontend identity.
     *
     * The backend remains the source of truth
     * for authentication cookies/session state.
     */
    clearLocalSession();
  }, []);

  useEffect(() => {
    if (!resetToken) {
      setStatus(STATUS_INVALID);

      setFailureMessage(
        "This password reset link is missing its security token. Request a new password reset link.",
      );
    }
  }, [resetToken]);

  const passwordChecks =
    useMemo(() => {
      const value = newPassword;

      return {
        length:
          value.length >= 12,
        uppercase:
          /[A-Z]/.test(value),
        lowercase:
          /[a-z]/.test(value),
        number:
          /\d/.test(value),
        special:
          /[^A-Za-z0-9]/.test(
            value,
          ),
        utf8:
          getUtf8Length(value) <=
          MAX_PASSWORD_UTF8_BYTES,
      };
    }, [newPassword]);

  const passwordLooksValid =
    Object.values(
      passwordChecks,
    ).every(Boolean);

  const passwordsMatch =
    Boolean(confirmPassword) &&
    newPassword ===
      confirmPassword;

  const handleSubmit = async (
    event,
  ) => {
    event.preventDefault();

    if (loading) return;

    if (!resetToken) {
      setStatus(STATUS_INVALID);

      setFailureMessage(
        "This password reset link is invalid. Request a new password reset link.",
      );

      return;
    }

    if (!newPassword) {
      toast.error(
        "Enter a new password",
      );
      return;
    }

    if (
      getUtf8Length(newPassword) >
      MAX_PASSWORD_UTF8_BYTES
    ) {
      toast.error(
        "Password is too long. Use no more than 72 UTF-8 bytes.",
      );
      return;
    }

    if (!passwordLooksValid) {
      toast.error(
        "Your new password does not meet the security requirements.",
      );
      return;
    }

    if (!confirmPassword) {
      toast.error(
        "Confirm your new password",
      );
      return;
    }

    if (
      newPassword !==
      confirmPassword
    ) {
      toast.error(
        "Passwords do not match",
      );
      return;
    }

    try {
      setLoading(true);
      setFailureMessage("");

      /*
       * Backend contract:
       *
       * POST /auth/password/reset
       * {
       *   "token": "...",
       *   "new_password": "...",
       *   "confirm_password": "..."
       * }
       */
      await api.post(
        "/auth/password/reset",
        {
          token: resetToken,
          new_password:
            newPassword,
          confirm_password:
            confirmPassword,
        },
      );

      /*
       * Remove password values from React
       * state immediately after success.
       */
      setNewPassword("");
      setConfirmPassword("");

      clearLocalSession();

      /*
       * Remove the reset token from the
       * browser address bar after successful
       * consumption.
       *
       * This does not navigate away from the
       * success screen.
       */
      window.history.replaceState(
        {},
        document.title,
        "/reset-password",
      );

      setStatus(STATUS_SUCCESS);

      toast.success(
        "Password reset successfully",
      );
    } catch (error) {
      const httpStatus =
        error?.response?.status;

      const message =
        getResetFailureMessage(
          error,
        );

      /*
       * Invalid, expired, consumed or malformed
       * reset tokens should not remain usable
       * from this UI.
       */
      if (
        httpStatus === 400 ||
        httpStatus === 401 ||
        httpStatus === 403 ||
        httpStatus === 404 ||
        httpStatus === 410
      ) {
        setNewPassword("");
        setConfirmPassword("");

        setFailureMessage(
          message,
        );

        setStatus(
          STATUS_INVALID,
        );

        return;
      }

      if (httpStatus === 429) {
        toast.error(message);
        return;
      }

      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const goToLogin = () => {
    clearLocalSession();

    navigate("/", {
      replace: true,
    });
  };

  if (
    status === STATUS_SUCCESS
  ) {
    return (
      <div className="flex min-h-[calc(100vh-85px)] w-full items-center justify-center px-4 py-10">
        <div className="flex w-full max-w-105 flex-col items-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-full border border-emerald-500/40 bg-emerald-500/10">
            <CheckCircle2
              size={31}
              className="text-emerald-400"
            />
          </div>

          <h1 className="mt-5 text-center text-3xl font-medium text-gray-100">
            Password Updated
          </h1>

          <p className="mt-3 text-center text-base leading-6 text-gray-300">
            Your INTEL-I password
            has been reset
            successfully.
          </p>

          <div className="mt-5 w-full rounded-2xl border border-emerald-500/20 bg-emerald-500/5 p-4">
            <div className="flex items-start gap-3">
              <ShieldCheck
                size={20}
                className="mt-0.5 shrink-0 text-emerald-400"
              />

              <p className="text-sm leading-6 text-slate-300">
                Existing
                authenticated sessions
                have been revoked. Sign
                in again using your new
                password and complete
                MFA if it is enabled on
                your account.
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={goToLogin}
            className="mt-6 inline-flex h-11 w-full items-center justify-center gap-2 rounded-full bg-blue-600 text-base font-medium text-white transition hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
          >
            <ArrowLeft
              size={17}
            />
            Continue to Login
          </button>
        </div>
      </div>
    );
  }

  if (
    status === STATUS_INVALID
  ) {
    return (
      <div className="flex min-h-[calc(100vh-85px)] w-full items-center justify-center px-4 py-10">
        <div className="flex w-full max-w-105 flex-col items-center">
          <div className="flex h-16 w-16 items-center justify-center rounded-full border border-amber-500/40 bg-amber-500/10">
            <AlertTriangle
              size={30}
              className="text-amber-400"
            />
          </div>

          <h1 className="mt-5 text-center text-3xl font-medium text-gray-100">
            Reset Link Unavailable
          </h1>

          <p className="mt-3 text-center text-base leading-6 text-gray-300">
            {failureMessage ||
              "This password reset link is invalid, expired, or has already been used."}
          </p>

          <div className="mt-5 w-full rounded-2xl border border-slate-700 bg-slate-900/40 p-4">
            <p className="text-center text-sm leading-6 text-slate-400">
              Return to the login
              screen and select
              <span className="font-medium text-slate-200">
                {" "}
                Forgot password?{" "}
              </span>
              to request a new reset
              link.
            </p>
          </div>

          <button
            type="button"
            onClick={goToLogin}
            className="mt-6 inline-flex h-11 w-full items-center justify-center gap-2 rounded-full bg-blue-600 text-base font-medium text-white transition hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400"
          >
            <ArrowLeft
              size={17}
            />
            Back to Login
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex min-h-[calc(100vh-85px)] w-full items-center justify-center px-4 py-10">
      <form
        onSubmit={handleSubmit}
        className="flex w-full max-w-105 flex-col items-center"
      >
        <div className="flex h-16 w-16 items-center justify-center rounded-full border border-blue-500/40 bg-blue-500/10">
          <KeyRound
            size={29}
            className="text-blue-400"
          />
        </div>

        <h1 className="mt-5 text-center text-3xl font-medium text-gray-100">
          Create New Password
        </h1>

        <p className="mt-3 text-center text-base leading-6 text-gray-300">
          Choose a strong new
          password for your INTEL-I
          account.
        </p>

        <div className="mt-6 w-full">
          <PasswordInput
            id="reset-new-password"
            label="New Password"
            value={newPassword}
            onChange={(event) =>
              setNewPassword(
                event.target.value,
              )
            }
            show={
              showNewPassword
            }
            onToggle={() =>
              setShowNewPassword(
                (previous) =>
                  !previous,
              )
            }
            autoFocus
            disabled={loading}
          />
        </div>

        <div className="mt-5 w-full">
          <PasswordInput
            id="reset-confirm-password"
            label="Confirm New Password"
            value={confirmPassword}
            onChange={(event) =>
              setConfirmPassword(
                event.target.value,
              )
            }
            show={
              showConfirmPassword
            }
            onToggle={() =>
              setShowConfirmPassword(
                (previous) =>
                  !previous,
              )
            }
            disabled={loading}
          />
        </div>

        {confirmPassword &&
          !passwordsMatch && (
            <div className="mt-3 flex w-full items-center gap-2 text-sm text-red-400">
              <AlertTriangle
                size={15}
              />
              Passwords do not match.
            </div>
          )}

        <div className="mt-5 w-full rounded-2xl border border-slate-800 bg-[#07111d] p-4">
          <div className="mb-3 flex items-center gap-2">
            <ShieldCheck
              size={17}
              className="text-blue-400"
            />

            <p className="text-sm font-semibold text-slate-200">
              Password requirements
            </p>
          </div>

          <div className="grid gap-2 sm:grid-cols-2">
            <Requirement
              valid={
                passwordChecks.length
              }
            >
              At least 12 characters
            </Requirement>

            <Requirement
              valid={
                passwordChecks.uppercase
              }
            >
              One uppercase letter
            </Requirement>

            <Requirement
              valid={
                passwordChecks.lowercase
              }
            >
              One lowercase letter
            </Requirement>

            <Requirement
              valid={
                passwordChecks.number
              }
            >
              One number
            </Requirement>

            <Requirement
              valid={
                passwordChecks.special
              }
            >
              One special character
            </Requirement>

            <Requirement
              valid={
                passwordChecks.utf8
              }
            >
              Maximum 72 UTF-8 bytes
            </Requirement>
          </div>
        </div>

        <button
          type="submit"
          disabled={
            loading ||
            !passwordLooksValid ||
            !passwordsMatch
          }
          className="mt-6 inline-flex h-11 w-full items-center justify-center gap-2 rounded-full bg-blue-600 text-lg font-medium text-white transition hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          {loading ? (
            <>
              <Loader2
                size={18}
                className="animate-spin"
              />
              Resetting...
            </>
          ) : (
            <>
              <Lock size={18} />
              Reset Password
            </>
          )}
        </button>

        <button
          type="button"
          disabled={loading}
          onClick={goToLogin}
          className="mt-5 inline-flex items-center gap-2 text-sm text-gray-400 transition hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <ArrowLeft
            size={16}
          />
          Back to Login
        </button>
      </form>
    </div>
  );
};

export default ResetPassword;