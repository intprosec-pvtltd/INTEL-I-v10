import { useEffect, useRef, useState } from "react";
import {
  ArrowLeft,
  CheckCircle2,
  Eye,
  EyeOff,
  KeyRound,
  Lock,
  Mail,
  ShieldCheck,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import toast from "react-hot-toast";
import api, {
  clearAuthStorage,
  saveCsrfToken,
} from "../api/axios";

const AUTH_EXPIRY_KEY = "login_expiry";
const DEFAULT_SESSION_DURATION_SECONDS = 24 * 60 * 60;
const DEFAULT_MFA_CHALLENGE_SECONDS = 5 * 60;

const MFA_MODE_TOTP = "totp";
const MFA_MODE_RECOVERY = "recovery";

const VIEW_LOGIN = "login";
const VIEW_FORGOT_PASSWORD = "forgot-password";
const VIEW_FORGOT_PASSWORD_SENT = "forgot-password-sent";

const clearLocalSession = () => {
  clearAuthStorage();
  localStorage.removeItem(AUTH_EXPIRY_KEY);
};

const normalizeRecoveryCode = (value) =>
  String(value || "")
    .toUpperCase()
    .replace(/[^A-Z0-9-]/g, "")
    .slice(0, 32);

const getErrorMessage = (error, fallback) => {
  const detail = error?.response?.data?.detail;

  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map((item) => item?.msg)
      .filter(Boolean);

    if (messages.length > 0) {
      return messages.join(", ");
    }
  }

  const message = error?.response?.data?.message;

  if (typeof message === "string" && message.trim()) {
    return message;
  }

  if (
    typeof error?.message === "string" &&
    error.message.trim()
  ) {
    return error.message;
  }

  return fallback;
};

const Login = () => {
  const navigate = useNavigate();
  const mfaInputRef = useRef(null);

  const [view, setView] = useState(VIEW_LOGIN);

  const [showPassword, setShowPassword] =
    useState(false);

  const [loading, setLoading] = useState(false);
  const [forgotLoading, setForgotLoading] =
    useState(false);

  const [formData, setFormData] = useState({
    email: "",
    password: "",
  });

  const [forgotEmail, setForgotEmail] =
    useState("");

  const [mfaRequired, setMfaRequired] =
    useState(false);

  const [mfaToken, setMfaToken] = useState("");
  const [mfaMode, setMfaMode] =
    useState(MFA_MODE_TOTP);

  const [mfaCode, setMfaCode] = useState("");
  const [mfaExpiresAt, setMfaExpiresAt] =
    useState(null);

  const [
    mfaSecondsRemaining,
    setMfaSecondsRemaining,
  ] = useState(0);

  useEffect(() => {
    let active = true;

    const storedUser =
      localStorage.getItem("user");

    const loginExpiry = Number(
      localStorage.getItem(AUTH_EXPIRY_KEY),
    );

    if (
      !storedUser ||
      !loginExpiry ||
      Date.now() >= loginExpiry
    ) {
      clearLocalSession();

      return () => {
        active = false;
      };
    }

    api
      .get("/auth/me")
      .then((response) => {
        if (
          !active ||
          !response.data?.id
        ) {
          return;
        }

        localStorage.setItem(
          "user",
          JSON.stringify(response.data),
        );

        navigate("/dashboard", {
          replace: true,
        });
      })
      .catch(() => {
        if (active) {
          clearLocalSession();
        }
      });

    return () => {
      active = false;
    };
  }, [navigate]);

  useEffect(() => {
    if (!mfaRequired) {
      return undefined;
    }

    window.requestAnimationFrame(() => {
      mfaInputRef.current?.focus();
    });

    return undefined;
  }, [mfaRequired, mfaMode]);

  useEffect(() => {
    if (
      !mfaRequired ||
      !mfaExpiresAt
    ) {
      setMfaSecondsRemaining(0);
      return undefined;
    }

    const updateRemainingTime = () => {
      const remaining = Math.max(
        0,
        Math.ceil(
          (mfaExpiresAt - Date.now()) /
            1000,
        ),
      );

      setMfaSecondsRemaining(remaining);

      if (remaining === 0) {
        setMfaToken("");
      }
    };

    updateRemainingTime();

    const interval =
      window.setInterval(
        updateRemainingTime,
        1000,
      );

    return () => {
      window.clearInterval(interval);
    };
  }, [
    mfaRequired,
    mfaExpiresAt,
  ]);

  const handleChange = (event) => {
    const { name, value } =
      event.target;

    setFormData((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const establishAuthenticatedSession = (
    data,
  ) => {
    if (!data?.user?.id) {
      throw new Error(
        "Invalid authentication response",
      );
    }

    clearLocalSession();

    localStorage.setItem(
      "user",
      JSON.stringify(data.user),
    );

    const sessionSeconds = Number(
      data.access_expires_in_seconds ||
        DEFAULT_SESSION_DURATION_SECONDS,
    );

    const boundedSessionSeconds =
      Number.isFinite(sessionSeconds)
        ? Math.max(
            60,
            Math.min(
              sessionSeconds,
              DEFAULT_SESSION_DURATION_SECONDS,
            ),
          )
        : DEFAULT_SESSION_DURATION_SECONDS;

    localStorage.setItem(
      AUTH_EXPIRY_KEY,
      String(
        Date.now() +
          boundedSessionSeconds * 1000,
      ),
    );

    if (data.csrf_token) {
      saveCsrfToken(
        data.csrf_token,
      );
    }

    // Show INTEL-I initialization animation once
    // after successful authentication.
    sessionStorage.setItem(
      "intel_i_show_startup",
      "true",
    );
  };

  const resetMfaChallenge = () => {
    setMfaRequired(false);
    setMfaToken("");
    setMfaCode("");
    setMfaMode(MFA_MODE_TOTP);
    setMfaExpiresAt(null);
    setMfaSecondsRemaining(0);
  };

  const beginMfaChallenge = (data) => {
    const challengeToken =
      data?.mfa_token ||
      data?.challenge_token ||
      data?.token ||
      "";

    if (!challengeToken) {
      throw new Error(
        "MFA challenge token was not returned by the server",
      );
    }

    const challengeSeconds = Number(
      data?.expires_in ||
        data?.expires_in_seconds ||
        DEFAULT_MFA_CHALLENGE_SECONDS,
    );

    const boundedChallengeSeconds =
      Number.isFinite(
        challengeSeconds,
      )
        ? Math.max(
            30,
            Math.min(
              challengeSeconds,
              10 * 60,
            ),
          )
        : DEFAULT_MFA_CHALLENGE_SECONDS;

    clearLocalSession();

    setMfaToken(challengeToken);
    setMfaRequired(true);
    setMfaMode(MFA_MODE_TOTP);
    setMfaCode("");

    setMfaExpiresAt(
      Date.now() +
        boundedChallengeSeconds *
          1000,
    );

    setMfaSecondsRemaining(
      boundedChallengeSeconds,
    );
  };

  const handleSubmit = async (
    event,
  ) => {
    event.preventDefault();

    if (loading) return;

    const email =
      formData.email
        .trim()
        .toLowerCase();

    const password =
      formData.password;

    if (!email || !password) {
      toast.error(
        "Enter email and password",
      );
      return;
    }

    try {
      setLoading(true);

      clearLocalSession();
      resetMfaChallenge();

      const response =
        await api.post(
          "/auth/login",
          {
            email,
            password,
          },
        );

      const data =
        response?.data;

      if (
        data?.mfa_required === true
      ) {
        beginMfaChallenge(data);

        toast.success(
          "Password verified. Enter your MFA code.",
        );

        return;
      }

      if (!data?.user) {
        throw new Error(
          "Invalid login response",
        );
      }

      establishAuthenticatedSession(
        data,
      );

      toast.success(
        "Login successful",
      );

      navigate(
        "/dashboard",
        {
          replace: true,
        },
      );
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Invalid email or password",
        ),
      );
    } finally {
      setLoading(false);
    }
  };

  const handleMfaCodeChange = (
    event,
  ) => {
    const value =
      event.target.value;

    if (
      mfaMode === MFA_MODE_TOTP
    ) {
      setMfaCode(
        value
          .replace(/\D/g, "")
          .slice(0, 6),
      );

      return;
    }

    setMfaCode(
      normalizeRecoveryCode(value),
    );
  };

  const handleMfaSubmit = async (
    event,
  ) => {
    event.preventDefault();

    if (loading) return;

    if (
      !mfaToken ||
      mfaSecondsRemaining <= 0
    ) {
      toast.error(
        "Your MFA challenge has expired. Sign in again.",
      );

      resetMfaChallenge();
      return;
    }

    const code =
      mfaMode === MFA_MODE_TOTP
        ? mfaCode.replace(
            /\D/g,
            "",
          )
        : normalizeRecoveryCode(
            mfaCode,
          );

    if (
      mfaMode === MFA_MODE_TOTP &&
      code.length !== 6
    ) {
      toast.error(
        "Enter the 6-digit authenticator code",
      );
      return;
    }

    if (
      mfaMode ===
        MFA_MODE_RECOVERY &&
      code.length < 8
    ) {
      toast.error(
        "Enter a valid recovery code",
      );
      return;
    }

    try {
      setLoading(true);

      const response =
        await api.post(
          "/auth/mfa/login/verify",
          {
            mfa_token:
              mfaToken,
            code,
          },
        );

      const data =
        response?.data;

      if (!data?.user) {
        throw new Error(
          "Invalid MFA verification response",
        );
      }

      establishAuthenticatedSession(
        data,
      );

      resetMfaChallenge();

      toast.success(
        "MFA verified",
      );

      navigate(
        "/dashboard",
        {
          replace: true,
        },
      );
    } catch (error) {
      const status =
        error?.response?.status;

      const message =
        getErrorMessage(
          error,
          "MFA verification failed",
        );

      if (
        status === 400 ||
        status === 401 ||
        status === 403 ||
        status === 410 ||
        status === 429
      ) {
        resetMfaChallenge();

        if (status === 429) {
          toast.error(
            "Too many verification attempts. Please sign in again later.",
          );
        } else {
          toast.error(
            `${message}. Please sign in again.`,
          );
        }

        return;
      }

      toast.error(message);
    } finally {
      setLoading(false);
    }
  };

  const handleBackToLogin = () => {
    if (
      loading ||
      forgotLoading
    ) {
      return;
    }

    resetMfaChallenge();

    setView(VIEW_LOGIN);

    setFormData(
      (previous) => ({
        ...previous,
        password: "",
      }),
    );
  };

  const switchMfaMode = () => {
    if (loading) return;

    setMfaCode("");

    setMfaMode(
      (previous) =>
        previous ===
        MFA_MODE_TOTP
          ? MFA_MODE_RECOVERY
          : MFA_MODE_TOTP,
    );
  };

  const openForgotPassword = () => {
    if (loading) return;

    resetMfaChallenge();

    setForgotEmail(
      formData.email
        .trim()
        .toLowerCase(),
    );

    setView(
      VIEW_FORGOT_PASSWORD,
    );
  };

  const handleForgotPassword =
    async (event) => {
      event.preventDefault();

      if (forgotLoading) {
        return;
      }

      const email =
        forgotEmail
          .trim()
          .toLowerCase();

      if (!email) {
        toast.error(
          "Enter your email address",
        );
        return;
      }

      try {
        setForgotLoading(true);

        /*
         * Do not use the response to reveal
         * whether the account exists.
         */
        await api.post(
          "/auth/password/forgot",
          {
            email,
          },
        );

        setView(
          VIEW_FORGOT_PASSWORD_SENT,
        );
      } catch (error) {
        /*
         * 429 must still be surfaced because
         * the user needs to know that the
         * rate limit was reached.
         */
        if (
          error?.response?.status ===
          429
        ) {
          toast.error(
            "Too many password reset requests. Please try again later.",
          );

          return;
        }

        setView(
          VIEW_FORGOT_PASSWORD_SENT,
        );
      } finally {
        setForgotLoading(false);
      }
    };

  const formatRemainingTime = (
    seconds,
  ) => {
    const safeSeconds =
      Math.max(
        0,
        Number(seconds) || 0,
      );

    const minutes =
      Math.floor(
        safeSeconds / 60,
      );

    const remainder =
      safeSeconds % 60;

    return `${minutes}:${String(
      remainder,
    ).padStart(2, "0")}`;
  };

  if (
    view ===
    VIEW_FORGOT_PASSWORD_SENT
  ) {
    return (
      <div className="mb-50 flex w-full max-w-105 flex-col items-center justify-center">
        <div className="flex w-full flex-col items-center">
          <div className="flex h-14 w-14 items-center justify-center rounded-full border border-emerald-500/40 bg-emerald-500/10">
            <CheckCircle2
              size={28}
              className="text-emerald-400"
            />
          </div>

          <h1 className="mt-4 text-center text-3xl font-medium text-gray-100">
            Check Your Email
          </h1>

          <p className="mt-3 text-center text-base leading-6 text-gray-300">
            If an active INTEL-I
            account exists for that
            email address, password
            reset instructions have
            been sent.
          </p>

          <div className="mt-5 w-full rounded-2xl border border-slate-700 bg-slate-900/40 px-4 py-4">
            <p className="text-center text-sm leading-6 text-slate-400">
              For security, INTEL-I
              does not reveal whether
              an email address is
              registered.
            </p>
          </div>

          <button
            type="button"
            onClick={
              handleBackToLogin
            }
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

  if (
    view ===
    VIEW_FORGOT_PASSWORD
  ) {
    return (
      <div className="mb-50 flex w-full max-w-105 flex-col items-center justify-center">
        <form
          className="flex w-full flex-col items-center"
          onSubmit={
            handleForgotPassword
          }
        >
          <div className="flex h-14 w-14 items-center justify-center rounded-full border border-blue-500/40 bg-blue-500/10">
            <KeyRound
              size={27}
              className="text-blue-400"
            />
          </div>

          <h1 className="mt-4 text-center text-3xl font-medium text-gray-100">
            Reset Password
          </h1>

          <p className="mt-3 text-center text-base leading-6 text-gray-300">
            Enter your account email
            address. If the account
            exists, INTEL-I will send
            password reset
            instructions.
          </p>

          <label
            className="sr-only"
            htmlFor="forgot-email"
          >
            Email
          </label>

          <div className="mt-6 flex h-12 w-full items-center gap-2 overflow-hidden rounded-full border border-[#1f4b7a] bg-transparent pr-4 pl-5 focus-within:border-blue-500">
            <Mail
              size={18}
              className="shrink-0 text-gray-400"
            />

            <input
              id="forgot-email"
              type="email"
              value={forgotEmail}
              onChange={(event) =>
                setForgotEmail(
                  event.target.value,
                )
              }
              placeholder="Email"
              autoComplete="email"
              maxLength={254}
              required
              disabled={
                forgotLoading
              }
              autoFocus
              className="h-full w-full bg-transparent text-white outline-none placeholder:text-gray-400 disabled:cursor-not-allowed"
            />
          </div>

          <button
            type="submit"
            disabled={
              forgotLoading
            }
            className="mt-6 h-11 w-full rounded-full bg-blue-600 text-lg font-medium text-white transition hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
          >
            {forgotLoading
              ? "Sending..."
              : "Send Reset Link"}
          </button>

          <button
            type="button"
            disabled={
              forgotLoading
            }
            onClick={
              handleBackToLogin
            }
            className="mt-5 inline-flex items-center gap-2 text-sm text-gray-400 transition hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <ArrowLeft
              size={16}
            />
            Back to Login
          </button>
        </form>
      </div>
    );
  }

  if (mfaRequired) {
    const challengeExpired =
      !mfaToken ||
      mfaSecondsRemaining <= 0;

    return (
      <div className="mb-50 flex w-full max-w-105 flex-col items-center justify-center">
        <form
          className="flex w-full flex-col items-center"
          onSubmit={
            handleMfaSubmit
          }
        >
          <div className="flex h-14 w-14 items-center justify-center rounded-full border border-blue-500/40 bg-blue-500/10">
            {mfaMode ===
            MFA_MODE_TOTP ? (
              <ShieldCheck
                size={28}
                className="text-blue-400"
              />
            ) : (
              <KeyRound
                size={28}
                className="text-blue-400"
              />
            )}
          </div>

          <h1 className="mt-4 text-center text-3xl font-medium text-gray-100">
            Two-Factor Authentication
          </h1>

          <p className="mt-3 text-center text-base leading-6 text-gray-300">
            {mfaMode ===
            MFA_MODE_TOTP
              ? "Enter the 6-digit code from your authenticator app."
              : "Enter one of the recovery codes you saved when MFA was enabled."}
          </p>

          {!challengeExpired && (
            <p className="mt-2 text-center text-sm text-gray-400">
              Verification expires
              in{" "}
              <span className="font-medium text-gray-200">
                {formatRemainingTime(
                  mfaSecondsRemaining,
                )}
              </span>
            </p>
          )}

          {challengeExpired ? (
            <div className="mt-6 w-full rounded-2xl border border-amber-500/30 bg-amber-500/10 px-4 py-4">
              <p className="text-center text-sm text-amber-200">
                Your MFA login
                challenge has expired.
                Sign in with your
                password again to
                generate a new
                challenge.
              </p>
            </div>
          ) : (
            <>
              <label
                className="sr-only"
                htmlFor="login-mfa-code"
              >
                {mfaMode ===
                MFA_MODE_TOTP
                  ? "Authenticator code"
                  : "Recovery code"}
              </label>

              <div className="mt-6 flex h-12 w-full items-center gap-2 overflow-hidden rounded-full border border-[#1f4b7a] bg-transparent pr-5 pl-5 focus-within:border-blue-500">
                {mfaMode ===
                MFA_MODE_TOTP ? (
                  <ShieldCheck
                    size={18}
                    className="shrink-0 text-gray-400"
                  />
                ) : (
                  <KeyRound
                    size={18}
                    className="shrink-0 text-gray-400"
                  />
                )}

                <input
                  ref={mfaInputRef}
                  id="login-mfa-code"
                  name="mfa-code"
                  type="text"
                  value={mfaCode}
                  onChange={
                    handleMfaCodeChange
                  }
                  placeholder={
                    mfaMode ===
                    MFA_MODE_TOTP
                      ? "000000"
                      : "Recovery code"
                  }
                  autoComplete={
                    mfaMode ===
                    MFA_MODE_TOTP
                      ? "one-time-code"
                      : "off"
                  }
                  inputMode={
                    mfaMode ===
                    MFA_MODE_TOTP
                      ? "numeric"
                      : "text"
                  }
                  maxLength={
                    mfaMode ===
                    MFA_MODE_TOTP
                      ? 6
                      : 32
                  }
                  required
                  disabled={loading}
                  className={
                    mfaMode ===
                    MFA_MODE_TOTP
                      ? "h-full w-full bg-transparent text-center text-xl tracking-[0.35em] text-white outline-none placeholder:text-gray-500 disabled:cursor-not-allowed"
                      : "h-full w-full bg-transparent text-center text-base tracking-wider text-white uppercase outline-none placeholder:normal-case placeholder:tracking-normal placeholder:text-gray-400 disabled:cursor-not-allowed"
                  }
                />
              </div>

              <button
                disabled={loading}
                type="submit"
                className="mt-6 h-11 w-full rounded-full bg-blue-600 text-lg font-medium text-white hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {loading
                  ? "Verifying..."
                  : "Verify & Continue"}
              </button>

              <button
                type="button"
                disabled={loading}
                onClick={
                  switchMfaMode
                }
                className="mt-4 text-sm font-medium text-blue-400 hover:text-blue-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {mfaMode ===
                MFA_MODE_TOTP
                  ? "Use a recovery code instead"
                  : "Use authenticator code instead"}
              </button>
            </>
          )}

          <button
            type="button"
            disabled={loading}
            onClick={
              handleBackToLogin
            }
            className="mt-5 inline-flex items-center gap-2 text-sm text-gray-400 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            <ArrowLeft
              size={16}
            />
            Back to login
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="mb-50 flex w-full max-w-105 flex-col items-center justify-center">
      <form
        className="flex w-full flex-col items-center"
        onSubmit={handleSubmit}
      >
        <h1 className="text-4xl font-medium text-gray-100">
          Login
        </h1>

        <p className="mt-3 text-center text-xl text-gray-300">
          Welcome back! Please sign
          in to continue
        </p>

        <label
          className="sr-only"
          htmlFor="login-email"
        >
          Email
        </label>

        <div className="mt-6 flex h-12 w-full items-center gap-2 overflow-hidden rounded-full border border-[#1f4b7a] bg-transparent pr-4 pl-5 focus-within:border-blue-500">
          <Mail
            size={18}
            className="shrink-0 text-gray-400"
          />

          <input
            id="login-email"
            name="email"
            type="email"
            value={
              formData.email
            }
            onChange={
              handleChange
            }
            placeholder="Email"
            autoComplete="email"
            maxLength={254}
            required
            disabled={loading}
            className="h-full w-full bg-transparent text-white outline-none placeholder:text-gray-400 disabled:cursor-not-allowed"
          />
        </div>

        <label
          className="sr-only"
          htmlFor="login-password"
        >
          Password
        </label>

        <div className="mt-6 flex h-12 w-full items-center gap-2 overflow-hidden rounded-full border border-[#1f4b7a] bg-transparent pr-4 pl-5 focus-within:border-blue-500">
          <Lock
            size={18}
            className="shrink-0 text-gray-400"
          />

          <input
            id="login-password"
            name="password"
            type={
              showPassword
                ? "text"
                : "password"
            }
            value={
              formData.password
            }
            onChange={
              handleChange
            }
            placeholder="Password"
            autoComplete="current-password"
            maxLength={128}
            required
            disabled={loading}
            className="h-full w-full bg-transparent text-white outline-none placeholder:text-gray-400 disabled:cursor-not-allowed"
          />

          <button
            type="button"
            onClick={() =>
              setShowPassword(
                (previous) =>
                  !previous,
              )
            }
            disabled={loading}
            className="shrink-0 text-gray-400 hover:text-white focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
            aria-label={
              showPassword
                ? "Hide password"
                : "Show password"
            }
            aria-pressed={
              showPassword
            }
          >
            {showPassword ? (
              <EyeOff
                size={18}
              />
            ) : (
              <Eye size={18} />
            )}
          </button>
        </div>

        <div className="mt-3 flex w-full justify-end">
          <button
            type="button"
            disabled={loading}
            onClick={
              openForgotPassword
            }
            className="text-sm font-medium text-blue-400 transition hover:text-blue-300 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 disabled:cursor-not-allowed disabled:opacity-60"
          >
            Forgot password?
          </button>
        </div>

        <button
          disabled={loading}
          type="submit"
          className="mt-5 h-11 w-full rounded-full bg-blue-600 text-xl font-medium text-white hover:bg-blue-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-400 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loading
            ? "Signing in..."
            : "Login"}
        </button>

        <p className="mt-4 text-center text-sm text-gray-400">
          Accounts are created by
          the Super Administrator.
        </p>
      </form>
    </div>
  );
};

export default Login;