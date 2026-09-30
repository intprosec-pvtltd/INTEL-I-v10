import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import api from "../api/axios";
import intelIStartupLogo from "../assets/logo.webp";

import {
  Activity,
  Check,
  Cpu,
  Database,
  MapPinned,
  RadioTower,
  Server,
  ShieldCheck,
  Wifi,
  ScanLine,
} from "lucide-react";

/* =========================================================
   CONFIGURATION
========================================================= */

const HEALTH_ENDPOINT =
  "/api/system/startup-health";

const HEALTH_POLL_INTERVAL_MS =
  4000;

const RESULT_REVEAL_INTERVAL_MS =
  180;

const STARTUP_STEPS = [
  {
    id: "security",
    label: "Secure Session",
    shortLabel: "SECURITY",
    description: "Authentication verified",
    icon: ShieldCheck,
  },
  {
    id: "backend",
    label: "Backend Services",
    shortLabel: "BACKEND",
    description: "API services connected",
    icon: Server,
  },
  {
    id: "database",
    label: "Database",
    shortLabel: "DATABASE",
    description: "PostgreSQL / PostGIS ready",
    icon: Database,
  },
  {
    id: "event-services",
    label: "Event Services",
    shortLabel: "EVENT BUS",
    description: "Redis / Kafka channels",
    icon: RadioTower,
  },
  {
    id: "gpu",
    label: "GPU Acceleration",
    shortLabel: "GPU",
    description: "Compute environment ready",
    icon: Cpu,
  },
  {
    id: "ai",
    label: "AI Detection",
    shortLabel: "AI ENGINE",
    description: "Detection & tracking models",
    icon: ScanLine,
  },
  {
    id: "recognition",
    label: "Recognition",
    shortLabel: "RECOGNITION",
    description: "ANPR / OCR / FRS",
    icon: Activity,
  },
  {
    id: "gis",
    label: "GIS Intelligence",
    shortLabel: "GIS",
    description: "Correlation services ready",
    icon: MapPinned,
  },
  {
    id: "streaming",
    label: "Camera Intelligence",
    shortLabel: "VIDEO",
    description: "RTSP / HLS analytics",
    icon: Wifi,
  },
];

const INITIAL_SERVICE_STATES =
  Object.fromEntries(
    STARTUP_STEPS.map((step) => [
      step.id,
      {
        status: "PENDING",
        healthy: false,
        required: false,
        detail: step.description,
      },
    ]),
  );

const normalizeServiceState = (
  value,
  fallbackDescription,
) => {
  const allowedStatuses = new Set([
    "READY",
    "DEGRADED",
    "FAILED",
    "DISABLED",
  ]);
  const status = String(
    value?.status || "FAILED",
  ).toUpperCase();

  return {
    status: allowedStatuses.has(status)
      ? status
      : "FAILED",
    healthy: Boolean(value?.healthy),
    required: Boolean(value?.required),
    detail:
      String(value?.detail || "").trim() ||
      fallbackDescription,
    latency_ms:
      Number.isFinite(value?.latency_ms)
        ? value.latency_ms
        : null,
  };
};

/* =========================================================
   COMPONENT
========================================================= */

const SystemStartupOverlay = ({
  onComplete,
}) => {
  const [serviceStates, setServiceStates] =
    useState(INITIAL_SERVICE_STATES);

  const [healthSummary, setHealthSummary] =
    useState(null);

  const [fadeOut, setFadeOut] =
    useState(false);

  const [logoError, setLogoError] =
    useState(false);

  const onCompleteRef =
    useRef(onComplete);

  useEffect(() => {
    onCompleteRef.current =
      onComplete;
  }, [onComplete]);

  /* =======================================================
     LIVE STARTUP HEALTH CHECK
  ======================================================= */

  useEffect(() => {
    let cancelled = false;
    let pollTimer = null;
    const revealTimers = [];

    const schedule = (callback, delay) => {
      const timer = window.setTimeout(
        callback,
        delay,
      );
      revealTimers.push(timer);
      return timer;
    };

    const markRequestInProgress = () => {
      setServiceStates((previous) => {
        const next = { ...previous };
        const candidate =
          STARTUP_STEPS.find(
            (step) =>
              ["PENDING", "FAILED"].includes(
                previous[step.id]
                  ?.status,
              ),
          ) || STARTUP_STEPS[0];

        next[candidate.id] = {
          ...next[candidate.id],
          status: "CHECKING",
        };
        return next;
      });
    };

    const runHealthCheck = async () => {
      if (cancelled) {
        return;
      }

      markRequestInProgress();

      try {
        const response = await api.get(
          HEALTH_ENDPOINT,
          {
            timeout: 12000,
          },
        );

        if (cancelled) {
          return;
        }

        const payload =
          response?.data || {};
        const services =
          payload.services || {};

        STARTUP_STEPS.forEach(
          (step, index) => {
            schedule(() => {
              if (cancelled) {
                return;
              }

              setServiceStates(
                (previous) => ({
                  ...previous,
                  [step.id]:
                    normalizeServiceState(
                      services[step.id],
                      step.description,
                    ),
                }),
              );
            }, index * RESULT_REVEAL_INTERVAL_MS);
          },
        );

        const revealDuration =
          STARTUP_STEPS.length *
          RESULT_REVEAL_INTERVAL_MS;

        schedule(() => {
          if (cancelled) {
            return;
          }

          setHealthSummary(payload);

          if (payload.ready === true) {
            schedule(
              () => setFadeOut(true),
              450,
            );
            schedule(
              () =>
                onCompleteRef.current?.(),
              780,
            );
            return;
          }

          pollTimer =
            window.setTimeout(
              runHealthCheck,
              HEALTH_POLL_INTERVAL_MS,
            );
        }, revealDuration);
      } catch (error) {
        if (cancelled) {
          return;
        }

        const unauthorized =
          error?.response?.status ===
          401;
        const failedId = unauthorized
          ? "security"
          : "backend";

        setServiceStates(
          (previous) => ({
            ...previous,
            [failedId]: {
              status: "FAILED",
              healthy: false,
              required: true,
              detail: unauthorized
                ? "Authenticated session could not be verified"
                : "Startup health endpoint is unavailable",
            },
          }),
        );
        setHealthSummary({
          ready: false,
          status: "NOT_READY",
          blocking_services: [failedId],
        });

        pollTimer = window.setTimeout(
          runHealthCheck,
          HEALTH_POLL_INTERVAL_MS,
        );
      }
    };

    runHealthCheck();

    return () => {
      cancelled = true;
      if (pollTimer) {
        window.clearTimeout(
          pollTimer,
        );
      }
      revealTimers.forEach((timer) =>
        window.clearTimeout(timer),
      );
    };
  }, []);

  /* =======================================================
     PROGRESS
  ======================================================= */

  const progress = Math.min(
    healthSummary?.ready
      ? 100
      : 96,
    Math.round(
      (Object.values(serviceStates).filter(
        (state) =>
          [
            "READY",
            "DEGRADED",
            "FAILED",
            "DISABLED",
          ].includes(state.status),
      ).length /
        STARTUP_STEPS.length) *
        100,
    ),
  );

  const systemReady =
    healthSummary?.ready === true &&
    STARTUP_STEPS.every(
      (step) =>
        !["PENDING", "CHECKING"].includes(
          serviceStates[step.id]
            ?.status,
        ),
    );

  /* =======================================================
     CURRENT INITIALIZATION STEP
  ======================================================= */

  const currentStepIndex =
    useMemo(() => {
      const checkingIndex =
        STARTUP_STEPS.findIndex(
          (step) =>
            ["CHECKING", "PENDING"].includes(
              serviceStates[step.id]
                ?.status,
            ),
        );

      if (checkingIndex >= 0) {
        return checkingIndex;
      }

      const blockingIndex =
        STARTUP_STEPS.findIndex(
          (step) =>
            step.id ===
            healthSummary
              ?.blocking_services?.[0],
        );

      return blockingIndex >= 0
        ? blockingIndex
        : 0;
    }, [
      healthSummary,
      serviceStates,
    ]);

  const currentStep =
    STARTUP_STEPS[
      currentStepIndex
    ];

  const currentStepState =
    serviceStates[currentStep.id] ||
    INITIAL_SERVICE_STATES[
      currentStep.id
    ];

  const completedCount =
    useMemo(
      () =>
        STARTUP_STEPS.filter(
          (step) =>
            [
              "READY",
              "DEGRADED",
              "DISABLED",
            ].includes(
              serviceStates[step.id]
                ?.status,
            ),
        ).length,
      [serviceStates],
    );

  /* =======================================================
     RENDER
  ======================================================= */

  return (
    <div
      className={`
        fixed inset-0 z-[999999]
        h-[100dvh] w-screen
        overflow-hidden
        bg-[#020914]
        text-white
        transition-opacity
        duration-300
        ${
          fadeOut
            ? "opacity-0"
            : "opacity-100"
        }
      `}
    >
      {/* ===================================================
          BACKGROUND
      =================================================== */}

      <div className="pointer-events-none absolute inset-0">
        {/* Grid */}
        <div
          className="absolute inset-0 opacity-[0.16]"
          style={{
            backgroundImage: `
              linear-gradient(
                rgba(15, 180, 220, 0.12) 1px,
                transparent 1px
              ),
              linear-gradient(
                90deg,
                rgba(15, 180, 220, 0.12) 1px,
                transparent 1px
              )
            `,
            backgroundSize:
              "48px 48px",
          }}
        />

        {/* Center glow */}
        <div
          className="
            absolute
            left-1/2
            top-[45%]
            h-[680px]
            w-[680px]
            -translate-x-1/2
            -translate-y-1/2
            rounded-full
            bg-cyan-500/[0.055]
            blur-[140px]
          "
        />

        {/* Left glow */}
        <div
          className="
            absolute
            -left-40
            top-1/2
            h-[500px]
            w-[500px]
            -translate-y-1/2
            rounded-full
            bg-blue-500/[0.04]
            blur-[120px]
          "
        />

        {/* Right glow */}
        <div
          className="
            absolute
            -right-40
            top-1/2
            h-[500px]
            w-[500px]
            -translate-y-1/2
            rounded-full
            bg-cyan-500/[0.04]
            blur-[120px]
          "
        />

        {/* Horizontal center line */}
        <div
          className="
            absolute
            left-0
            top-1/2
            h-px
            w-full
            bg-gradient-to-r
            from-transparent
            via-cyan-400/30
            to-transparent
          "
        />

        {/* Animated scan */}
        <div className="intel-main-scan absolute left-0 h-px w-full bg-gradient-to-r from-transparent via-cyan-300/80 to-transparent shadow-[0_0_18px_rgba(34,211,238,0.7)]" />
      </div>

      {/* ===================================================
          MAIN SCREEN
      =================================================== */}

      <div
        className="
          relative
          z-10
          flex
          h-full
          w-full
          items-center
          justify-center
          px-4
          py-4
          sm:px-6
          xl:px-8
        "
      >
        <div
          className="
            flex
            h-full
            max-h-[900px]
            w-full
            max-w-[1380px]
            flex-col
          "
        >
          {/* =================================================
              HEADER
          ================================================= */}

          <header className="flex shrink-0 items-center justify-between pb-4">
            {/* Brand */}
            <div
              className="
                relative
                flex
                h-[72px]
                min-w-0
                items-center
                sm:h-[82px]
              "
            >
              <div className="absolute inset-x-[8%] inset-y-[18%] rounded-full bg-blue-500/10 blur-2xl" />

              {!logoError ? (
                <img
                  src={intelIStartupLogo}
                  alt="INTEL-I — AI-Powered Security Intelligence"
                  onError={() =>
                    setLogoError(true)
                  }
                  className="
                    intel-brand-logo
                    relative
                    z-10
                    h-[68px]
                    w-auto
                    max-w-[240px]
                    object-contain
                    object-left
                    drop-shadow-[0_0_14px_rgba(0,132,255,0.20)]
                    sm:h-[78px]
                    sm:max-w-[310px]
                  "
                />
              ) : (
                <div>
                  <div className="flex items-baseline">
                    <h1 className="text-[28px] font-black tracking-[0.27em] text-white sm:text-[34px]">
                      INTEL
                    </h1>
                    <span className="ml-1 text-[28px] font-black tracking-[0.12em] text-cyan-400 sm:text-[34px]">
                      -I
                    </span>
                  </div>
                  <p className="mt-0.5 text-[8px] font-bold tracking-[0.12em] text-cyan-300/70 sm:text-[9px]">
                    AI-POWERED SECURITY INTELLIGENCE
                  </p>
                </div>
              )}
            </div>

            {/* Top status */}
            <div
              className="
                hidden
                items-center
                gap-3
                rounded-full
                border
                border-cyan-900/70
                bg-[#071727]/80
                px-4
                py-2.5
                md:flex
              "
            >
              <span className="relative flex h-2.5 w-2.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-60" />

                <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-cyan-400" />
              </span>

              <span className="text-[10px] font-bold tracking-[0.18em] text-cyan-200">
                SYSTEM BOOT SEQUENCE
              </span>

              <span className="text-xs font-black text-white">
                {progress}%
              </span>
            </div>
          </header>

          {/* =================================================
              MAIN CONTENT
          ================================================= */}

          <div
            className="
              grid
              min-h-0
              flex-1
              grid-cols-1
              gap-4
              lg:grid-cols-[1.35fr_0.65fr]
            "
          >
            {/* ===============================================
                LEFT PANEL
            =============================================== */}

            <section
              className="
                flex
                min-h-0
                flex-col
                rounded-[20px]
                border
                border-[#123b55]
                bg-[#051321]/90
                p-4
                shadow-[0_20px_80px_rgba(0,0,0,0.30)]
                backdrop-blur-xl
                sm:p-5
              "
            >
              {/* Panel header */}
              <div className="mb-4 flex shrink-0 items-start justify-between gap-4">
                <div>
                  <p
                    className="
                      text-[9px]
                      font-black
                      tracking-[0.3em]
                      text-cyan-400
                      sm:text-[10px]
                    "
                  >
                    SYSTEM INITIALIZATION
                  </p>

                  <h2
                    className="
                      mt-1
                      text-lg
                      font-bold
                      text-slate-100
                      sm:text-xl
                    "
                  >
                    Preparing operational
                    environment
                  </h2>

                  <p className="mt-1 text-xs text-slate-500">
                    Validating core INTEL-I
                    services and AI subsystems
                  </p>
                </div>

                <div
                  className="
                    flex
                    h-10
                    w-10
                    shrink-0
                    items-center
                    justify-center
                    rounded-xl
                    border
                    border-cyan-800/50
                    bg-cyan-500/[0.06]
                    text-cyan-300
                  "
                >
                  <ScanLine size={20} />
                </div>
              </div>

              {/* =============================================
                  3 x 3 SYSTEM GRID
              ============================================= */}

              <div
                className="
                  grid
                  min-h-0
                  flex-1
                  grid-cols-1
                  gap-2.5
                  sm:grid-cols-2
                  xl:grid-cols-3
                "
              >
                {STARTUP_STEPS.map(
                  (step) => {
                    const StepIcon =
                      step.icon;

                    const serviceState =
                      serviceStates[
                        step.id
                      ] || {};
                    const status =
                      serviceState.status ||
                      "PENDING";
                    const started =
                      status !== "PENDING";
                    const active =
                      status === "CHECKING";
                    const failed =
                      status === "FAILED";
                    const degraded =
                      status === "DEGRADED";
                    const complete =
                      [
                        "READY",
                        "DEGRADED",
                        "DISABLED",
                      ].includes(status);
                    const statusLabel = {
                      READY: "Ready",
                      DEGRADED: "Degraded",
                      FAILED: "Failed",
                      DISABLED: "Disabled",
                      CHECKING: "Checking",
                      PENDING: "Standby",
                    }[status];

                    return (
                      <div
                        key={step.id}
                        className={`
                          relative
                          flex
                          min-h-[92px]
                          overflow-hidden
                          rounded-xl
                          border
                          p-3
                          transition-all
                          duration-500

                          ${
                            active
                              ? `
                                border-cyan-400/55
                                bg-cyan-400/[0.07]
                                shadow-[0_0_28px_rgba(34,211,238,0.07)]
                              `
                              : failed
                              ? `
                                border-red-500/40
                                bg-red-500/[0.06]
                              `
                              : degraded
                              ? `
                                border-amber-500/35
                                bg-amber-500/[0.05]
                              `
                              : complete
                              ? `
                                border-[#14506a]
                                bg-[#082030]/75
                              `
                              : `
                                border-[#10283b]
                                bg-[#06111e]/50
                                opacity-[0.45]
                              `
                          }
                        `}
                      >
                        {/* Active glow */}
                        {active && (
                          <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-cyan-300 to-transparent" />
                        )}

                        <div className="flex w-full items-start gap-3">
                          {/* Icon */}
                          <div
                            className={`
                              flex
                              h-10
                              w-10
                              shrink-0
                              items-center
                              justify-center
                              rounded-lg
                              border

                              ${
                                active
                                  ? `
                                    border-cyan-400/40
                                    bg-cyan-400/10
                                    text-cyan-300
                                  `
                                  : failed
                                  ? `
                                    border-red-500/30
                                    bg-red-500/10
                                    text-red-400
                                  `
                                  : degraded
                                  ? `
                                    border-amber-500/30
                                    bg-amber-500/10
                                    text-amber-400
                                  `
                                  : complete
                                  ? `
                                    border-emerald-500/20
                                    bg-emerald-400/[0.07]
                                    text-emerald-400
                                  `
                                  : `
                                    border-slate-700/40
                                    bg-slate-800/30
                                    text-slate-600
                                  `
                              }
                            `}
                          >
                            {complete ? (
                              <Check
                                size={18}
                                strokeWidth={
                                  3
                                }
                              />
                            ) : (
                              <StepIcon
                                size={18}
                                className={
                                  active
                                    ? "animate-pulse"
                                    : ""
                                }
                              />
                            )}
                          </div>

                          {/* Text */}
                          <div className="min-w-0 flex-1">
                            <div className="flex items-start justify-between gap-2">
                              <p
                                className={`
                                  truncate
                                  text-[13px]
                                  font-bold

                                  ${
                                    active
                                      ? "text-white"
                                      : failed
                                      ? "text-red-100"
                                      : complete
                                      ? "text-slate-100"
                                      : "text-slate-600"
                                  }
                                `}
                              >
                                {
                                  step.label
                                }
                              </p>

                              <span
                                className={`
                                  shrink-0
                                  text-[8px]
                                  font-black
                                  uppercase
                                  tracking-[0.12em]

                                  ${
                                    active
                                      ? "text-cyan-300"
                                      : failed
                                      ? "text-red-400"
                                      : degraded
                                      ? "text-amber-400"
                                      : complete
                                      ? "text-emerald-400"
                                      : "text-slate-700"
                                  }
                                `}
                              >
                                {statusLabel}
                              </span>
                            </div>

                            <p
                              className={`
                                mt-1
                                line-clamp-2
                                text-[10px]
                                leading-4

                                ${
                                  started
                                    ? "text-slate-500"
                                    : "text-slate-700"
                                }
                              `}
                            >
                              {
                                serviceState.detail ||
                                step.description
                              }
                            </p>

                            {/* Mini progress indicator */}
                            <div className="mt-2 h-[2px] overflow-hidden rounded-full bg-slate-800/80">
                              <div
                                className={`
                                  h-full
                                  rounded-full
                                  transition-all
                                  duration-700

                                  ${
                                    complete
                                      ? degraded
                                        ? "w-full bg-amber-400/70"
                                        : "w-full bg-emerald-400/70"
                                      : failed
                                      ? "w-full bg-red-400/70"
                                      : active
                                      ? "intel-service-progress bg-cyan-400"
                                      : "w-0"
                                  }
                                `}
                              />
                            </div>
                          </div>
                        </div>
                      </div>
                    );
                  },
                )}
              </div>
            </section>

            {/* ===============================================
                RIGHT PANEL
            =============================================== */}

            <section
              className="
                relative
                flex
                min-h-0
                flex-col
                overflow-hidden
                rounded-[20px]
                border
                border-[#123b55]
                bg-[#051321]/90
                p-5
                shadow-[0_20px_80px_rgba(0,0,0,0.30)]
                backdrop-blur-xl
              "
            >
              {/* Corner detail */}
              <div className="absolute right-0 top-0 h-24 w-24 border-r border-t border-cyan-400/10" />

              <div className="relative flex h-full flex-col">
                {/* Header */}
                <div className="shrink-0 text-center">
                  <p className="text-[9px] font-black tracking-[0.3em] text-cyan-400 sm:text-[10px]">
                    SYSTEM HEALTH
                  </p>

                  <p className="mt-1 text-xs text-slate-500">
                    Initialization status
                  </p>
                </div>

                {/* =============================================
                    PROGRESS CIRCLE
                ============================================= */}

                <div className="flex min-h-0 flex-1 items-center justify-center py-3">
                  <div className="relative flex h-[210px] w-[210px] items-center justify-center sm:h-[240px] sm:w-[240px]">
                    {/* Outer rotating marks */}
                    <div className="intel-health-ring absolute inset-0 rounded-full border border-dashed border-cyan-400/20" />

                    {/* Glow */}
                    <div className="absolute inset-[28px] rounded-full bg-cyan-400/[0.04] blur-2xl" />

                    <svg
                      viewBox="0 0 120 120"
                      className="absolute inset-[10px] h-[calc(100%-20px)] w-[calc(100%-20px)] -rotate-90"
                    >
                      {/* Track */}
                      <circle
                        cx="60"
                        cy="60"
                        r="50"
                        fill="none"
                        stroke="rgba(22, 54, 77, 0.85)"
                        strokeWidth="4"
                      />

                      {/* Progress */}
                      <circle
                        cx="60"
                        cy="60"
                        r="50"
                        fill="none"
                        stroke={
                          systemReady
                            ? "rgb(52,211,153)"
                            : "rgb(34,211,238)"
                        }
                        strokeWidth="4"
                        strokeLinecap="round"
                        pathLength="100"
                        strokeDasharray="100"
                        strokeDashoffset={
                          100 -
                          progress
                        }
                        className="transition-all duration-200 ease-linear"
                      />

                      {/* Secondary thin ring */}
                      <circle
                        cx="60"
                        cy="60"
                        r="43"
                        fill="none"
                        stroke="rgba(34,211,238,0.10)"
                        strokeWidth="1"
                        strokeDasharray="2 5"
                      />
                    </svg>

                    {/* Center */}
                    <div className="relative z-10 text-center">
                      {systemReady ? (
                        <>
                          <div
                            className="
                              mx-auto
                              flex
                              h-14
                              w-14
                              items-center
                              justify-center
                              rounded-full
                              border
                              border-emerald-400/30
                              bg-emerald-400/10
                              text-emerald-400
                            "
                          >
                            <Check
                              size={28}
                              strokeWidth={
                                3
                              }
                            />
                          </div>

                          <p className="mt-3 text-xl font-black tracking-[0.12em] text-emerald-400">
                            READY
                          </p>
                        </>
                      ) : (
                        <>
                          <div className="text-[46px] font-black leading-none text-white sm:text-[52px]">
                            {progress}
                            <span className="ml-0.5 text-xl text-cyan-400">
                              %
                            </span>
                          </div>

                          <p className="mt-2 text-[9px] font-black tracking-[0.25em] text-cyan-300/70">
                            INITIALIZED
                          </p>
                        </>
                      )}
                    </div>
                  </div>
                </div>

                {/* =============================================
                    CURRENT STATUS
                ============================================= */}

                <div className="shrink-0">
                  <div
                    className={`
                      rounded-xl
                      border
                      px-4
                      py-3
                      text-center
                      transition-all
                      duration-300

                      ${
                        systemReady
                          ? `
                            border-emerald-500/30
                            bg-emerald-400/[0.05]
                          `
                          : `
                            border-cyan-900/60
                            bg-[#071928]
                          `
                      }
                    `}
                  >
                    {systemReady ? (
                      <>
                        <div className="flex items-center justify-center gap-2">
                          <ShieldCheck
                            size={17}
                            className="text-emerald-400"
                          />

                          <span className="text-sm font-black tracking-[0.12em] text-emerald-400">
                            SYSTEM READY
                          </span>
                        </div>

                        <p className="mt-1 text-[10px] text-slate-500">
                          Entering INTEL-I
                          operational dashboard
                        </p>
                      </>
                    ) : (
                      <>
                        <div className="flex items-center justify-center gap-2">
                          <span className="relative flex h-2 w-2">
                            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-cyan-400 opacity-60" />

                            <span className="relative inline-flex h-2 w-2 rounded-full bg-cyan-400" />
                          </span>

                          <span className="text-sm font-bold text-cyan-300">
                            {
                              currentStep.label
                            }
                          </span>
                        </div>

                        <p className="mt-1 text-[10px] text-slate-500">
                          {
                            currentStepState.detail ||
                            currentStep.description
                          }
                        </p>
                      </>
                    )}
                  </div>

                  {/* Overall progress */}
                  <div className="mt-4">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-[9px] font-bold uppercase tracking-[0.16em] text-slate-600">
                        Initialization sequence
                      </span>

                      <span className="text-[10px] font-black text-cyan-300">
                        {completedCount}
                        /
                        {
                          STARTUP_STEPS.length
                        }
                      </span>
                    </div>

                    <div className="h-1 overflow-hidden rounded-full bg-[#10283b]">
                      <div
                        className={`
                          h-full
                          rounded-full
                          transition-all
                          duration-200
                          ease-linear

                          ${
                            systemReady
                              ? "bg-emerald-400"
                              : "bg-gradient-to-r from-blue-500 via-cyan-400 to-cyan-300"
                          }
                        `}
                        style={{
                          width: `${progress}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </div>

          {/* =================================================
              FOOTER
          ================================================= */}

          <footer
            className="
              mt-3
              flex
              shrink-0
              items-center
              justify-between
              border-t
              border-cyan-900/30
              pt-3
              text-[8px]
              font-semibold
              uppercase
              tracking-[0.18em]
              text-slate-600
              sm:text-[9px]
            "
          >
            <div className="flex items-center gap-2">
              <ShieldCheck
                size={12}
                className="text-cyan-500/70"
              />

              <span>
                Secure Intelligence
                Environment
              </span>
            </div>

            <div className="hidden items-center gap-5 md:flex">
              <span>
                Core Services{" "}
                <span
                  className={
                    systemReady
                      ? "text-emerald-400"
                      : "text-cyan-500"
                  }
                >
                  {systemReady
                    ? "ACTIVE"
                    : "CHECKING"}
                </span>
              </span>

              <span>
                Platform{" "}
                <span className="text-cyan-500">
                  INTEL-I
                </span>
              </span>
            </div>
          </footer>
        </div>
      </div>

      {/* ===================================================
          ANIMATION CSS
      =================================================== */}

      <style>{`
        @keyframes intelMainScan {
          0% {
            top: 4%;
            opacity: 0;
          }

          8% {
            opacity: 0.75;
          }

          92% {
            opacity: 0.55;
          }

          100% {
            top: 96%;
            opacity: 0;
          }
        }

        @keyframes intelLogoRing {
          from {
            transform: rotate(0deg);
          }

          to {
            transform: rotate(360deg);
          }
        }

        @keyframes intelLogoRingReverse {
          from {
            transform: rotate(360deg);
          }

          to {
            transform: rotate(0deg);
          }
        }

        @keyframes intelHealthRing {
          from {
            transform: rotate(0deg);
          }

          to {
            transform: rotate(360deg);
          }
        }

        @keyframes intelBrandLogo {
          0%, 100% {
            filter: brightness(1) saturate(1.04);
            transform: translateY(0);
          }

          50% {
            filter: brightness(1.08) saturate(1.12);
            transform: translateY(-1px);
          }
        }

        @keyframes intelServiceProgress {
          0% {
            width: 8%;
            opacity: 0.5;
          }

          50% {
            width: 68%;
            opacity: 1;
          }

          100% {
            width: 95%;
            opacity: 0.7;
          }
        }

        .intel-main-scan {
          animation:
            intelMainScan
            4.5s
            linear
            infinite;
        }

        .intel-logo-ring {
          animation:
            intelLogoRing
            8s
            linear
            infinite;
        }

        .intel-logo-ring-reverse {
          animation:
            intelLogoRingReverse
            5s
            linear
            infinite;
        }

        .intel-health-ring {
          animation:
            intelHealthRing
            18s
            linear
            infinite;
        }

        .intel-brand-logo {
          animation:
            intelBrandLogo
            3.8s
            ease-in-out
            infinite;
        }

        .intel-service-progress {
          animation:
            intelServiceProgress
            900ms
            ease-in-out
            infinite
            alternate;
        }

        @media (max-height: 780px) {
          .intel-logo-ring,
          .intel-logo-ring-reverse {
            animation-duration: 6s;
          }
        }

        @media (prefers-reduced-motion: reduce) {
          .intel-main-scan,
          .intel-logo-ring,
          .intel-logo-ring-reverse,
          .intel-brand-logo,
          .intel-health-ring,
          .intel-service-progress {
            animation: none !important;
          }
        }
      `}</style>
    </div>
  );
};

export default SystemStartupOverlay;
