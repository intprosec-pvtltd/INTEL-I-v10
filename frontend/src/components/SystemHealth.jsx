import React, { useMemo, useEffect } from "react";
import { formatIndianTime } from "../utils/dateTime";

import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  CircleOff,
  Cpu,
  Database,
  Gauge,
  HardDrive,
  MemoryStick,
  RadioTower,
  RefreshCw,
  Server,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";

const STATUS_META = {
  ONLINE: {
    label: "ONLINE",
    className: "text-emerald-300",
    dot: "bg-emerald-400",
    border: "border-emerald-400/20",
    bg: "bg-emerald-400/5",
  },
  HEALTHY: {
    label: "HEALTHY",
    className: "text-emerald-300",
    dot: "bg-emerald-400",
    border: "border-emerald-400/20",
    bg: "bg-emerald-400/5",
  },
  READY: {
    label: "READY",
    className: "text-emerald-300",
    dot: "bg-emerald-400",
    border: "border-emerald-400/20",
    bg: "bg-emerald-400/5",
  },
  DEGRADED: {
    label: "DEGRADED",
    className: "text-amber-300",
    dot: "bg-amber-400",
    border: "border-amber-400/20",
    bg: "bg-amber-400/5",
  },
  RECONNECTING: {
    label: "RECONNECTING",
    className: "text-amber-300",
    dot: "bg-amber-400",
    border: "border-amber-400/20",
    bg: "bg-amber-400/5",
  },
  AUTHENTICATION_FAILED: {
    label: "AUTH FAILED",
    className: "text-red-300",
    dot: "bg-red-400",
    border: "border-red-400/20",
    bg: "bg-red-400/5",
  },
  TIMEOUT: {
    label: "TIMEOUT",
    className: "text-red-300",
    dot: "bg-red-400",
    border: "border-red-400/20",
    bg: "bg-red-400/5",
  },
  AI_DISABLED: {
    label: "AI DISABLED",
    className: "text-amber-300",
    dot: "bg-amber-400",
    border: "border-amber-400/20",
    bg: "bg-amber-400/5",
  },
  INITIALIZING: {
    label: "CHECKING",
    className: "text-blue-300",
    dot: "bg-blue-400",
    border: "border-blue-400/20",
    bg: "bg-blue-400/5",
  },
  CHECKING: {
    label: "CHECKING",
    className: "text-blue-300",
    dot: "bg-blue-400",
    border: "border-blue-400/20",
    bg: "bg-blue-400/5",
  },
  OFFLINE: {
    label: "OFFLINE",
    className: "text-red-300",
    dot: "bg-red-400",
    border: "border-red-400/20",
    bg: "bg-red-400/5",
  },
  UNAVAILABLE: {
    label: "UNAVAILABLE",
    className: "text-slate-400",
    dot: "bg-slate-500",
    border: "border-slate-600/40",
    bg: "bg-slate-600/5",
  },
  UNKNOWN: {
    label: "UNKNOWN",
    className: "text-slate-400",
    dot: "bg-slate-500",
    border: "border-slate-600/40",
    bg: "bg-slate-600/5",
  },
};

const normalizeStatus = (value, fallback = "UNKNOWN") =>
  String(value || fallback).trim().toUpperCase();

const StatusBadge = ({ value, initializing = false }) => {
  const status = initializing
    ? "CHECKING"
    : normalizeStatus(value);
  const meta = STATUS_META[status] || STATUS_META.UNKNOWN;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold tracking-wider ${meta.border} ${meta.bg} ${meta.className}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${meta.dot} ${
          initializing ? "animate-pulse" : ""
        }`}
      />
      {meta.label}
    </span>
  );
};

const HealthMetric = ({
  icon: Icon,
  label,
  value,
  unit = "",
  unavailable = false,
}) => {
  const displayValue = unavailable
    ? "Telemetry unavailable"
    : value === null || value === undefined || value === ""
      ? "—"
      : value;

  return (
    <div className="rounded-xl border border-slate-700/60 bg-[#091728] px-3.5 py-3">
      <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
        {Icon ? <Icon size={13} /> : null}
        {label}
      </div>

      <div
        className={`mt-1.5 font-bold ${
          unavailable
            ? "text-xs text-slate-400"
            : "text-lg text-white"
        }`}
      >
        {displayValue}
        {!unavailable &&
        value !== null &&
        value !== undefined &&
        value !== "" &&
        unit ? (
          <span className="ml-1 text-xs font-medium text-slate-500">
            {unit}
          </span>
        ) : null}
      </div>
    </div>
  );
};

const ServiceCard = ({
  icon: Icon,
  title,
  status,
  latency,
  initializing,
}) => (
  <div className="rounded-xl border border-slate-700/60 bg-[#0b1727] p-4">
    <div className="flex items-start justify-between gap-3">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-blue-400/20 bg-blue-400/10">
          {Icon ? <Icon size={17} className="text-blue-400" /> : null}
        </div>

        <div className="min-w-0">
          <p className="truncate text-sm font-bold text-white">
            {title}
          </p>

          {latency !== null &&
            latency !== undefined &&
            !initializing && (
              <p className="mt-0.5 text-[10px] text-slate-500">
                Latency: {Number(latency).toFixed(1)} ms
              </p>
            )}
        </div>
      </div>

      <StatusBadge value={status} initializing={initializing} />
    </div>
  </div>
);

const ProgressBar = ({ value }) => {
  const numeric = Number(value);
  const hasValue = Number.isFinite(numeric);
  const bounded = hasValue
    ? Math.max(0, Math.min(100, numeric))
    : 0;

  return (
    <div className="h-2 overflow-hidden rounded-full bg-slate-800">
      {hasValue ? (
        <div
          className="h-full rounded-full bg-blue-400 transition-all duration-500"
          style={{ width: `${bounded}%` }}
        />
      ) : (
        <div className="h-full w-1/3 animate-pulse rounded-full bg-slate-700" />
      )}
    </div>
  );
};

const SkeletonServiceCard = ({ icon: Icon, title }) => (
  <div className="animate-pulse rounded-xl border border-slate-700/60 bg-[#0b1727] p-4">
    <div className="flex items-start justify-between gap-3">
      <div className="flex min-w-0 items-center gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-blue-400/10 bg-blue-400/5">
          {Icon ? <Icon size={17} className="text-blue-400/50" /> : null}
        </div>
        <div>
          <p className="text-sm font-bold text-slate-300">{title}</p>
          <div className="mt-2 h-2.5 w-16 rounded bg-slate-800" />
        </div>
      </div>
      <div className="h-5 w-16 rounded-full bg-slate-800" />
    </div>
  </div>
);

const SystemHealth = ({
  isOpen,
  isInitializing,
  health,
  healthError,
  healthLoading,
  refreshing,
  countdown,
  startupAutoCloseActive,
  onClose,
  onRefresh,
}) => {
  useEffect(() => {
    if (!isOpen) return undefined;

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        onClose?.();
      }
    };

    window.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [isOpen, onClose]);

  const cameraCounts = useMemo(() => {
    const cameras = health?.cameras || {};

    return {
      total:
        cameras.total === null || cameras.total === undefined
          ? null
          : Number(cameras.total),
      online:
        cameras.online === null || cameras.online === undefined
          ? null
          : Number(cameras.online),
      degraded:
        cameras.degraded === null || cameras.degraded === undefined
          ? null
          : Number(cameras.degraded),
      reconnecting:
        cameras.reconnecting === null ||
        cameras.reconnecting === undefined
          ? null
          : Number(cameras.reconnecting),
      offline:
        cameras.offline === null || cameras.offline === undefined
          ? null
          : Number(cameras.offline),
      activeWorkers:
        cameras.active_workers === null ||
        cameras.active_workers === undefined
          ? null
          : Number(cameras.active_workers),
    };
  }, [health]);

  const inference = health?.inference || {};
  const gpu = health?.ai?.gpu || inference?.gpu || {};
  const host = health?.host || {};
  const modelMetrics = inference?.model_metrics || {};
  const modelMetricRows = Object.entries(modelMetrics).filter(([name]) => name !== "totals");
  const cameraDetails = Array.isArray(health?.cameras?.details) ? health.cameras.details : [];

  const onlinePercentage = useMemo(() => {
    if (
      !Number.isFinite(cameraCounts.total) ||
      cameraCounts.total <= 0 ||
      !Number.isFinite(cameraCounts.online)
    ) {
      return null;
    }

    return Math.round(
      (cameraCounts.online / cameraCounts.total) * 100
    );
  }, [cameraCounts]);

  if (!isOpen) {
    return null;
  }

  const overallStatus = normalizeStatus(
    health?.overall,
    isInitializing ? "INITIALIZING" : "UNKNOWN"
  );

  const hasGpuTelemetry =
    gpu?.available === true &&
    (
      Number.isFinite(Number(gpu?.utilization_percent)) ||
      Number.isFinite(Number(gpu?.memory_percent)) ||
      Number.isFinite(Number(gpu?.temperature_c))
    );

  const initializationStartedAt =
    health?.initialization?.started_at;

  const initializationCompletedAt =
    health?.initialization?.completed_at;

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-3 backdrop-blur-sm sm:p-6"
      role="dialog"
      aria-modal="true"
      aria-labelledby="intel-i-system-health-title"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !isInitializing) {
          onClose?.();
        }
      }}
    >
      <div className="relative flex max-h-[94vh] w-full max-w-9xl flex-col overflow-hidden rounded-2xl border border-blue-400/20 bg-[#07111f] shadow-2xl shadow-black/50">
        <div className="pointer-events-none absolute -right-24 -top-24 h-64 w-64 rounded-full bg-blue-500/10 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-32 left-1/3 h-64 w-64 rounded-full bg-cyan-500/5 blur-3xl" />

        {/* HEADER */}
        <div className="relative flex items-start justify-between gap-4 border-b border-slate-700/60 px-5 py-4 sm:px-6 sm:py-5">
          <div className="flex min-w-0 items-center gap-3">
            <div
              className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border ${
                isInitializing
                  ? "border-blue-400/20 bg-blue-400/10"
                  : overallStatus === "DEGRADED"
                    ? "border-amber-400/20 bg-amber-400/10"
                    : "border-emerald-400/20 bg-emerald-400/10"
              }`}
            >
              <Activity
                size={21}
                className={
                  isInitializing
                    ? "animate-pulse text-blue-400"
                    : overallStatus === "DEGRADED"
                      ? "text-amber-400"
                      : "text-emerald-400"
                }
              />
            </div>

            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h2
                  id="intel-i-system-health-title"
                  className="text-base font-bold tracking-tight text-white sm:text-lg"
                >
                  INTEL-I System Health
                </h2>

                <StatusBadge
                  value={overallStatus}
                  initializing={isInitializing}
                />
              </div>

              <p className="mt-0.5 text-xs text-slate-500">
                {isInitializing
                  ? "System initialization and infrastructure status"
                  : "Backend, infrastructure, AI, camera and host health"}
              </p>
            </div>
          </div>

          <button
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-slate-700 bg-slate-900/70 text-slate-400 transition hover:border-slate-500 hover:bg-slate-800 hover:text-white focus:outline-none focus:ring-2 focus:ring-blue-400/30"
            aria-label="Close System Health"
            title="Close"
          >
            <X size={17} />
          </button>
        </div>

        <div className="relative min-h-0 flex-1 overflow-y-auto">
          {isInitializing ? (
            /* INITIALIZATION */
            <div className="p-4 sm:p-5">
              <div className="mb-4 rounded-xl border border-blue-400/15 bg-blue-400/5 px-4 py-3">
                <div className="flex items-center gap-3">
                  <RefreshCw
                    size={16}
                    className="animate-spin text-blue-400"
                  />
                  <div>
                    <p className="text-sm font-semibold text-blue-200">
                      Initializing INTEL-I services...
                    </p>
                    <p className="mt-0.5 text-xs text-slate-400">
                      {healthError
                        ? "Health service unavailable. Retry to continue the authoritative health check."
                        : healthLoading
                          ? "Checking system health..."
                          : "Waiting for authoritative health status..."}
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
                <SkeletonServiceCard icon={Server} title="Backend" />
                <SkeletonServiceCard icon={Database} title="PostgreSQL" />
                <SkeletonServiceCard icon={Wifi} title="Redis" />
                <SkeletonServiceCard icon={RadioTower} title="Kafka / Event Bus" />
                <SkeletonServiceCard icon={Activity} title="WebSocket" />
              </div>

              <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-2">
                <div className="animate-pulse rounded-2xl border border-slate-700/60 bg-[#091728] p-4">
                  <div className="h-3 w-32 rounded bg-slate-800" />
                  <div className="mt-2 h-5 w-44 rounded bg-slate-800" />
                  <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    {[1, 2, 3, 4].map((item) => (
                      <div key={item} className="h-20 rounded-xl bg-slate-800/80" />
                    ))}
                  </div>
                </div>

                <div className="animate-pulse rounded-2xl border border-slate-700/60 bg-[#091728] p-4">
                  <div className="h-3 w-32 rounded bg-slate-800" />
                  <div className="mt-2 h-5 w-44 rounded bg-slate-800" />
                  <div className="mt-4 grid grid-cols-2 gap-3">
                    {[1, 2, 3, 4].map((item) => (
                      <div key={item} className="h-20 rounded-xl bg-slate-800/80" />
                    ))}
                  </div>
                </div>
              </div>

              <div className="mt-4 rounded-xl border border-blue-400/10 bg-[#091728] px-4 py-3 text-xs text-blue-200/80">
                Waiting for the first valid authoritative health response.
                No camera, GPU, VRAM, CPU, RAM or disk values are displayed
                until initialization is complete.
              </div>

              {healthError && (
                <div className="mt-4 flex flex-col gap-3 rounded-xl border border-red-400/15 bg-red-400/5 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex items-start gap-2">
                    <AlertTriangle
                      size={15}
                      className="mt-0.5 shrink-0 text-red-400"
                    />
                    <div>
                      <p className="text-xs font-semibold text-red-300">
                        Health service unavailable
                      </p>
                      <p className="mt-0.5 text-[11px] text-slate-400">
                        {healthError}
                      </p>
                    </div>
                  </div>

                  <button
                    type="button"
                    onClick={onRefresh}
                    disabled={refreshing}
                    className="inline-flex shrink-0 items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-blue-400/40 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    <RefreshCw
                      size={13}
                      className={refreshing ? "animate-spin" : ""}
                    />
                    Retry
                  </button>
                </div>
              )}
            </div>
          ) : (
            /* AUTHORITATIVE HEALTH */
            <>
              <div className="border-b border-slate-700/60 px-4 py-4 sm:px-5">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-emerald-400/80">
                      System initialization completed
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      First authoritative readiness response received from
                      /api/system/health.
                    </p>
                  </div>

                  <div className="flex items-center gap-2">
                    {startupAutoCloseActive &&
                      countdown !== null && (
                        <span className="rounded-lg border border-blue-400/20 bg-blue-400/5 px-3 py-2 text-[11px] font-semibold text-blue-200">
                          Closing automatically in {countdown}s
                        </span>
                      )}

                    <button
                      type="button"
                      onClick={onRefresh}
                      disabled={refreshing}
                      className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-blue-400/40 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      <RefreshCw
                        size={13}
                        className={refreshing ? "animate-spin" : ""}
                      />
                      {refreshing ? "Refreshing" : "Refresh"}
                    </button>
                  </div>
                </div>

                {healthError && (
                  <div className="mt-3 rounded-lg border border-amber-400/15 bg-amber-400/5 px-3 py-2 text-[11px] text-amber-300">
                    Last refresh warning: {healthError}
                  </div>
                )}
              </div>

              {/* CORE SERVICES */}
              <div className="border-b border-slate-700/60 p-4 sm:p-5">
                <div className="mb-3">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                    Core Services
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Critical infrastructure availability
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-5">
                  <ServiceCard
                    icon={Server}
                    title="Backend"
                    status={health?.backend?.status}
                  />

                  <ServiceCard
                    icon={Database}
                    title="PostgreSQL"
                    status={health?.database?.status}
                    latency={health?.database?.latency_ms}
                  />

                  <ServiceCard
                    icon={Wifi}
                    title="Redis"
                    status={health?.redis?.status}
                    latency={health?.redis?.latency_ms}
                  />

                  <ServiceCard
                    icon={RadioTower}
                    title="Kafka / Event Bus"
                    status={health?.kafka?.status}
                    latency={health?.kafka?.latency_ms}
                  />

                  <ServiceCard
                    icon={Activity}
                    title="WebSocket"
                    status={health?.websocket?.status}
                  />
                </div>
              </div>

              {/* CAMERA + AI */}
              <div className="grid grid-cols-1 gap-4 border-b border-slate-700/60 p-4 sm:p-5 xl:grid-cols-2">
                <div className="rounded-2xl border border-slate-700/60 bg-[#091728] p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                        Camera Network
                      </p>
                      <p className="mt-1 text-sm font-bold text-white">
                        Camera Health
                      </p>
                    </div>

                    <StatusBadge value={health?.cameras?.status} />
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
                    <HealthMetric
                      icon={Gauge}
                      label="Total"
                      value={cameraCounts.total}
                    />
                    <HealthMetric
                      icon={CheckCircle2}
                      label="Online"
                      value={cameraCounts.online}
                    />
                    <HealthMetric
                      icon={AlertTriangle}
                      label="Degraded"
                      value={cameraCounts.degraded}
                    />
                    <HealthMetric
                      icon={CircleOff}
                      label="Offline"
                      value={cameraCounts.offline}
                    />
                  </div>

                  {Number.isFinite(cameraCounts.reconnecting) &&
                    cameraCounts.reconnecting > 0 && (
                      <div className="mt-3 rounded-lg border border-amber-400/15 bg-amber-400/5 px-3 py-2 text-xs text-amber-300">
                        Reconnecting cameras:{" "}
                        <span className="font-bold">
                          {cameraCounts.reconnecting}
                        </span>
                      </div>
                    )}

                  <div className="mt-4">
                    <div className="mb-1.5 flex items-center justify-between text-[10px] font-semibold uppercase tracking-wider text-slate-500">
                      <span>Online Coverage</span>
                      <span className="text-slate-300">
                        {onlinePercentage === null
                          ? "—"
                          : `${onlinePercentage}%`}
                      </span>
                    </div>
                    <ProgressBar value={onlinePercentage} />
                  </div>

                  <div className="mt-3 text-[10px] text-slate-500">
                    Active workers:{" "}
                    <span className="font-semibold text-slate-300">
                      {cameraCounts.activeWorkers ?? "—"}
                    </span>
                  </div>

                  {Array.isArray(health?.cameras?.details) &&
                    health.cameras.details.length > 0 && (
                      <details className="mt-4 overflow-hidden rounded-xl border border-slate-700/60 bg-slate-900/40">
                        <summary className="cursor-pointer px-3 py-2.5 text-xs font-semibold text-slate-300 transition hover:text-white">
                          Camera Details
                        </summary>
                        <div className="max-h-56 overflow-y-auto border-t border-slate-700/60">
                          {health.cameras.details.map((camera, index) => (
                            <div
                              key={
                                camera?.cam_id ||
                                camera?.camera_id ||
                                `camera-${index}`
                              }
                              className="flex items-center justify-between gap-3 border-b border-slate-800 px-3 py-2.5 last:border-b-0"
                            >
                              <div className="min-w-0">
                                <p className="truncate text-xs font-semibold text-slate-200">
                                  {camera?.camera_name ||
                                    camera?.cam_id ||
                                    camera?.camera_id ||
                                    `Camera ${index + 1}`}
                                </p>
                              </div>
                              <StatusBadge value={camera?.status} />
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                </div>

                <div className="rounded-2xl border border-slate-700/60 bg-[#091728] p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                        AI Processing
                      </p>
                      <p className="mt-1 text-sm font-bold text-white">
                        GPU / AI Health
                      </p>
                    </div>

                    <StatusBadge value={health?.ai?.status} />
                  </div>

                  <div className="mt-4 grid grid-cols-2 gap-3">
                    <HealthMetric
                      icon={Cpu}
                      label="GPU"
                      value={gpu?.utilization_percent}
                      unit="%"
                      unavailable={!hasGpuTelemetry}
                    />
                    <HealthMetric
                      icon={MemoryStick}
                      label="VRAM"
                      value={gpu?.memory_percent}
                      unit="%"
                      unavailable={!hasGpuTelemetry}
                    />
                    <HealthMetric
                      icon={Activity}
                      label="GPU Temp"
                      value={gpu?.temperature_c}
                      unit="°C"
                      unavailable={!hasGpuTelemetry}
                    />
                    <HealthMetric
                      icon={Gauge}
                      label="WS Clients"
                      value={health?.websocket?.active_clients}
                    />
                  </div>

                  {!hasGpuTelemetry && (
                    <div className="mt-3 rounded-lg border border-slate-700 bg-slate-900/60 px-3 py-2 text-xs text-slate-400">
                      GPU telemetry unavailable. The AI service can still
                      report its logical status.
                    </div>
                  )}
                </div>
              </div>

              {/* CENTRAL INFERENCE */}
              <div className="border-b border-slate-700/60 p-4 sm:p-5">
                <div className="mb-3 flex items-start justify-between gap-3">
                  <div>
                    <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">Central Inference</p>
                    <p className="mt-1 text-xs text-slate-500">Adaptive FPS, bounded queue, true primary batching and conditional model execution</p>
                  </div>
                  <StatusBadge value={inference?.status || "UNAVAILABLE"} />
                </div>

                <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 xl:grid-cols-6">
                  <HealthMetric icon={RadioTower} label="Inference Backend" value={inference?.backend || "local"} /><HealthMetric icon={RadioTower} label="AI Cameras" value={inference?.active_cameras} />
                  <HealthMetric icon={Gauge} label="Queue" value={inference?.queue_depth} unit={` / ${inference?.queue_capacity ?? "—"}`} />
                  <HealthMetric icon={Activity} label="Batch" value={inference?.current_batch_size} />
                  <HealthMetric icon={Gauge} label="Avg Batch" value={inference?.average_batch_size} />
                  <HealthMetric icon={Activity} label="Latency" value={inference?.average_inference_latency_ms} unit="ms" />
                  <HealthMetric icon={AlertTriangle} label="Dropped" value={inference?.dropped_frames} />
                  <HealthMetric icon={Cpu} label="NVDEC" value={inference?.gpu?.decoder_utilization_percent} unit="%" unavailable={inference?.gpu?.decoder_utilization_percent === null || inference?.gpu?.decoder_utilization_percent === undefined} />
                  <HealthMetric icon={Server} label="Decoder" value={inference?.decoder?.selected_device || "—"} />
                  <HealthMetric icon={Gauge} label="Load" value={inference?.scheduler?.load_factor !== undefined ? Math.round(Number(inference.scheduler.load_factor) * 100) : null} unit="%" />
                  <HealthMetric icon={RefreshCw} label="Expired" value={inference?.expired_frames ?? 0} /><HealthMetric icon={RefreshCw} label="Stale Replaced" value={inference?.replaced_stale_frames} />
                  <HealthMetric icon={AlertTriangle} label="Evicted" value={inference?.evicted_low_priority_frames} />
                  <HealthMetric icon={Activity} label="Requests" value={inference?.requests} />
                </div>

                {modelMetricRows.length > 0 && (
                  <details className="mt-4 overflow-hidden rounded-xl border border-slate-700/60 bg-slate-900/40">
                    <summary className="cursor-pointer px-3 py-2.5 text-xs font-semibold text-slate-300 transition hover:text-white">Model Invocation Metrics</summary>
                    <div className="overflow-x-auto border-t border-slate-700/60">
                      <table className="min-w-full text-left text-[11px]">
                        <thead className="bg-slate-900/70 text-slate-500">
                          <tr>
                            <th className="px-3 py-2">Model</th><th className="px-3 py-2">Calls</th><th className="px-3 py-2">Frames</th><th className="px-3 py-2">Avoided Cache</th><th className="px-3 py-2">Avoided Trigger</th><th className="px-3 py-2">Avg ms</th><th className="px-3 py-2">Avg GPU ms</th><th className="px-3 py-2">Errors</th>
                          </tr>
                        </thead>
                        <tbody>
                          {modelMetricRows.map(([name, metric]) => (
                            <tr key={name} className="border-t border-slate-800 text-slate-300">
                              <td className="px-3 py-2 font-mono text-blue-300">{name}</td>
                              <td className="px-3 py-2">{metric?.calls ?? 0}</td>
                              <td className="px-3 py-2">{metric?.frames ?? 0}</td>
                              <td className="px-3 py-2">{metric?.avoided_by_cache ?? 0}</td>
                              <td className="px-3 py-2">{metric?.avoided_by_trigger ?? 0}</td>
                              <td className="px-3 py-2">{metric?.average_latency_ms ?? 0}</td>
                              <td className="px-3 py-2">{metric?.average_gpu_ms ?? 0}</td>
                              <td className="px-3 py-2">{metric?.errors ?? 0}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </details>
                )}
              </div>

              {/* HOST */}
              <div className="p-4 sm:p-5">
                <div className="mb-3">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                    Host Resources
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Runtime resource utilization
                  </p>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                  <HealthMetric
                    icon={Cpu}
                    label="CPU"
                    value={host?.cpu_percent}
                    unit="%"
                  />
                  <HealthMetric
                    icon={MemoryStick}
                    label="RAM"
                    value={host?.ram_percent}
                    unit="%"
                  />
                  <HealthMetric
                    icon={HardDrive}
                    label="Disk"
                    value={host?.disk_percent}
                    unit="%"
                  />
                  <HealthMetric
                    icon={RadioTower}
                    label="WebSockets"
                    value={health?.websocket?.active_clients}
                  />
                </div>

                <div className="mt-4 flex flex-wrap items-center gap-3 text-[10px] text-slate-500">
                  <span>
                    Health endpoint:
                    <span className="ml-1 font-mono text-slate-400">
                      /api/system/health
                    </span>
                  </span>

                  {health?.timestamp && (
                    <span>
                      Updated:
                      <span className="ml-1 text-slate-400">
                        {formatIndianTime(health.timestamp)}
                      </span>
                    </span>
                  )}

                  {initializationStartedAt && (
                    <span>
                      Init started:
                      <span className="ml-1 text-slate-400">
                        {formatIndianTime(initializationStartedAt)}
                      </span>
                    </span>
                  )}

                  {initializationCompletedAt && (
                    <span>
                      Init completed:
                      <span className="ml-1 text-slate-400">
                        {formatIndianTime(initializationCompletedAt)}
                      </span>
                    </span>
                  )}
                </div>
              </div>
              {/* PER-CAMERA OPERATIONAL TELEMETRY */}
              <div className="border-b border-slate-700/60 p-4 sm:p-5">
                <div className="mb-3">
                  <p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                    Per-Camera Operational Telemetry
                  </p>
                  <p className="mt-1 text-xs text-slate-500">
                    Authoritative backend state, stream health, queue pressure and clock status
                  </p>
                </div>
                <div className="overflow-x-auto rounded-xl border border-slate-700/60">
                  <table className="min-w-full text-left text-xs">
                    <thead className="bg-slate-900/70 text-[10px] uppercase tracking-wider text-slate-500">
                      <tr>
                        <th className="px-3 py-2">Camera</th>
                        <th className="px-3 py-2">State</th>
                        <th className="px-3 py-2">FPS</th>
                        <th className="px-3 py-2">Latency</th>
                        <th className="px-3 py-2">Dropped</th>
                        <th className="px-3 py-2">Queue</th>
                        <th className="px-3 py-2">Clock</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800">
                      {cameraDetails.map((camera) => {
                        const stream = camera?.stream || {};
                        const clock = camera?.time_sync || {};
                        return (
                          <tr key={camera.cam_id} className="bg-[#091728] text-slate-300">
                            <td className="px-3 py-2 font-semibold text-white">
                              {camera.cam_id}
                              <div className="text-[10px] font-normal text-slate-500">{camera.camera_name}</div>
                            </td>
                            <td className="px-3 py-2"><StatusBadge value={camera.status} /></td>
                            <td className="px-3 py-2">{Number.isFinite(Number(stream.fps)) ? Number(stream.fps).toFixed(1) : "—"}</td>
                            <td className="px-3 py-2">{Number.isFinite(Number(stream.latency_ms)) ? `${Number(stream.latency_ms).toFixed(1)} ms` : "—"}</td>
                            <td className="px-3 py-2">{Number(stream.dropped_frames || 0)}</td>
                            <td className="px-3 py-2">{Number(stream.queue_depth || 0)} / {Number(stream.queue_capacity || 0)}</td>
                            <td className="px-3 py-2"><StatusBadge value={clock.sync_status || "UNKNOWN"} /></td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  {!cameraDetails.length && (
                    <div className="px-4 py-5 text-center text-xs text-slate-500">No camera telemetry available.</div>
                  )}
                </div>
              </div>

              <div className="border-b border-slate-700/60 p-4 sm:p-5">
                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <HealthMetric icon={Gauge} label="Stream Buffers" value={health?.stream_manager?.active_streams ?? "—"} unit="active" />
                  <HealthMetric icon={Activity} label="Buffer Capacity" value={health?.stream_manager?.buffer_capacity ?? "—"} unit="frames/camera" />
                  <HealthMetric icon={Wifi} label="Clock Offset Limit" value={health?.time_sync?.max_camera_clock_offset_ms ?? "—"} unit="ms" />
                </div>
              </div>

            </>
          )}
        </div>

        {/* FOOTER */}
        <div className="relative flex flex-col gap-2 border-t border-slate-700/60 bg-[#091728]/90 px-4 py-3 sm:flex-row sm:items-center sm:justify-between sm:px-5">
          <div className="text-[10px] text-slate-500">
            {isInitializing
              ? "Waiting for authoritative readiness..."
              : startupAutoCloseActive && countdown !== null
                ? "System initialization completed"
                : "System health is available on demand."}
          </div>

          <div className="flex items-center gap-2">
            {healthError && isInitializing && (
              <button
                type="button"
                onClick={onRefresh}
                disabled={refreshing}
                className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-xs font-semibold text-slate-200 transition hover:border-blue-400/40 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <RefreshCw
                  size={13}
                  className={refreshing ? "animate-spin" : ""}
                />
                Retry
              </button>
            )}

            <button
              type="button"
              onClick={onClose}
              className="inline-flex items-center justify-center rounded-lg border border-slate-600 bg-slate-800/80 px-4 py-2 text-xs font-semibold text-slate-200 transition hover:border-blue-400/50 hover:bg-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-400/30"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SystemHealth;