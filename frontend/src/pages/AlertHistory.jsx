import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { useLocation, useNavigate } from "react-router-dom";

import {
  SearchX,
  AlertTriangle,
  Clock,
  MapPin,
  Image as ImageIcon,
  RefreshCw,
  Camera,
  Activity,
  Navigation,
  Search,
  X,
  Eye,
  Car,
  UserRound,
  FileText,
  ShieldAlert,
} from "lucide-react";

import toast from "react-hot-toast";
import api from "../api/axios";
import { formatIndianDateTime as formatDateTime } from "../utils/dateTime";

const levelStyle = {
  HIGH: {
    border: "border-red-500/50",
    badge: "border-red-500/30 bg-red-500/10 text-red-300",
    icon: "text-red-300",
    dot: "bg-red-400",
  },

  MEDIUM: {
    border: "border-orange-500/50",
    badge: "border-orange-500/30 bg-orange-500/10 text-orange-300",
    icon: "text-orange-300",
    dot: "bg-orange-400",
  },

  LOW: {
    border: "border-yellow-500/50",
    badge: "border-yellow-500/20 bg-yellow-500/10 text-yellow-300",
    icon: "text-yellow-300",
    dot: "bg-yellow-400",
  },

  INFO: {
    border: "border-blue-500/50",
    badge: "border-blue-500/20 bg-blue-500/10 text-blue-300",
    icon: "text-blue-300",
    dot: "bg-blue-400",
  },
};

const defaultStyle = {
  border: "border-slate-700",
  badge: "border-slate-700 bg-slate-900 text-slate-300",
  icon: "text-slate-300",
  dot: "bg-slate-500",
};

/*
 * Alert APIs are not always returned in one identical envelope. Depending on
 * the backend version, the collection can be a direct array or live under
 * alerts/items/results/data. Keeping this compatibility layer prevents a
 * successful 200 response from being incorrectly rendered as an empty list.
 */
const extractAlertArray = (payload) => {
  const candidates = [
    payload,
    payload?.alerts,
    payload?.items,
    payload?.results,
    payload?.alert_history,
    payload?.data,
    payload?.data?.alerts,
    payload?.data?.items,
    payload?.data?.results,
    payload?.data?.alert_history,
  ];

  const collection = candidates.find(Array.isArray);

  return collection
    ? collection.filter((item) => item && typeof item === "object")
    : [];
};

const ALERT_REFRESH_INTERVAL_MS = 10_000;

const getAlertTypeMeta = (alert) => {
  const raw = [
    alert?.alert_rule,
    alert?.rule,
    alert?.alert_type,
    alert?.behavior,
    alert?.event_type,
    alert?.incident_type,
    alert?.watchlist_category,
    alert?.description,
    alert?.message,
  ]
    .filter(
      (value) =>
        value !== null && value !== undefined && String(value).trim() !== "",
    )
    .join(" ")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .toUpperCase();

  if (
    alert?.plate ||
    alert?.plate_number ||
    alert?.license_plate ||
    /\bANPR\b/.test(raw) ||
    /\bLPR\b/.test(raw) ||
    /\bLICENSE\s*PLATE\b/.test(raw) ||
    /\bPLATE\b/.test(raw) ||
    /\bVEHICLE\b/.test(raw) ||
    /\bCAR\b/.test(raw) ||
    /\bTRUCK\b/.test(raw) ||
    /\bBUS\b/.test(raw) ||
    /\bMOTORCYCLE\b/.test(raw) ||
    /\bBIKE\b/.test(raw) ||
    /\bSTOLEN\s*VEHICLE\b/.test(raw) ||
    /\bVEHICLE\s*WATCHLIST\b/.test(raw)
  ) {
    return {
      key: "VEHICLE",
      label: "VEHICLE",
      icon: Car,
      border: "border-l-cyan-500",
      borderSoft: "border-cyan-500/30",
      iconClass: "text-cyan-300",
      iconBg: "bg-cyan-500/10",
      iconRing: "ring-1 ring-cyan-500/10",
    };
  }

  if (
    /\bPERSON\b/.test(raw) ||
    /\bFACE\b/.test(raw) ||
    /\bFACIAL\b/.test(raw) ||
    /\bPERSON\s*WATCHLIST\b/.test(raw) ||
    /\bFACE\s*MATCH\b/.test(raw) ||
    /\bWANTED\s*PERSON\b/.test(raw) ||
    /\bMISSING\s*PERSON\b/.test(raw) ||
    /\bPERSON\s*RECOGNITION\b/.test(raw)
  ) {
    return {
      key: "PERSON",
      label: "PERSON",
      icon: UserRound,
      border: "border-l-purple-500",
      borderSoft: "border-purple-500/30",
      iconClass: "text-purple-300",
      iconBg: "bg-purple-500/10",
      iconRing: "ring-1 ring-purple-500/10",
    };
  }

  return {
    key: "BEHAVIOR",
    label: "BEHAVIORAL",
    icon: Activity,
    border: "border-l-amber-500",
    borderSoft: "border-amber-500/30",
    iconClass: "text-amber-300",
    iconBg: "bg-amber-500/10",
    iconRing: "ring-1 ring-amber-500/10",
  };
};

/* ============================================================
 * HELPERS
 * ============================================================ */

const safeText = (value, fallback = "N/A") => {
  if (value === null || value === undefined || value === "") {
    return fallback;
  }

  return String(value);
};

/*
 * Alert-producing services may return confidence as either a ratio
 * (0..1), a percentage (0..100), or a percentage string. Keep all
 * normalization here so cards and the detail view always agree.
 */
const getAlertConfidence = (alert) => {
  const candidates = [
    alert?.confidence_score,
    alert?.confidence,
    alert?.detection_confidence,
    alert?.match_confidence,
    alert?.recognition_confidence,
    alert?.face_confidence,
    alert?.plate_confidence,
    alert?.ocr_confidence,
    alert?.event_confidence,
    alert?.behavior_confidence,
    alert?.watchlist_match_confidence,
    alert?.score,
    alert?.metadata?.confidence_score,
    alert?.metadata?.confidence,
    alert?.details?.confidence_score,
    alert?.details?.confidence,
  ];

  const rawValue = candidates.find(
    (value) =>
      value !== null && value !== undefined && String(value).trim() !== "",
  );

  if (rawValue === undefined) {
    return null;
  }

  const rawText = String(rawValue).trim();
  const hasPercentSign = rawText.includes("%");
  const numericValue = Number(rawText.replace(/%/g, "").trim());

  if (!Number.isFinite(numericValue) || numericValue < 0) {
    return null;
  }

  const percentage = hasPercentSign
    ? numericValue
    : numericValue <= 1
      ? numericValue * 100
      : numericValue;

  if (percentage > 100) {
    return null;
  }

  return Math.round(percentage * 10) / 10;
};

const formatConfidenceScore = (alert) => {
  const confidence = getAlertConfidence(alert);

  return confidence === null ? "N/A" : `${confidence.toFixed(1)}%`;
};

const getConfidenceStyle = (alert) => {
  const confidence = getAlertConfidence(alert);

  if (confidence === null) {
    return "text-slate-400";
  }

  if (confidence >= 80) {
    return "text-emerald-300";
  }

  if (confidence >= 60) {
    return "text-amber-300";
  }

  return "text-red-300";
};

const normalizeAlertLevel = (alert) => {
  const value = String(
    alert?.severity ||
      alert?.alert_level ||
      alert?.level ||
      alert?.priority ||
      "",
  ).toUpperCase();

  if (value.includes("CRITICAL") || value.includes("HIGH")) {
    return "HIGH";
  }

  if (value.includes("WARNING") || value.includes("MEDIUM")) {
    return "MEDIUM";
  }

  if (value.includes("LOW")) {
    return "LOW";
  }

  return "INFO";
};

const getAlertType = (alert) => {
  const raw = String(
    alert?.alert_rule ||
      alert?.rule ||
      alert?.alert_type ||
      alert?.behavior ||
      "GENERAL_ALERT",
  );

  return raw.replace(/_/g, " ").replace(/\s+/g, " ").trim().toUpperCase();
};

const getAlertDomain = (alert) => {
  const text = [
    alert?.alert_rule,
    alert?.rule,
    alert?.alert_type,
    alert?.behavior,
    alert?.object_type,
    alert?.incident_type,
    alert?.watchlist_category,
  ]
    .filter(Boolean)
    .join(" ")
    .replace(/[_-]+/g, " ")
    .toUpperCase();

  if (
    alert?.plate ||
    alert?.plate_number ||
    alert?.license_plate ||
    /\b(ANPR|LPR|PLATE|VEHICLE|CAR|TRUCK|BUS|MOTORCYCLE|BIKE)\b/.test(text)
  ) {
    return {
      label: "VEHICLE",
      icon: Car,
      className: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
    };
  }

  if (
    /\b(PERSON|FACE|FACIAL|WANTED PERSON|MISSING PERSON|PERSON WATCHLIST|PERSON RECOGNITION)\b/.test(
      text,
    )
  ) {
    return {
      label: "PERSON",
      icon: UserRound,
      className: "border-purple-500/30 bg-purple-500/10 text-purple-300",
    };
  }

  return {
    label: "BEHAVIOR",
    icon: Activity,
    className: "border-amber-500/30 bg-amber-500/10 text-amber-300",
  };
};

const getSnapshotId = (item) => {
  if (item?.snapshot_id) {
    return item.snapshot_id;
  }

  if (item?.snapshot_path) {
    const match = String(item.snapshot_path).match(/snapshot\/(\d+)/);

    if (match?.[1]) {
      return match[1];
    }
  }

  return null;
};

const isUploadAlert = (alert) => {
  const sourceType = String(alert?.source_type || alert?.source || "")
    .trim()
    .toLowerCase();

  const cameraId = String(alert?.cam_id || alert?.camera_id || "")
    .trim()
    .toLowerCase();

  const uploadSources = new Set([
    "upload",
    "uploaded",
    "upload_video",
    "uploaded_video",
    "recorded",
    "recorded_video",
    "video_upload",
  ]);

  if (uploadSources.has(sourceType)) {
    return true;
  }

  return (
    cameraId.startsWith("upload_") ||
    cameraId.startsWith("uploaded_") ||
    cameraId.startsWith("recorded_")
  );
};

const getAlertUniqueKey = (alert) => {
  return String(
    alert.alert_id ||
      alert.id ||
      `${alert.cam_id || "unknown"}_${alert.rule || alert.alert_rule || alert.alert_type || "unknown"}_${alert.track_id || "global"}_${
        alert.timestamp || alert.created_at || ""
      }`,
  );
};

const deduplicateAlerts = (items) => {
  const seen = new Set();

  const result = [];

  for (const item of items) {
    const key = getAlertUniqueKey(item);

    if (seen.has(key)) {
      continue;
    }

    seen.add(key);
    result.push(item);
  }

  return result;
};

const getSearchText = (item) =>
  [
    item?.alert_rule,
    item?.rule,
    item?.alert_type,
    item?.cam_id,
    item?.camera_id,
    item?.source_type,
    item?.source,
    item?.track_id,
    item?.plate,
    item?.plate_number,
    item?.license_plate,
    item?.watchlist_category,
    item?.zone,
    item?.description,
    item?.message,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

/* ============================================================
 * COMPONENT
 * ============================================================ */

const AlertHistory = ({ onAlertSelect }) => {
  const location = useLocation();

  const navigate = useNavigate();

  const [alerts, setAlerts] = useState([]);

  const [snapshotUrls, setSnapshotUrls] = useState({});

  const [loading, setLoading] = useState(true);

  const [refreshing, setRefreshing] = useState(false);

  const [search, setSearch] = useState("");

  const [levelFilter, setLevelFilter] = useState("ALL");

  const [domainFilter, setDomainFilter] = useState("ALL");

  const [sourceFilter, setSourceFilter] = useState("ALL");

  const [selectedAlert, setSelectedAlert] = useState(null);

  const objectUrlsRef = useRef({});

  const fetchInFlightRef = useRef(false);

  /*
   * Stores references already opened during
   * this mounted AlertHistory instance.
   *
   * This protects against:
   * - repeated effect execution;
   * - React StrictMode;
   * - route-state cleanup renders;
   * - closing a modal causing the old alert to reopen.
   */
  const consumedReferenceRef = useRef(new Set());

  /* ==========================================================
   * SNAPSHOT URL CLEANUP
   * ========================================================== */

  const cleanupSnapshotUrls = useCallback(() => {
    Object.values(objectUrlsRef.current).forEach((url) => {
      if (url) {
        URL.revokeObjectURL(url);
      }
    });

    objectUrlsRef.current = {};

    setSnapshotUrls({});
  }, []);

  /* ==========================================================
   * LOAD SNAPSHOT
   * ========================================================== */

  const loadSnapshotBlob = useCallback(async (snapshotId) => {
    if (!snapshotId) {
      return null;
    }

    const key = String(snapshotId);

    if (objectUrlsRef.current[key]) {
      return objectUrlsRef.current[key];
    }

    const response = await api.get(
      `/snapshot/${encodeURIComponent(snapshotId)}`,
      {
        responseType: "blob",

        withCredentials: true,
      },
    );

    const contentType = response.headers?.["content-type"] || "";

    if (!contentType.startsWith("image/")) {
      throw new Error("Invalid snapshot response");
    }

    const blobUrl = URL.createObjectURL(response.data);

    objectUrlsRef.current[key] = blobUrl;

    return blobUrl;
  }, []);

  /* ==========================================================
   * SNAPSHOT BATCH
   * ========================================================== */

  const loadSnapshotsForAlerts = useCallback(
    async (items) => {
      const snapshotEntries = {};

      await Promise.allSettled(
        items.map(async (item) => {
          const snapshotId = getSnapshotId(item);

          if (!snapshotId) {
            return;
          }

          try {
            const blobUrl = await loadSnapshotBlob(snapshotId);

            if (blobUrl) {
              snapshotEntries[String(snapshotId)] = blobUrl;
            }
          } catch (error) {
            console.warn("Snapshot load failed:", snapshotId, error);
          }
        }),
      );

      setSnapshotUrls((previous) => ({
        ...previous,
        ...snapshotEntries,
      }));
    },
    [loadSnapshotBlob],
  );

  /* ==========================================================
   * FETCH ALERT HISTORY
   * ========================================================== */

  const fetchAlertHistory = useCallback(
    async ({ refresh = false, silent = false } = {}) => {
      if (fetchInFlightRef.current) {
        return;
      }

      fetchInFlightRef.current = true;

      try {
        if (refresh) {
          setRefreshing(true);
        } else if (!silent) {
          setLoading(true);
        }

        const response = await api.get("/alerts", {
          params: {
            limit: 500,
          },

          withCredentials: true,
        });

        const contentType = String(
          response.headers?.["content-type"] || "",
        ).toLowerCase();

        if (contentType.includes("text/html")) {
          throw new Error(
            "Alert API returned HTML instead of JSON. Check the Vite /alerts proxy target.",
          );
        }

        const rawAlerts = extractAlertArray(response.data);

        const uniqueAlerts = deduplicateAlerts(rawAlerts);

        setAlerts(uniqueAlerts);

        await loadSnapshotsForAlerts(uniqueAlerts);
      } catch (error) {
        console.error("Error fetching alert history:", error);

        // Background refresh failures must not erase alerts already visible
        // to an operator or repeatedly display toast notifications.
        if (!silent) {
          toast.error(
            error?.response?.data?.detail ||
              error?.response?.data?.message ||
              "Failed to fetch alerts",
          );
        }
      } finally {
        setLoading(false);
        setRefreshing(false);

        fetchInFlightRef.current = false;
      }
    },
    [loadSnapshotsForAlerts],
  );

  /* ==========================================================
   * INITIAL LOAD
   * ========================================================== */

  useEffect(() => {
    void fetchAlertHistory();

    const refreshIfVisible = () => {
      if (!document.hidden) {
        void fetchAlertHistory({ silent: true });
      }
    };

    const intervalId = window.setInterval(
      refreshIfVisible,
      ALERT_REFRESH_INTERVAL_MS,
    );

    const handleVisibilityChange = () => {
      if (!document.hidden) {
        void fetchAlertHistory({ silent: true });
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);

    return () => {
      window.clearInterval(intervalId);

      document.removeEventListener(
        "visibilitychange",
        handleVisibilityChange,
      );

      cleanupSnapshotUrls();
    };
  }, [fetchAlertHistory, cleanupSnapshotUrls]);

  /* ==========================================================
   * DEEP LINK / ASSISTANT REFERENCE
   *
   * Supports:
   *
   * /alert-history?alert_id=123
   *
   * and:
   *
   * location.state = {
   *   intelIReference: {
   *     type: "alert",
   *     id: "123"
   *   }
   * }
   *
   * IMPORTANT:
   *
   * The reference is consumed with replace:true immediately
   * after opening. This is what prevents the modal from
   * reopening after Close.
   * ========================================================== */

  useEffect(() => {
    if (!alerts.length) {
      return;
    }

    const params = new URLSearchParams(location.search);

    const stateReference = location.state?.intelIReference;

    const stateReferenceType = String(stateReference?.type || "")
      .trim()
      .toLowerCase();

    const stateAlertId =
      stateReference &&
      (stateReferenceType === "alert" ||
        stateReferenceType === "alerts" ||
        stateReferenceType === "alert_history" ||
        stateReferenceType === "alert-history")
        ? stateReference.id
        : null;

    const queryAlertId =
      params.get("alert_id") || params.get("alertId") || params.get("alert");

    const requestedAlertId = stateAlertId || queryAlertId;

    if (
      requestedAlertId === null ||
      requestedAlertId === undefined ||
      String(requestedAlertId).trim() === ""
    ) {
      return;
    }

    /*
     * Every assistant navigation includes a timestamp.
     * Therefore the same alert can still be opened again
     * later when the user explicitly clicks it again.
     */
    const referenceTimestamp =
      location.state?.intelIReferenceTimestamp ||
      location.state?.intelINavigationTime ||
      "";

    const referenceKey = `alert:${String(requestedAlertId)}:${String(referenceTimestamp)}`;

    if (consumedReferenceRef.current.has(referenceKey)) {
      return;
    }

    const match = alerts.find(
      (item) => String(item.alert_id || item.id) === String(requestedAlertId),
    );

    /*
     * Alerts load asynchronously.
     * Do not consume the reference until the alert exists.
     */
    if (!match) {
      return;
    }

    consumedReferenceRef.current.add(referenceKey);

    /*
     * Open ONCE.
     */
    setSelectedAlert(match);

    /*
     * Remove query-based reference.
     */
    params.delete("alert_id");

    params.delete("alertId");

    params.delete("alert");

    const nextSearch = params.toString();

    /*
     * Preserve unrelated stack metadata,
     * but remove the one-shot assistant reference.
     */
    const nextState = {
      ...(location.state || {}),

      intelIReferenceConsumed: true,

      intelIReferenceConsumedAt: Date.now(),
    };

    delete nextState.intelIReference;

    delete nextState.intelIReferenceSource;

    delete nextState.intelIReferenceTimestamp;

    /*
     * CRITICAL:
     *
     * Replace current Alert History entry.
     *
     * This means:
     *
     * Assistant
     *    ↓
     * Alert History
     *    ↓ clean using replace
     *
     * Browser Back still returns to Assistant.
     */
    navigate(
      {
        pathname: location.pathname,

        search: nextSearch ? `?${nextSearch}` : "",

        hash: location.hash || "",
      },
      {
        replace: true,
        state: nextState,
      },
    );
  }, [
    alerts,
    location.pathname,
    location.search,
    location.hash,
    location.state,
    navigate,
  ]);

  /* ==========================================================
   * FILTERS
   * ========================================================== */

  const filteredAlerts = useMemo(() => {
    const query = search.trim().toLowerCase();

    return alerts.filter((item) => {
      const level = normalizeAlertLevel(item);

      const domain = getAlertDomain(item);

      const matchesSearch = !query || getSearchText(item).includes(query);

      const matchesLevel = levelFilter === "ALL" || level === levelFilter;

      const matchesDomain =
        domainFilter === "ALL" || domain.label === domainFilter;

      const source = isUploadAlert(item) ? "UPLOAD" : "LIVE_CAMERA";

      const matchesSource = sourceFilter === "ALL" || source === sourceFilter;

      return matchesSearch && matchesLevel && matchesDomain && matchesSource;
    });
  }, [alerts, search, levelFilter, domainFilter, sourceFilter]);

  /* ==========================================================
   * STATS
   * ========================================================== */

  const stats = useMemo(() => {
    const high = alerts.filter(
      (item) => normalizeAlertLevel(item) === "HIGH",
    ).length;

    const medium = alerts.filter(
      (item) => normalizeAlertLevel(item) === "MEDIUM",
    ).length;

    const vehicle = alerts.filter(
      (item) => getAlertDomain(item).label === "VEHICLE",
    ).length;

    const person = alerts.filter(
      (item) => getAlertDomain(item).label === "PERSON",
    ).length;

    const behavior = alerts.filter(
      (item) => getAlertDomain(item).label === "BEHAVIOR",
    ).length;

    return {
      total: alerts.length,
      high,
      medium,
      vehicle,
      person,
      behavior,
    };
  }, [alerts]);

  const hasFilters =
    Boolean(search) ||
    levelFilter !== "ALL" ||
    domainFilter !== "ALL" ||
    sourceFilter !== "ALL";

  const clearFilters = () => {
    setSearch("");

    setLevelFilter("ALL");

    setDomainFilter("ALL");

    setSourceFilter("ALL");
  };

  /* ==========================================================
   * MAP
   * ========================================================== */

  const handleViewOnMap = useCallback(
    (alert) => {
      if (isUploadAlert(alert)) {
        toast.error(
          "Uploaded-video alerts do not have a live camera map location.",
        );

        return;
      }

      if (!alert?.cam_id) {
        toast.error("This alert is not associated with a camera.");

        return;
      }

      if (onAlertSelect) {
        onAlertSelect(alert);
      } else {
        toast(`Camera: ${alert.cam_id}`, {
          icon: "📍",
        });
      }
    },
    [onAlertSelect],
  );

  /* ==========================================================
   * OPEN / CLOSE DETAIL
   * ========================================================== */

  const openAlert = (alert) => {
    setSelectedAlert(alert);
  };

  /*
   * IMPORTANT:
   *
   * Closing the detail modal does NOT navigate.
   * It only clears local UI state.
   *
   * Browser Back and modal Close are two different operations.
   */
  const closeAlert = () => {
    setSelectedAlert(null);
  };

  /* ==========================================================
   * RENDER
   * ========================================================== */

  return (
    <div className="min-h-full space-y-5 rounded-2xl border border-slate-800 bg-[#07101b] p-4 text-white shadow-[0_15px_50px_rgba(0,0,0,0.18)]">
      {/* ======================================================
          HEADER
      ====================================================== */}

      <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-gradient-to-br from-[#091426] via-[#07101d] to-[#050b14] p-5">
        <div className="pointer-events-none absolute -right-16 -top-16 h-48 w-48 rounded-full bg-blue-500/10 blur-3xl" />

        <div className="relative flex flex-col gap-4 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <div className="mb-2 flex items-center gap-2">
              <div className="rounded-lg border border-blue-400/20 bg-blue-400/10 p-2 text-blue-300">
                <AlertTriangle size={18} />
              </div>

              <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-blue-300/80">
                Security Operations
              </span>
            </div>

            <h1 className="text-2xl font-bold tracking-tight">Alert History</h1>

            <p className="mt-1 text-sm text-slate-500">
              Review, filter and investigate alerts generated by the INTEL-I
              video intelligence pipeline.
            </p>
          </div>

          <button
            type="button"
            onClick={() => fetchAlertHistory({ refresh: true })}
            disabled={loading || refreshing}
            className="inline-flex items-center justify-center gap-2 rounded-lg border border-blue-500/30 bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <RefreshCw size={16} className={refreshing ? "animate-spin" : ""} />

            {refreshing ? "Refreshing..." : "Refresh Alerts"}
          </button>
        </div>
      </div>

      {/* ======================================================
          STATS
      ====================================================== */}

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-6">
        {[
          ["Total", stats.total, "text-blue-300"],
          ["High", stats.high, "text-red-300"],
          ["Medium", stats.medium, "text-orange-300"],
          ["Vehicle", stats.vehicle, "text-cyan-300"],
          ["Person", stats.person, "text-purple-300"],
          ["Behavior", stats.behavior, "text-amber-300"],
        ].map(([label, value, color]) => (
          <div
            key={label}
            className="rounded-xl border border-slate-800 bg-slate-950/60 p-3.5"
          >
            <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
              {label}
            </div>

            <div className={`mt-2 text-2xl font-bold ${color}`}>{value}</div>
          </div>
        ))}
      </div>

      {/* ======================================================
          FILTERS
      ====================================================== */}

      <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
        <div className="grid grid-cols-1 gap-2 xl:grid-cols-[minmax(0,1fr)_150px_150px_180px_auto]">
          <label className="relative">
            <Search
              size={16}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-600"
            />

            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search alert, camera, plate, track, source or watchlist..."
              className="w-full rounded-lg border border-slate-800 bg-[#08111d] py-2.5 pl-9 pr-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/50"
            />
          </label>

          <select
            value={levelFilter}
            onChange={(event) => setLevelFilter(event.target.value)}
            className="rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/50"
          >
            <option value="ALL">All severity</option>

            <option value="HIGH">High</option>

            <option value="MEDIUM">Medium</option>

            <option value="LOW">Low</option>

            <option value="INFO">Info</option>
          </select>

          <select
            value={domainFilter}
            onChange={(event) => setDomainFilter(event.target.value)}
            className="rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/50"
          >
            <option value="ALL">All domains</option>

            <option value="VEHICLE">Vehicle</option>

            <option value="PERSON">Person</option>

            <option value="BEHAVIOR">Behavior</option>
          </select>

          <select
            value={sourceFilter}
            onChange={(event) => setSourceFilter(event.target.value)}
            className="rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/50"
          >
            <option value="ALL">All sources</option>

            <option value="LIVE_CAMERA">Live Camera</option>

            <option value="UPLOAD">Upload</option>
          </select>

          {hasFilters ? (
            <button
              type="button"
              onClick={clearFilters}
              className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2.5 text-xs font-medium text-slate-400 hover:bg-slate-800 hover:text-white"
            >
              Clear
            </button>
          ) : (
            <div />
          )}
        </div>

        <div className="mt-3 flex items-center justify-between border-t border-slate-800 pt-3 text-[11px] text-slate-600">
          <span>
            Showing{" "}
            <span className="font-semibold text-slate-300">
              {filteredAlerts.length}
            </span>{" "}
            of{" "}
            <span className="font-semibold text-slate-300">
              {alerts.length}
            </span>{" "}
            alerts
          </span>

          <span>Limit: 500</span>
        </div>
      </div>

      {/* ======================================================
          CONTENT
      ====================================================== */}

      {loading ? (
        <div className="flex min-h-[50vh] items-center justify-center rounded-2xl border border-slate-800 bg-slate-950/40">
          <div className="text-center">
            <RefreshCw
              size={28}
              className="mx-auto mb-3 animate-spin text-blue-300"
            />

            <div className="text-sm text-slate-400">
              Loading alert history...
            </div>
          </div>
        </div>
      ) : filteredAlerts.length === 0 ? (
        <div className="flex min-h-[50vh] flex-col items-center justify-center rounded-2xl border border-slate-800 bg-slate-950/40 px-6 text-center">
          <div className="rounded-full border border-slate-800 bg-slate-900 p-5">
            <SearchX size={42} className="text-slate-600" />
          </div>

          <div className="mt-4 text-lg font-semibold text-slate-300">
            No alerts found
          </div>

          <div className="mt-1 max-w-md text-sm leading-6 text-slate-600">
            No alerts match the current search and filter criteria, or the
            backend has not generated any alerts.
          </div>

          {hasFilters && (
            <button
              type="button"
              onClick={clearFilters}
              className="mt-4 rounded-lg bg-slate-800 px-4 py-2 text-xs text-slate-300 hover:bg-slate-700"
            >
              Clear filters
            </button>
          )}
        </div>
      ) : (
        <div className="space-y-3">
          {filteredAlerts.map((item, index) => {
            const level = normalizeAlertLevel(item);

            const severityStyle = levelStyle[level] || defaultStyle;

            const typeStyle = getAlertTypeMeta(item);

            const TypeIcon = typeStyle.icon;

            const snapshotId = getSnapshotId(item);

            const snapshotURL = snapshotId
              ? snapshotUrls[String(snapshotId)]
              : null;

            const uploadAlert = isUploadAlert(item);

            const alertKey =
              item.alert_id ||
              item.id ||
              `${item.cam_id || "unknown"}_${
                item.alert_rule || item.rule || item.alert_type || "unknown"
              }_${item.track_id || "global"}_${index}`;

            const plate = item.plate || item.plate_number || item.license_plate;

            return (
              <article
                key={alertKey}
                className={`rounded-2xl border-l-4 ${typeStyle.border} border border-slate-800 bg-[#07101b] p-4 shadow-[0_12px_40px_rgba(0,0,0,0.16)] transition hover:border-slate-700`}
              >
                <div className="flex flex-col gap-5 xl:flex-row">
                  {/* LEFT */}

                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="flex min-w-0 items-center gap-3">
                        <div
                          title={typeStyle.label}
                          aria-label={typeStyle.label}
                          className={`flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border ${typeStyle.borderSoft} ${typeStyle.iconBg} ${typeStyle.iconRing}`}
                        >
                          <TypeIcon size={21} className={typeStyle.iconClass} />
                        </div>

                        <div className="min-w-0">
                          <button
                            type="button"
                            onClick={() => openAlert(item)}
                            className="block max-w-[620px] truncate text-left text-lg font-bold text-slate-100 hover:text-blue-300"
                          >
                            {getAlertType(item)}
                          </button>

                          <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-600">
                            <span>#{safeText(item.alert_id || item.id)}</span>

                            <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900/80 px-2 py-0.5 text-[10px] font-semibold text-slate-300">
                              <TypeIcon size={10} />

                              {typeStyle.label}
                            </span>
                          </div>
                        </div>
                      </div>

                      <div className="flex shrink-0 items-center gap-2">
                        <span
                          className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase tracking-wide ${severityStyle.badge}`}
                          title="Alert severity"
                        >
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${severityStyle.dot}`}
                          />

                          {level}
                        </span>
                      </div>
                    </div>

                    {/* DETAILS */}

                    <div className="mt-5 grid grid-cols-1 gap-2 md:grid-cols-2 2xl:grid-cols-4">
                      <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2.5">
                        <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                          Camera / Source
                        </div>

                        <div className="flex items-center gap-2 text-xs text-slate-300">
                          <Camera size={14} className="text-blue-300" />

                          <span className="truncate">
                            {uploadAlert
                              ? "Uploaded Video"
                              : safeText(item.cam_id || item.camera_id)}
                          </span>
                        </div>
                      </div>

                      <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2.5">
                        <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                          Source
                        </div>

                        <div className="flex items-center gap-2 text-xs text-slate-300">
                          {uploadAlert ? (
                            <ImageIcon size={14} className="text-violet-300" />
                          ) : (
                            <Camera size={14} className="text-emerald-300" />
                          )}

                          <span className="truncate">
                            {uploadAlert ? "UPLOAD" : "LIVE CAMERA"}
                          </span>
                        </div>
                      </div>

                      <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2.5">
                        <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                          Time
                        </div>

                        <div className="flex items-center gap-2 text-xs text-slate-300">
                          <Clock size={14} className="text-amber-300" />

                          <span className="truncate">
                            {formatDateTime(item.created_at || item.timestamp)}
                          </span>
                        </div>
                      </div>

                      <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2.5">
                        <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                          Track
                        </div>

                        <div className="flex items-center gap-2 text-xs text-slate-300">
                          <Activity size={14} className="text-purple-300" />

                          <span className="truncate">
                            {safeText(item.track_id)}
                          </span>
                        </div>
                      </div>

                      <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2.5">
                        <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                          Confidence Score
                        </div>

                        <div className="flex items-center gap-2 text-xs">
                          <Activity size={14} className="text-emerald-300" />

                          <span
                            className={`font-mono font-semibold ${getConfidenceStyle(item)}`}
                          >
                            {formatConfidenceScore(item)}
                          </span>
                        </div>
                      </div>

                      {plate && (
                        <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2.5">
                          <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                            Registration
                          </div>

                          <div className="font-mono text-xs font-semibold text-cyan-300">
                            {plate}
                          </div>
                        </div>
                      )}

                      {(item.watchlist_category || item.watchlist_entry_id) && (
                        <div className="rounded-lg border border-red-500/10 bg-red-500/[0.03] px-3 py-2.5">
                          <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-red-400/60">
                            Watchlist
                          </div>

                          <div className="flex items-center gap-2 text-xs text-red-300">
                            <ShieldAlert size={13} />

                            <span className="truncate">
                              {safeText(item.watchlist_category, "Exact Match")}
                            </span>
                          </div>
                        </div>
                      )}

                      {item.zone && (
                        <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3 py-2.5">
                          <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                            Zone
                          </div>

                          <div className="flex items-center gap-2 text-xs text-slate-300">
                            <MapPin size={14} className="text-purple-300" />

                            <span className="truncate">{item.zone}</span>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* MESSAGE */}

                    {(item.message || item.description) && (
                      <div className="mt-3 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                        <div className="mb-1 text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                          Alert Context
                        </div>

                        <div className="whitespace-pre-wrap text-xs leading-5 text-slate-400">
                          {item.message || item.description}
                        </div>
                      </div>
                    )}

                    {/* ACTIONS */}

                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      {!uploadAlert && item.cam_id && (
                        <button
                          type="button"
                          onClick={() => handleViewOnMap(item)}
                          className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-xs font-semibold text-white hover:bg-blue-500"
                        >
                          <Navigation size={14} />
                          View on Map
                        </button>
                      )}

                      <button
                        type="button"
                        onClick={() => openAlert(item)}
                        className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800 hover:text-white"
                      >
                        <Eye size={14} />
                        View Details
                      </button>
                    </div>
                  </div>

                  {/* SNAPSHOT */}

                  <div className="flex shrink-0 flex-col gap-2 xl:w-[230px]">
                    <div className="text-right text-[9px] font-semibold uppercase tracking-[0.14em] text-slate-600 xl:text-left">
                      Alert Snapshot
                    </div>

                    {snapshotURL ? (
                      <button
                        type="button"
                        onClick={() => openAlert(item)}
                        className="group relative h-[210px] w-full overflow-hidden rounded-xl border border-slate-700 bg-[#03080e]"
                      >
                        <img
                          src={snapshotURL}
                          alt="Alert snapshot"
                          className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                        />

                        <div className="absolute inset-0 flex items-center justify-center bg-black/40 opacity-0 transition group-hover:opacity-100">
                          <Eye size={22} />
                        </div>
                      </button>
                    ) : (
                      <div className="flex h-[210px] w-full flex-col items-center justify-center rounded-xl border border-slate-700 bg-[#03080e] text-center text-slate-600">
                        {snapshotId ? (
                          <>
                            <RefreshCw size={24} className="animate-spin" />

                            <span className="mt-2 text-xs">
                              Loading snapshot...
                            </span>
                          </>
                        ) : (
                          <>
                            <ImageIcon size={42} />

                            <span className="mt-2 text-xs">No snapshot</span>
                          </>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}

      {/* ======================================================
          ALERT DETAIL MODAL
      ====================================================== */}

      {selectedAlert && (
        <div
          className="fixed inset-0 z-[130] flex items-center justify-center bg-black/80 p-4 backdrop-blur-md"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              closeAlert();
            }
          }}
        >
          <div className="max-h-[92vh] w-full max-w-5xl overflow-hidden rounded-2xl border border-slate-700 bg-[#07101b] shadow-[0_30px_120px_rgba(0,0,0,0.55)]">
            <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  {(() => {
                    const typeStyle = getAlertTypeMeta(selectedAlert);

                    const TypeIcon = typeStyle.icon;

                    const severity = normalizeAlertLevel(selectedAlert);

                    const severityStyle = levelStyle[severity] || defaultStyle;

                    return (
                      <>
                        <span className="inline-flex items-center gap-1.5 rounded-full border border-slate-700 bg-slate-900/80 px-2.5 py-1 text-[10px] font-semibold text-slate-300">
                          <TypeIcon size={12} />

                          {typeStyle.label}
                        </span>

                        <span
                          className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-bold uppercase ${severityStyle.badge}`}
                        >
                          <span
                            className={`h-1.5 w-1.5 rounded-full ${severityStyle.dot}`}
                          />

                          {severity}
                        </span>
                      </>
                    );
                  })()}
                </div>

                <h2 className="mt-2 text-lg font-semibold text-white">
                  {getAlertType(selectedAlert)}
                </h2>
              </div>

              <button
                type="button"
                onClick={closeAlert}
                className="rounded-lg p-2 text-slate-500 hover:bg-slate-800 hover:text-white"
                aria-label="Close alert details"
              >
                <X size={19} />
              </button>
            </div>

            <div className="max-h-[calc(92vh-76px)] overflow-y-auto p-5">
              <div className="grid gap-5 lg:grid-cols-[1fr_320px]">
                {/* EVIDENCE */}

                <div>
                  {(() => {
                    const snapshotId = getSnapshotId(selectedAlert);

                    const snapshotURL = snapshotId
                      ? snapshotUrls[String(snapshotId)]
                      : null;

                    return snapshotURL ? (
                      <div className="overflow-hidden rounded-xl border border-slate-800 bg-black">
                        <img
                          src={snapshotURL}
                          alt="Alert evidence"
                          className="max-h-[65vh] w-full object-contain"
                        />
                      </div>
                    ) : (
                      <div className="flex min-h-[420px] items-center justify-center rounded-xl border border-slate-800 bg-black text-slate-600">
                        <div className="text-center">
                          <ImageIcon size={52} className="mx-auto mb-3" />

                          <div className="text-sm">Snapshot not available</div>
                        </div>
                      </div>
                    );
                  })()}
                </div>

                {/* INFORMATION */}

                <div className="space-y-4">
                  <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                    <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                      Alert Information
                    </div>

                    <div className="space-y-3 text-xs">
                      <div className="flex justify-between gap-3">
                        <span className="text-slate-600">Alert ID</span>

                        <span className="font-mono text-slate-300">
                          {safeText(selectedAlert.alert_id || selectedAlert.id)}
                        </span>
                      </div>

                      <div className="flex justify-between gap-3">
                        <span className="text-slate-600">Camera</span>

                        <span className="text-right text-slate-300">
                          {isUploadAlert(selectedAlert)
                            ? "Uploaded Video"
                            : safeText(
                                selectedAlert.cam_id || selectedAlert.camera_id,
                              )}
                        </span>
                      </div>

                      <div className="flex justify-between gap-3">
                        <span className="text-slate-600">Track</span>

                        <span className="font-mono text-slate-300">
                          {safeText(selectedAlert.track_id)}
                        </span>
                      </div>

                      <div className="flex justify-between gap-3">
                        <span className="text-slate-600">Confidence Score</span>

                        <span
                          className={`font-mono font-semibold ${getConfidenceStyle(selectedAlert)}`}
                        >
                          {formatConfidenceScore(selectedAlert)}
                        </span>
                      </div>

                      <div className="flex justify-between gap-3">
                        <span className="text-slate-600">Time</span>

                        <span className="text-right text-slate-300">
                          {formatDateTime(
                            selectedAlert.created_at || selectedAlert.timestamp,
                          )}
                        </span>
                      </div>

                      {(selectedAlert.plate ||
                        selectedAlert.plate_number ||
                        selectedAlert.license_plate) && (
                        <div className="flex justify-between gap-3">
                          <span className="text-slate-600">Plate</span>

                          <span className="font-mono font-semibold text-cyan-300">
                            {selectedAlert.plate ||
                              selectedAlert.plate_number ||
                              selectedAlert.license_plate}
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {(selectedAlert.message || selectedAlert.description) && (
                    <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                      <div className="mb-2 flex items-center gap-2">
                        <FileText size={14} className="text-slate-600" />

                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                          Context
                        </span>
                      </div>

                      <div className="whitespace-pre-wrap text-xs leading-5 text-slate-400">
                        {selectedAlert.message || selectedAlert.description}
                      </div>
                    </div>
                  )}

                  {(selectedAlert.watchlist_category ||
                    selectedAlert.watchlist_entry_id) && (
                    <div className="rounded-xl border border-red-500/10 bg-red-500/[0.03] p-4">
                      <div className="mb-2 flex items-center gap-2">
                        <ShieldAlert size={15} className="text-red-300" />

                        <span className="text-[10px] font-semibold uppercase tracking-[0.14em] text-red-300/70">
                          Watchlist Finding
                        </span>
                      </div>

                      <div className="text-sm font-semibold text-red-200">
                        {selectedAlert.watchlist_category || "Exact Match"}
                      </div>

                      {selectedAlert.watchlist_entry_id && (
                        <div className="mt-1 text-xs text-slate-600">
                          Entry #{selectedAlert.watchlist_entry_id}
                        </div>
                      )}
                    </div>
                  )}

                  {!isUploadAlert(selectedAlert) && selectedAlert.cam_id && (
                    <button
                      type="button"
                      onClick={() => {
                        handleViewOnMap(selectedAlert);

                        closeAlert();
                      }}
                      className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-500"
                    >
                      <Navigation size={16} />
                      View on Map
                    </button>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AlertHistory;
