import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  SearchX,
  AlertTriangle,
  Camera,
  Clock,
  MapPin,
  Activity,
  Image as ImageIcon,
  Loader2,
} from "lucide-react";

import api, {
  WEBSOCKET_URL,
} from "../api/axios";
import { formatIndianDateTime as formatDateTime } from "../utils/dateTime";

// ============================================================
// CONFIGURATION
// ============================================================

const MAX_ALERTS = 50;

const BASE_RECONNECT_DELAY_MS = 3000;

const MAX_RECONNECT_DELAY_MS = 30000;

const HEARTBEAT_CHECK_MS = 30000;

const HEARTBEAT_TIMEOUT_MS = 90000;

// Delay the first socket open very slightly. React StrictMode intentionally
// mounts/unmounts effects twice in development; the delay lets the throw-away
// effect clean itself up before a CONNECTING WebSocket is created.
const INITIAL_CONNECT_DELAY_MS = 100;

// ============================================================
// ALERT STYLES
// ============================================================

const LEVEL_STYLES = Object.freeze({
  HIGH: {
    border: "border-red-500",
    badge: "bg-red-600 text-white",
    icon: "text-red-300",
  },

  MEDIUM: {
    border: "border-orange-500",
    badge: "bg-orange-500 text-white",
    icon: "text-orange-200",
  },

  LOW: {
    border: "border-yellow-500",
    badge: "bg-yellow-500 text-black",
    icon: "text-yellow-300",
  },

  INFO: {
    border: "border-blue-500",
    badge: "bg-blue-600 text-white",
    icon: "text-blue-300",
  },

  // Backward-compatible aliases for alerts persisted before the new
  // four-level presentation contract was introduced.
  CRITICAL: {
    border: "border-red-500",
    badge: "bg-red-600 text-white",
    icon: "text-red-300",
  },
  WARNING: {
    border: "border-orange-500",
    badge: "bg-orange-500 text-white",
    icon: "text-orange-200",
  },
});

const DEFAULT_LEVEL_STYLE =
  LEVEL_STYLES.LOW;

// ============================================================
// ALERT LEVEL
// ============================================================

const getAlertLevel = (type) => {
  const value = String(type || "").toUpperCase();

  if (value.includes("CRITICAL") || value.includes("HIGH")) {
    return "HIGH";
  }

  if (value.includes("WARNING") || value.includes("MEDIUM")) {
    return "MEDIUM";
  }

  if (value.includes("INFO")) {
    return "INFO";
  }

  return "INFO";
};

// ============================================================
// SNAPSHOT PATH
// ============================================================

const getSnapshotPath = (
  alert
) => {
  const path =
    alert?.snapshot_path ||
    alert?.snapshotPath ||
    alert?.snapshot_url ||
    alert?.snapshotUrl ||
    alert?.image_url ||
    alert?.imageUrl ||
    "";

  return typeof path ===
    "string"
    ? path.trim()
    : "";
};


const buildAlertKey = (
  alert
) => {
  if (alert?.alert_id || alert?.id) return `alert:${alert.alert_id || alert.id}`;
  return [
    alert?.alert_id ||
      alert?.id ||
      "no_id",

    alert?.cam_id ||
      "unknown",

    alert?.rule ||
      alert?.alert_rule ||
      "unknown",

    alert?.track_id ||
      "global",

    alert?.created_at ||
      alert?.timestamp ||
      "",
  ].join("_");
};


const getSafeSnapshotPath =
  (snapshotPath) => {
    if (
      !snapshotPath ||
      typeof snapshotPath !==
        "string"
    ) {
      return "";
    }

    if (
      snapshotPath.startsWith(
        "/"
      ) &&
      !snapshotPath.startsWith(
        "//"
      )
    ) {
      return snapshotPath;
    }

    /*
     * Allow same-origin absolute URLs only.
     */
    try {
      const url =
        new URL(
          snapshotPath,
          window.location.origin
        );

      if (
        url.origin !==
        window.location.origin
      ) {
        return "";
      }

      if (
        url.protocol !==
          "http:" &&
        url.protocol !==
          "https:"
      ) {
        return "";
      }

      return (
        url.pathname +
        url.search
      );
    } catch {
      return "";
    }
  };

// ============================================================
// SNAPSHOT IMAGE
// ============================================================

const SnapshotImage = ({
  snapshotPath,
}) => {
  const [
    snapshotURL,
    setSnapshotURL,
  ] = useState("");

  const [
    loading,
    setLoading,
  ] = useState(
    Boolean(
      snapshotPath
    )
  );

  const [
    failed,
    setFailed,
  ] = useState(false);

  useEffect(() => {
    let objectUrl = "";

    let cancelled =
      false;

    const safePath =
      getSafeSnapshotPath(
        snapshotPath
      );

    const loadSnapshot =
      async () => {
        setSnapshotURL(
          ""
        );

        setFailed(
          false
        );

        if (!safePath) {
          setLoading(
            false
          );

          return;
        }

        try {
          setLoading(
            true
          );

          const response =
            await api.get(
              safePath,
              {
                responseType:
                  "blob",
              }
            );

          if (
            cancelled
          ) {
            return;
          }

          const contentType =
            response
              ?.headers?.[
              "content-type"
            ] || "";

          if (
            !contentType.startsWith(
              "image/"
            )
          ) {
            throw new Error(
              "Invalid image content type"
            );
          }

          objectUrl =
            URL.createObjectURL(
              response.data
            );

          setSnapshotURL(
            objectUrl
          );
        } catch {
          if (
            !cancelled
          ) {
            setFailed(
              true
            );
          }
        } finally {
          if (
            !cancelled
          ) {
            setLoading(
              false
            );
          }
        }
      };

    loadSnapshot();

    return () => {
      cancelled = true;

      if (
        objectUrl
      ) {
        URL.revokeObjectURL(
          objectUrl
        );
      }
    };
  }, [
    snapshotPath,
  ]);

  if (
    !snapshotPath ||
    failed
  ) {
    return (
      <div className="mb-4 w-full h-[220px] rounded-xl border border-[#1f3f63] bg-[#081321] flex flex-col items-center justify-center text-slate-400">
        <ImageIcon
          size={32}
        />

        <p className="text-sm mt-2">
          No snapshot available
        </p>
      </div>
    );
  }

  if (
    loading ||
    !snapshotURL
  ) {
    return (
      <div className="mb-4 w-full h-[220px] rounded-xl border border-[#1f3f63] bg-[#081321] flex flex-col items-center justify-center text-slate-400">
        <Loader2
          size={30}
          className="animate-spin"
        />

        <p className="text-sm mt-2">
          Loading snapshot...
        </p>
      </div>
    );
  }

  return (
    <a
      href={
        snapshotURL
      }
      target="_blank"
      rel="noopener noreferrer"
      className="block mb-4 w-full h-[220px] rounded-xl border border-[#1f3f63] overflow-hidden bg-[#070d16] group"
    >
      <img
        src={
          snapshotURL
        }
        alt="Alert snapshot"
        loading="lazy"
        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300"
      />
    </a>
  );
};


const isValidAlertPayload = (value) => {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value)
  ) {
    return false;
  }

  const hasID =
    value.alert_id !== undefined &&
    value.alert_id !== null
      ? true
      : value.id !== undefined &&
        value.id !== null;

  const hasRule = Boolean(
    value.rule ||
    value.alert_rule ||
    value.event_type
  );

  const hasSource = Boolean(
    value.cam_id ||
    value.camera_id ||
    value.source_type ||
    value.source
  );

  return hasID && hasRule && hasSource;
};

// ============================================================
// ALERT NORMALIZATION / FILTERING
// ============================================================

const normalizeAlert = (value) => {
  if (!isValidAlertPayload(value)) {
    return null;
  }

  const severity = getAlertLevel(
    value.severity ||
      value.alert_level ||
      value.level ||
      value.alert_type
  );

  const sourceType = String(
    value.source_type ||
      value.source ||
      "unknown"
  )
    .trim()
    .toLowerCase();

  const rule =
    value.rule ||
    value.alert_rule ||
    value.event_type;

  return {
    ...value,
    type: "alert",
    severity,
    source_type: sourceType,
    cam_id:
      value.cam_id ||
      value.camera_id ||
      null,
    alert_id:
      value.alert_id ||
      value.id ||
      null,
    id:
      value.id ||
      value.alert_id ||
      null,
    rule,
    alert_rule: rule,
  };
};


// ============================================================
// ALERT SOURCE CLASSIFICATION
// ============================================================

const isUploadAlertPayload = (
  alert,
  uploadVideoMap = {}
) => {
  const sourceType = String(
    alert?.source_type ||
      alert?.source ||
      ""
  )
    .trim()
    .toLowerCase();

  const cameraId = String(
    alert?.cam_id ||
      alert?.camera_id ||
      ""
  )
    .trim()
    .toLowerCase();

  const uploadVideoInfo =
    uploadVideoMap?.[
      alert?.cam_id ||
        alert?.camera_id
    ];

  return (
    Boolean(uploadVideoInfo) ||
    [
      "upload",
      "uploaded",
      "upload_video",
      "uploaded_video",
      "recorded",
      "recorded_video",
      "video_upload",
    ].includes(sourceType) ||
    cameraId.startsWith("upload_") ||
    cameraId.startsWith("uploaded_") ||
    cameraId.startsWith("recorded_")
  );
};
// ============================================================
// ALERT
// ============================================================

const Alert = ({
  filterCamID,
  allowedTypes,
  uploadVideoMap = {},
  sourceMode = "live",
  title,
  emptyMessage,
  onAlertSelect,
}) => {
  const [
    alerts,
    setAlerts,
  ] = useState([]);

  // This component is intentionally page-scoped:
  //   "live"   -> Dashboard / live-camera pages only
  //   "upload" -> Upload Video page only
  // There is no source-selection tab inside the alert panel.
  const normalizedSourceMode =
    String(sourceMode).trim().toLowerCase() === "upload"
      ? "upload"
      : "live";

  const panelTitle =
    title ||
    (normalizedSourceMode === "upload"
      ? "Upload Video Alerts"
      : "Live Alerts");

  const panelEmptyMessage =
    emptyMessage ||
    (normalizedSourceMode === "upload"
      ? "No current upload-video alerts"
      : "No current live-camera alerts");

  const sessionStartRef = useRef(new Date().toISOString());

  const wsRef =
    useRef(null);

  const reconnectTimerRef =
    useRef(null);

  const initialConnectTimerRef =
    useRef(null);

  const heartbeatTimerRef =
    useRef(null);

  const reconnectAttemptsRef =
    useRef(0);

  const mountedRef =
    useRef(false);

  const intentionallyClosedRef =
    useRef(false);

  const lastHeartbeatRef =
    useRef(Date.now());

  // ==========================================================
  // ALLOWED TYPES KEY
  // ==========================================================

  const allowedTypesKey =
    useMemo(() => {
      if (
        !Array.isArray(
          allowedTypes
        )
      ) {
        return "";
      }

      return allowedTypes
        .map(
          (value) =>
            String(value)
        )
        .join("|");
    }, [
      allowedTypes,
    ]);

  const uploadVideoMapKey =
    useMemo(
      () =>
        Object.keys(
          uploadVideoMap || {}
        )
          .sort()
          .join("|"),
      [uploadVideoMap]
    );

  // ==========================================================
  // WEBSOCKET
  // ==========================================================

  useEffect(() => {
    mountedRef.current =
      true;

    intentionallyClosedRef.current =
      false;

    // Keep current-session alerts when upload metadata changes the socket effect.

    const acceptAlert = (rawAlert) => {
      const data = normalizeAlert(rawAlert);

      if (!data) {
        return null;
      }

      if (
        filterCamID &&
        String(data.cam_id || "") !== String(filterCamID)
      ) {
        return null;
      }

      if (Array.isArray(allowedTypes) && allowedTypes.length > 0) {
        const allowed = allowedTypes.map((item) =>
          String(item).trim().toLowerCase()
        );

        if (!allowed.includes(String(data.source_type || "").toLowerCase())) {
          return null;
        }
      }

      const isUploadAlert =
        isUploadAlertPayload(
          data,
          uploadVideoMap
        );

      // Strict page-level source isolation:
      // Dashboard receives only live-camera alerts.
      // Upload Video receives only uploaded-video alerts.
      if (
        normalizedSourceMode === "upload" &&
        !isUploadAlert
      ) {
        return null;
      }

      if (
        normalizedSourceMode === "live" &&
        isUploadAlert
      ) {
        return null;
      }

      return data;
    };

    let disposed = false;
    let recovering = false;
    let recoveryCursor = 0;
    const mergeAlert = (alert) => {
      setAlerts((previous) => {
        const key = buildAlertKey(alert);
        const existing = previous.find((item) => buildAlertKey(item) === key);
        if (existing) {
          // Recovery has fewer evidence fields than the live event; retain them.
          return previous;
        }
        return [alert, ...previous].slice(0, MAX_ALERTS);
      });
    };
    const recoverSessionAlerts = async () => {
      if (disposed || recovering) return;
      recovering = true;
      try {
        for (let page = 0; page < 10 && !disposed; page += 1) {
          const response = await api.get("/alerts/recover", { params: {
            after_id: recoveryCursor, limit: 100,
            created_after: sessionStartRef.current,
          }});
          if (disposed) break;
          const items = response.data?.alerts || [];
          for (const raw of items) {
            recoveryCursor = Math.max(recoveryCursor, Number(raw.alert_id || raw.id) || 0);
            const accepted = acceptAlert(raw);
            if (accepted) mergeAlert(accepted);
          }
          if (items.length < 100) break;
        }
      } catch {
        // Retry on the next tick; WebSocket delivery remains active.
      } finally { recovering = false; }
    };
    const recoveryTimer = window.setInterval(recoverSessionAlerts, 5000);
    recoverSessionAlerts();

    const clearInitialConnectTimer =
      () => {
        if (initialConnectTimerRef.current) {
          window.clearTimeout(initialConnectTimerRef.current);
          initialConnectTimerRef.current = null;
        }
      };

    const clearReconnectTimer =
      () => {
        if (
          reconnectTimerRef.current
        ) {
          clearTimeout(
            reconnectTimerRef.current
          );

          reconnectTimerRef.current =
            null;
        }
      };

    const clearHeartbeatTimer =
      () => {
        if (
          heartbeatTimerRef.current
        ) {
          clearInterval(
            heartbeatTimerRef.current
          );

          heartbeatTimerRef.current =
            null;
        }
      };

    const closeSocket =
      () => {
        clearHeartbeatTimer();

        const socket = wsRef.current;

        if (!socket) {
          return;
        }

        // Detach this socket from the component first so late events from a
        // previous effect instance can never affect the current connection.
        if (wsRef.current === socket) {
          wsRef.current = null;
        }

        /*
         * Calling close() while readyState === CONNECTING makes Chromium log:
         * "WebSocket is closed before the connection is established."
         *
         * That is especially common under React StrictMode. Instead, detach
         * all handlers and close the socket immediately after it reaches OPEN.
         */
        if (socket.readyState === WebSocket.CONNECTING) {
          socket.onmessage = null;
          socket.onerror = null;
          socket.onclose = null;
          socket.onopen = () => {
            socket.onopen = null;

            try {
              socket.close(1000, "client_cleanup");
            } catch {
              // Best-effort cleanup only.
            }
          };

          return;
        }

        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.onopen = null;

        if (socket.readyState === WebSocket.OPEN) {
          try {
            socket.close(1000, "client_cleanup");
          } catch {
            // Best-effort cleanup only.
          }
        }
      };

    const handleAuthFailure =
      () => {
        intentionallyClosedRef.current =
          true;

        closeSocket();

        /*
         * Do not display or log authentication
         * tokens.
         */
        try {
          localStorage.removeItem(
            "user"
          );

          localStorage.removeItem(
            "csrf_token"
          );
        } catch {
          // Ignore storage failures.
        }

        window.location.replace(
          "/"
        );
      };

    const scheduleReconnect =
      () => {
        if (
          !mountedRef.current ||
          intentionallyClosedRef.current
        ) {
          return;
        }

        const existingState = wsRef.current?.readyState;

        if (
          existingState === WebSocket.OPEN ||
          existingState === WebSocket.CONNECTING
        ) {
          return;
        }

        reconnectAttemptsRef.current +=
          1;

        const exponential = Math.min(
          BASE_RECONNECT_DELAY_MS * 2 ** Math.min(reconnectAttemptsRef.current - 1, 6),
          MAX_RECONNECT_DELAY_MS
        );
        const delay = Math.round(exponential * (0.8 + Math.random() * 0.4));

        clearReconnectTimer();

        reconnectTimerRef.current =
          window.setTimeout(
            () => {
              connectWebSocket();
            },
            delay
          );
      };

    const startHeartbeat =
      () => {
        clearHeartbeatTimer();

        lastHeartbeatRef.current =
          Date.now();

        heartbeatTimerRef.current =
          window.setInterval(
            () => {
              const socket =
                wsRef.current;

              if (
                !socket ||
                socket.readyState !==
                  WebSocket.OPEN
              ) {
                return;
              }

              if (
                Date.now() -
                  lastHeartbeatRef.current >
                HEARTBEAT_TIMEOUT_MS
              ) {
                try {
                  socket.close();
                } catch {
                  // Ignore.
                }
                return;
              }

              try {
                socket.send("ping");
              } catch {
                // onclose will schedule reconnection.
              }
            },
            HEARTBEAT_CHECK_MS
          );
      };

    const connectWebSocket =
      () => {
        clearReconnectTimer();

        const wsURL = WEBSOCKET_URL;

        if (!wsURL) {
          console.warn(
            "WebSocket URL is not configured"
          );

          return;
        }

        let parsedURL;

        try {
          parsedURL =
            new URL(
              wsURL,
              window.location.origin
            );
        } catch {
          console.error(
            "Invalid WebSocket URL configuration"
          );

          return;
        }

        if (
          parsedURL.protocol !==
            "ws:" &&
          parsedURL.protocol !==
            "wss:"
        ) {
          console.error(
            "WebSocket URL must use ws:// or wss://"
          );

          return;
        }

        if (
          !mountedRef.current ||
          intentionallyClosedRef.current ||
          document.hidden
        ) {
          return;
        }

        const existingSocket = wsRef.current;

        if (
          existingSocket?.readyState === WebSocket.OPEN ||
          existingSocket?.readyState === WebSocket.CONNECTING
        ) {
          return;
        }

        if (existingSocket?.readyState === WebSocket.CLOSING) {
          scheduleReconnect();
          return;
        }

        if (existingSocket?.readyState === WebSocket.CLOSED) {
          wsRef.current = null;
        }

        let socket;

        try {
          socket =
            new WebSocket(
              parsedURL.toString()
            );
        } catch {
          scheduleReconnect();

          return;
        }

        wsRef.current =
          socket;

        socket.onopen = () => {
          if (
            !mountedRef.current ||
            intentionallyClosedRef.current ||
            wsRef.current !== socket
          ) {
            try {
              socket.close(1000, "stale_connection");
            } catch {
              // Best-effort stale socket cleanup.
            }
            return;
          }

          reconnectAttemptsRef.current =
            0;

          lastHeartbeatRef.current =
            Date.now();

          recoverSessionAlerts();
          startHeartbeat();
        };

        socket.onmessage = (
          event
        ) => {
          if (
            typeof event.data !==
            "string"
          ) {
            return;
          }

          let data;

          try {
            data =
              JSON.parse(
                event.data
              );
          } catch {
            return;
          }

          if (
            !data ||
            typeof data !==
              "object"
          ) {
            return;
          }

          if (
            data.type === "ping" ||
            data.type === "heartbeat"
          ) {
            lastHeartbeatRef.current =
              Date.now();

            if (
              socket.readyState ===
              WebSocket.OPEN
            ) {
              try {
                socket.send(
                  JSON.stringify({
                    type: "pong",
                  })
                );
              } catch {
                // Ignore.
              }
            }

            return;
          }

          if (data.type === "pong") {
            lastHeartbeatRef.current =
              Date.now();
            return;
          }

          // The backend may wrap an alert inside an event envelope.
          const candidateAlert =
            data.type === "alert" ||
            data.type === "watchlist_alert" ||
            data.event === "ALERT_CREATED" ||
            data.event === "WATCHLIST_MATCH"
              ? data.alert || data.payload || data
              : data;

          const normalizedAlert =
            acceptAlert(
              candidateAlert
            );
          if (!normalizedAlert) {
            return;
          }

          // Any valid server event proves the socket is alive.
          lastHeartbeatRef.current = Date.now();

          setAlerts((previous) => {
            const newKey = buildAlertKey(normalizedAlert);

            if (previous.some((item) => buildAlertKey(item) === newKey)) {
              return previous.map((item) => buildAlertKey(item) === newKey
                ? { ...item, ...normalizedAlert } : item);
            }

            return [normalizedAlert, ...previous].slice(0, MAX_ALERTS);
          });
        };

        socket.onerror = () => {
          /*
           * onclose handles reconnect.
           */
        };

        socket.onclose = (
          event
        ) => {
          const wasCurrent = wsRef.current === socket;

          if (wasCurrent) {
            wsRef.current = null;
            clearHeartbeatTimer();
          }

          if (
            !wasCurrent ||
            !mountedRef.current ||
            intentionallyClosedRef.current
          ) {
            return;
          }

          /*
           * 1008 is commonly used for
           * authentication/policy rejection.
           */
          if (
            event.code ===
            1008
          ) {
            handleAuthFailure();
            return;
          }

          scheduleReconnect();
        };
      };

    const handleVisibilityChange =
      () => {
        if (
          !mountedRef.current
        ) {
          return;
        }

        if (document.hidden) return;

        intentionallyClosedRef.current = false;
        const state = wsRef.current?.readyState;
        if (state !== WebSocket.OPEN && state !== WebSocket.CONNECTING) {
          reconnectAttemptsRef.current = 0;
          connectWebSocket();
        }
      };

    const handleOnline = () => {
      if (!mountedRef.current || intentionallyClosedRef.current) return;
      reconnectAttemptsRef.current = 0;
      connectWebSocket();
    };

    // Delay the initial connection so React StrictMode's development-only
    // probe mount can clean up without ever creating a CONNECTING socket.
    clearInitialConnectTimer();
    initialConnectTimerRef.current = window.setTimeout(() => {
      initialConnectTimerRef.current = null;

      if (
        mountedRef.current &&
        !intentionallyClosedRef.current &&
        !document.hidden
      ) {
        connectWebSocket();
      }
    }, INITIAL_CONNECT_DELAY_MS);

    document.addEventListener(
      "visibilitychange",
      handleVisibilityChange
    );
    window.addEventListener("online", handleOnline);

    return () => {
      disposed = true;
      window.clearInterval(recoveryTimer);
      mountedRef.current =
        false;

      intentionallyClosedRef.current =
        true;

      document.removeEventListener(
        "visibilitychange",
        handleVisibilityChange
      );
      window.removeEventListener("online", handleOnline);

      clearInitialConnectTimer();
      clearReconnectTimer();
      clearHeartbeatTimer();
      closeSocket();
    };
  }, [
    filterCamID,
    allowedTypesKey,
    normalizedSourceMode,
    uploadVideoMapKey,
  ]);

  // ==========================================================
  // PAGE-SCOPED REAL-TIME ALERTS
  // ==========================================================

  // Alerts have already been source-filtered before insertion.
  // No dashboard/upload tabs and no cross-source mixing.
  const visibleAlerts = alerts;
  const visibleEmptyMessage = panelEmptyMessage;

  // ==========================================================
  // GIS
  // ==========================================================

  const handleViewOnMap =
    (alert) => {
      if (
        !alert?.cam_id
      ) {
        return;
      }

      if (
        typeof onAlertSelect ===
        "function"
      ) {
        onAlertSelect(
          alert
        );
      }
    };

  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <section
      className="flex h-full min-h-0 w-full flex-col overflow-hidden"
      aria-label={panelTitle}
    >
      {/* HEADER */}

      <div className="mb-4 flex shrink-0 flex-col gap-3">
        <div className="flex items-center justify-between gap-3">
          <h2 className="text-2xl font-bold text-blue-400">
            {panelTitle}
          </h2>

          {visibleAlerts.length > 0 && (
            <span className="rounded-full bg-[#0d2038] border border-[#183b63] px-3 py-1 text-xs font-bold text-[#86bde8]">
              {visibleAlerts.length} current
            </span>
          )}
        </div>

      </div>

      {/* EMPTY */}

      {visibleAlerts.length ===
      0 ? (
        <div className="flex min-h-0 flex-1 flex-col items-center justify-center gap-2 overflow-hidden p-5 text-gray-400">
          <SearchX
            className="text-white"
            size={120}
          />

          <p className="text-xl text-white font-bold">
            {visibleEmptyMessage}
          </p>
        </div>
      ) : (
        <div
          className="min-h-0 flex-1 touch-pan-y space-y-4 overflow-x-hidden overflow-y-auto overscroll-y-contain pb-4 pr-2 [overflow-anchor:auto] [scrollbar-gutter:stable]"
          role="feed"
          aria-label={`${panelTitle} feed`}
          aria-live="polite"
          aria-relevant="additions"
          tabIndex={0}
        >
          {visibleAlerts.map(
            (
              alert,
              index
            ) => {
              const level =
                getAlertLevel(
                  alert?.severity ||
                    alert?.alert_level ||
                    alert?.level ||
                    alert?.alert_type
                );

              const style =
                LEVEL_STYLES[
                  level
                ] ||
                DEFAULT_LEVEL_STYLE;

              const uploadVideoInfo =
                uploadVideoMap?.[
                  alert?.cam_id
                ];

              const snapshotPath =
                getSnapshotPath(
                  alert
                );

              const isUploadAlert =
                isUploadAlertPayload(
                  alert,
                  uploadVideoMap
                );

              const canViewOnMap =
                !isUploadAlert &&
                Boolean(alert?.cam_id || alert?.camera_id);

              return (
                <div
                  key={`${buildAlertKey(
                    alert
                  )}_${index}`}
                  className={`relative w-full [overflow-anchor:auto] bg-[#0d1828] border-l-4 ${style.border} rounded-sm p-4 text-white hover:bg-[#101f34] transition-all duration-200`}
                >
                  {/* HEADER */}

                  <div className="flex items-start justify-between gap-3 mb-4">
                    <div className="flex items-start gap-3 min-w-0">
                      <div className="w-10 h-10 rounded-full border border-slate-600 bg-[#081321] flex items-center justify-center shrink-0">
                        <AlertTriangle
                          size={21}
                          className={
                            style.icon
                          }
                        />
                      </div>

                      <div className="min-w-0">
                        {uploadVideoInfo && (
                          <p className="text-cyan-300 text-sm font-bold">
                            Video{" "}
                            {
                              uploadVideoInfo.videoNumber
                            }
                          </p>
                        )}

                        <h3 className="text-lg font-bold leading-snug break-words">
                          {alert?.rule ||
                            alert?.alert_rule ||
                            "Unknown Alert"}
                        </h3>

                        <p className="text-xs text-slate-400 mt-1">
                          Track:{" "}
                          {alert?.track_id ||
                            "N/A"}
                        </p>

                        {(alert?.plate ||
                          alert?.plate_number ||
                          alert?.license_plate) && (
                          <p className="text-xs text-cyan-300 mt-1 font-mono font-bold">
                            Plate:{" "}
                            {alert?.plate ||
                              alert?.plate_number ||
                              alert?.license_plate}
                          </p>
                        )}

                        {(alert?.watchlist_category ||
                          alert?.watchlist_entry_id) && (
                          <p className="text-xs text-red-300 mt-1 font-semibold">
                            Watchlist:{" "}
                            {alert?.watchlist_category ||
                              "Exact Match"}
                          </p>
                        )}

                        {alert?.event_type && <p className="text-xs text-blue-300 mt-1 font-semibold">Event: {String(alert.event_type).replaceAll("_", " ")}</p>}
                        {alert?.incident_id && <p className="text-xs text-purple-300 mt-1 font-semibold">Incident: #{alert.incident_id}</p>}
                        {(alert?.person_count != null || alert?.vehicle_count != null) && <p className="text-xs text-slate-300 mt-1">People: {alert?.person_count ?? 0} · Vehicles: {alert?.vehicle_count ?? 0}</p>}
                        {(alert?.vehicle_attributes?.color || alert?.vehicle_attributes?.make || alert?.vehicle_attributes?.model) && <p className="text-xs text-emerald-300 mt-1">Vehicle: {[alert?.vehicle_attributes?.color,alert?.vehicle_attributes?.make,alert?.vehicle_attributes?.model].filter(Boolean).join(" · ")}</p>}
                      </div>
                    </div>

                    <span
                      className={`px-3 py-1 rounded-full text-xs font-bold uppercase shrink-0 ${style.badge}`}
                    >
                      {level}
                    </span>
                  </div>

                  {/* SNAPSHOT */}

                  <SnapshotImage
                    snapshotPath={
                      snapshotPath
                    }
                  />

                  {/* DETAILS */}

                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
                    {uploadVideoInfo && (
                      <div className="bg-[#090f1b] rounded-lg px-3 py-2 flex items-center gap-2 min-w-0">
                        <Camera
                          size={15}
                          className="text-cyan-300 shrink-0"
                        />

                        <span className="truncate font-bold text-cyan-300">
                          Video{" "}
                          {
                            uploadVideoInfo.videoNumber
                          }
                        </span>
                      </div>
                    )}

                    <div className="bg-[#090f1b] rounded-lg px-3 py-2 flex items-center gap-2 min-w-0">
                      <Camera
                        size={15}
                        className="text-blue-300 shrink-0"
                      />

                      <span className="truncate">
                        {alert?.cam_id ||
                          alert?.camera_id ||
                          "N/A"}
                      </span>
                    </div>

                    <div className="bg-[#090f1b] rounded-lg px-3 py-2 flex items-center gap-2 min-w-0">
                      <Activity
                        size={15}
                        className="text-green-300 shrink-0"
                      />

                      <span className="truncate">
                        {alert?.source_type ||
                          alert?.source ||
                          "N/A"}
                      </span>
                    </div>

                    <div className="bg-[#090f1b] rounded-lg px-3 py-2 flex items-center gap-2 min-w-0">
                      <Clock
                        size={15}
                        className="text-yellow-300 shrink-0"
                      />

                      <span className="truncate">
                        {formatDateTime(
                          alert?.created_at ||
                            alert?.timestamp
                        )}
                      </span>
                    </div>

                    {alert?.zone && (
                      <div className="bg-[#090f1b] rounded-lg px-3 py-2 flex items-center gap-2 min-w-0 sm:col-span-2">
                        <MapPin
                          size={15}
                          className="text-purple-300 shrink-0"
                        />

                        <span className="truncate">
                          {alert.zone}
                        </span>
                      </div>
                    )}

                    {/* GIS BUTTON — upload-video alerts intentionally have
                        no View on Map action because they are not tied to a
                        live camera GIS position. */}
                    {canViewOnMap && (
                      <button
                        type="button"
                        onClick={() => handleViewOnMap(alert)}
                        className="sm:col-span-2 bg-blue-600 hover:bg-blue-700 rounded-lg px-3 py-2 flex items-center justify-center gap-2 font-bold transition"
                      >
                        <MapPin size={15} />
                        View on Map
                      </button>
                    )}
                  </div>
                </div>
              );
            }
          )}
        </div>
      )}
    </section>
  );
};

export default Alert;
