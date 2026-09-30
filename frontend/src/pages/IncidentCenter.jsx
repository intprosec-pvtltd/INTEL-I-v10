import React, { useEffect, useMemo, useState } from "react";

import toast from "react-hot-toast";
import { formatIndianDateTime as formatDate } from "../utils/dateTime";

import {
  ClipboardList,
  RefreshCw,
  CheckCircle2,
  ShieldAlert,
  Search,
  Eye,
  X,
  Clock3,
  Camera,
  Car,
  UserRound,
  FileText,
  AlertTriangle,
  Activity,
  MapPin,
  Upload,
  Radio,
  Fingerprint,
  Download,
  ShieldCheck,
} from "lucide-react";

import api, {
  getIncidents,
  updateIncident,
  downloadIncidentEvidenceReport,
} from "../api/axios.js";
import WatchlistJourneyEvidence from "../components/WatchlistJourneyEvidence.jsx";

/* ============================================================
 * HELPERS
 * ============================================================ */

const getErrorMessage = (error, fallback = "Request failed") => {
  const detail = error?.response?.data?.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }

        if (item && typeof item.msg === "string") {
          const location = Array.isArray(item.loc) ? item.loc.join(".") : "";

          return location ? `${location}: ${item.msg}` : item.msg;
        }

        return "Validation error";
      })
      .join(", ");
  }

  if (detail && typeof detail === "object" && typeof detail.msg === "string") {
    return detail.msg;
  }

  return error?.message || fallback;
};

const getSeverityClass = (severity) => {
  const value = String(severity || "INFO").toUpperCase();

  if (value === "CRITICAL") {
    return {
      label: "CRITICAL",
      className: "border-red-500/40 bg-red-500/10 text-red-300",
      dot: "bg-red-400",
    };
  }

  if (value === "HIGH") {
    return {
      label: "HIGH",
      className: "border-orange-500/40 bg-orange-500/10 text-orange-300",
      dot: "bg-orange-400",
    };
  }

  if (value === "MEDIUM") {
    return {
      label: "MEDIUM",
      className: "border-amber-500/40 bg-amber-500/10 text-amber-300",
      dot: "bg-amber-400",
    };
  }

  if (value === "LOW") {
    return {
      label: "LOW",
      className: "border-blue-500/40 bg-blue-500/10 text-blue-300",
      dot: "bg-blue-400",
    };
  }

  return {
    label: "INFO",
    className: "border-slate-700 bg-slate-900 text-slate-400",
    dot: "bg-slate-500",
  };
};

const getStatusClass = (status) => {
  const value = String(status || "OPEN").toUpperCase();

  if (value === "RESOLVED" || value === "CLOSED") {
    return "border-emerald-500/30 bg-emerald-500/10 text-emerald-300";
  }

  if (value === "UNDER_REVIEW") {
    return "border-cyan-500/30 bg-cyan-500/10 text-cyan-300";
  }

  if (value === "ESCALATED") {
    return "border-red-500/30 bg-red-500/10 text-red-300";
  }

  return "border-slate-700 bg-slate-900 text-slate-300";
};

const getIncidentDomain = (incident) => {
  const type = String(incident?.incident_type || "").toUpperCase();

  if (type.includes("VEHICLE") || type.includes("ANPR")) {
    return "VEHICLE";
  }

  if (type.includes("PERSON") || type.includes("FACE")) {
    return "PERSON";
  }

  if (
    type.includes("CROWD") ||
    type.includes("TRAFFIC") ||
    type.includes("BEHAV")
  ) {
    return "BEHAVIOR";
  }

  return "GENERAL";
};

const getDomainMeta = (domain) => {
  switch (domain) {
    case "VEHICLE":
      return {
        label: "VEHICLE",
        icon: Car,
        className: "border-blue-500/30 bg-blue-500/10 text-blue-300",
      };

    case "PERSON":
      return {
        label: "PERSON",
        icon: UserRound,
        className: "border-purple-500/30 bg-purple-500/10 text-purple-300",
      };

    case "BEHAVIOR":
      return {
        label: "BEHAVIOR",
        icon: Activity,
        className: "border-amber-500/30 bg-amber-500/10 text-amber-300",
      };

    default:
      return {
        label: "GENERAL",
        icon: ClipboardList,
        className: "border-slate-700 bg-slate-900 text-slate-300",
      };
  }
};

/**
 * Evidence object_type from older/backend records can be generic or wrong
 * (for example "VEHICLE" on a behavioral alert). The incident domain is
 * authoritative for display unless this is an actual vehicle/person incident.
 */
const getEvidenceDisplayType = (incident, item) => {
  const domain = getIncidentDomain(incident);
  const rawType = String(item?.object_type || "")
    .trim()
    .toUpperCase();

  if (domain === "VEHICLE") return "VEHICLE";
  if (domain === "PERSON") return "PERSON";
  if (domain === "BEHAVIOR") return "BEHAVIOR";

  // GENERAL incidents must never be relabeled VEHICLE merely because an
  // evidence row contains a stale/default object_type. Preserve useful
  // non-vehicle evidence labels; otherwise show the incident domain.
  if (rawType && rawType !== "VEHICLE" && rawType !== "OBJECT") {
    return rawType;
  }

  return "GENERAL";
};

/**
 * The evidence API can contain large metadata objects and internal camera IDs.
 * Keep those implementation details out of the incident UI. Operators only
 * need a clear source: a live-camera location or an uploaded-camera number.
 */
const firstTextValue = (...values) => {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }

  return "";
};

const isJsonLikeText = (value) => {
  if (typeof value !== "string") {
    return false;
  }

  const text = value.trim();

  if (
    !(
      (text.startsWith("{") && text.endsWith("}")) ||
      (text.startsWith("[") && text.endsWith("]"))
    )
  ) {
    return false;
  }

  try {
    JSON.parse(text);
    return true;
  } catch {
    return false;
  }
};

const getSafeEvidenceDescription = (value) => {
  if (typeof value !== "string" || !value.trim() || isJsonLikeText(value)) {
    return "";
  }

  return value.trim();
};

const getEvidenceCameraMeta = (incident, item, index = 0) => {
  const metadata =
    item?.metadata &&
    typeof item.metadata === "object" &&
    !Array.isArray(item.metadata)
      ? item.metadata
      : {};

  const camera =
    item?.camera &&
    typeof item.camera === "object" &&
    !Array.isArray(item.camera)
      ? item.camera
      : {};

  const location = firstTextValue(
    item?.location_name,
    item?.camera_location,
    item?.location,
    camera.location_name,
    camera.camera_location,
    camera.location,
    metadata.location_name,
    metadata.camera_location,
    metadata.location,
    metadata.address,
    incident?.location_name,
    incident?.camera_location,
    incident?.location,
  );

  const sourceType = firstTextValue(
    item?.source_type,
    item?.camera_type,
    camera.source_type,
    camera.camera_type,
    metadata.source_type,
    metadata.camera_type,
    metadata.source,
    incident?.source_type,
    incident?.camera_type,
  ).toLowerCase();

  const cameraId = firstTextValue(
    item?.cam_id,
    item?.camera_id,
    metadata.cam_id,
    metadata.camera_id,
    incident?.cam_id,
    incident?.camera_id,
  );

  const cameraName = firstTextValue(
    item?.camera_name,
    item?.name,
    camera.name,
    metadata.camera_name,
  );

  const isUpload =
    sourceType.includes("upload") ||
    sourceType.includes("file") ||
    sourceType.includes("video") ||
    /^upload(?:ed)?[_\s-]/i.test(cameraId) ||
    /^upload camera\s+\d+$/i.test(cameraName) ||
    metadata.is_upload === true ||
    metadata.uploaded === true;

  if (isUpload) {
    const friendlyName = /^upload camera\s+\d+$/i.test(cameraName)
      ? cameraName.replace(/\s+/g, " ")
      : "";
    const numberFromId = cameraId.match(/^upload(?:ed)?[_\s-]*(\d+)/i)?.[1];

    return {
      label: friendlyName || `Upload Camera ${numberFromId || index + 1}`,
      helper: "Uploaded video source",
      icon: Upload,
      className: "border-violet-500/20 bg-violet-500/[0.06] text-violet-200",
    };
  }

  const safeLiveName =
    cameraName &&
    cameraName !== cameraId &&
    !/^(?:cam|camera)[_\s-]*\d/i.test(cameraName)
      ? cameraName
      : "";

  return {
    label: location || safeLiveName || "Live Camera",
    helper: location ? "Live camera location" : "Location unavailable",
    icon: location ? MapPin : Radio,
    className: "border-cyan-500/20 bg-cyan-500/[0.06] text-cyan-200",
  };
};

const getIncidentCameraLabel = (incident) => {
  const evidence = Array.isArray(incident?.evidence) ? incident.evidence : [];

  return getEvidenceCameraMeta(incident, evidence[0] || {}, 0).label;
};

const isClosedStatus = (status) => {
  const value = String(status || "").toUpperCase();

  return value === "RESOLVED" || value === "CLOSED";
};

/* ============================================================
 * SNAPSHOT VALIDATION
 * ============================================================ */

/**
 * Validate that the backend response actually contains
 * image data instead of JSON/HTML/text.
 */
const validateSnapshotBlob = async (blob, contentType = "") => {
  if (!(blob instanceof Blob)) {
    throw new Error("Snapshot endpoint did not return binary data.");
  }

  if (blob.size <= 0) {
    throw new Error("Snapshot response is empty.");
  }

  const normalizedType = String(contentType || blob.type || "").toLowerCase();

  /*
   * Never allow common API error payloads to be
   * converted into image Blob URLs.
   */
  if (
    normalizedType.includes("application/json") ||
    normalizedType.includes("text/html") ||
    normalizedType.includes("text/plain")
  ) {
    let message = "Snapshot endpoint returned a non-image response.";

    try {
      const text = await blob.text();

      if (text) {
        try {
          const parsed = JSON.parse(text);

          message = parsed?.detail || parsed?.message || message;
        } catch {
          message = text.slice(0, 500) || message;
        }
      }
    } catch {
      // Keep default error.
    }

    throw new Error(message);
  }

  /*
   * If server explicitly declares image/*,
   * accept it.
   */
  if (normalizedType.startsWith("image/")) {
    return blob;
  }

  /*
   * Some servers return application/octet-stream.
   * Inspect the binary signature before accepting it.
   */
  const header = new Uint8Array(await blob.slice(0, 16).arrayBuffer());

  const isJpeg =
    header.length >= 3 &&
    header[0] === 0xff &&
    header[1] === 0xd8 &&
    header[2] === 0xff;

  const isPng =
    header.length >= 8 &&
    header[0] === 0x89 &&
    header[1] === 0x50 &&
    header[2] === 0x4e &&
    header[3] === 0x47 &&
    header[4] === 0x0d &&
    header[5] === 0x0a &&
    header[6] === 0x1a &&
    header[7] === 0x0a;

  const isWebp =
    header.length >= 12 &&
    header[0] === 0x52 &&
    header[1] === 0x49 &&
    header[2] === 0x46 &&
    header[3] === 0x46 &&
    header[8] === 0x57 &&
    header[9] === 0x45 &&
    header[10] === 0x42 &&
    header[11] === 0x50;

  if (isJpeg || isPng || isWebp) {
    let type = blob.type;

    if (!type) {
      if (isJpeg) {
        type = "image/jpeg";
      } else if (isPng) {
        type = "image/png";
      } else if (isWebp) {
        type = "image/webp";
      }
    }

    return type ? new Blob([blob], { type }) : blob;
  }

  throw new Error(
    `Snapshot response is not a valid image. Content-Type: ${
      normalizedType || "unknown"
    }`,
  );
};

/* ============================================================
 * SMALL COMPONENTS
 * ============================================================ */

const StatCard = ({
  title,
  value,
  description,
  icon: Icon,
  className = "text-blue-300",
}) => (
  <div className="rounded-xl border border-slate-800/90 bg-slate-950/60 p-4 shadow-[0_10px_35px_rgba(0,0,0,0.16)]">
    <div className="flex items-start justify-between gap-3">
      <div className="min-w-0">
        <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-500">
          {title}
        </div>

        <div className="mt-2 text-2xl font-bold text-white">{value}</div>
      </div>

      <div
        className={`shrink-0 rounded-xl border border-white/5 bg-white/[0.03] p-2.5 ${className}`}
      >
        <Icon size={18} />
      </div>
    </div>

    <div className="mt-2 text-xs text-slate-600">{description}</div>
  </div>
);

const Badge = ({ children, className }) => (
  <span
    className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-[10px] font-semibold tracking-wide ${className}`}
  >
    {children}
  </span>
);

/* ============================================================
 * COMPONENT
 * ============================================================ */

const IncidentCenter = () => {
  const [incidents, setIncidents] = useState([]);

  const [loading, setLoading] = useState(true);

  const [refreshing, setRefreshing] = useState(false);

  const [resolvingId, setResolvingId] = useState(null);

  const [selectedIncident, setSelectedIncident] = useState(null);

  const [search, setSearch] = useState("");

  const [severityFilter, setSeverityFilter] = useState("ALL");

  const [statusFilter, setStatusFilter] = useState("ALL");

  const [domainFilter, setDomainFilter] = useState("ALL");

  const [snapshotUrl, setSnapshotUrl] = useState(null);

  const [snapshotLoading, setSnapshotLoading] = useState(false);

  const [snapshotError, setSnapshotError] = useState("");

  const [openingSnapshotId, setOpeningSnapshotId] = useState(null);

  const [exportingEvidenceId, setExportingEvidenceId] = useState(null);

  /* ==========================================================
   * LOAD INCIDENTS
   * ========================================================== */

  const load = async ({ silent = false } = {}) => {
    if (silent) {
      setRefreshing(true);
    } else {
      setLoading(true);
    }

    try {
      const data = await getIncidents({
        limit: 200,
      });

      const backendIncidents = Array.isArray(data?.incidents)
        ? data.incidents
        : [];

      setIncidents(backendIncidents);
    } catch (error) {
      setIncidents([]);

      toast.error(getErrorMessage(error, "Failed to load incidents"));
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  /* ==========================================================
   * DEEP LINK FROM INTELLIGENCE ASSISTANT
   * ========================================================== */

  useEffect(() => {
    if (!incidents.length || selectedIncident) return;
    const id = new URLSearchParams(window.location.search).get("incident_id");
    if (!id) return;
    const match = incidents.find((item) => String(item.id) === String(id));
    if (match) setSelectedIncident(match);
  }, [incidents, selectedIncident]);

  const exportEvidencePdf = async (incident) => {
    const incidentId = incident?.id;

    if (incidentId === null || incidentId === undefined || incidentId === "") {
      toast.error("Incident ID is required for evidence export");
      return;
    }

    setExportingEvidenceId(incidentId);

    try {
      const response = await downloadIncidentEvidenceReport(
        incidentId,
        "RESTRICTED"
      );

      const blob = response?.data instanceof Blob
        ? response.data
        : new Blob([response?.data], { type: "application/pdf" });

      if (!blob.size) {
        throw new Error("The evidence PDF response was empty.");
      }

      const contentType = String(response?.headers?.["content-type"] || "").toLowerCase();
      if (contentType && !contentType.includes("application/pdf")) {
        throw new Error("The server returned an unexpected evidence export format.");
      }

      const disposition = String(response?.headers?.["content-disposition"] || "");
      const match = disposition.match(/filename="?([^";]+)"?/i);
      const safeFilename = (match?.[1] || `intel-i-evidence-incident-${incidentId}.pdf`)
        .replace(/[^a-zA-Z0-9._-]/g, "_")
        .slice(0, 180);

      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = safeFilename;
      anchor.rel = "noopener";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);

      const reportId = response?.headers?.["x-report-id"];
      toast.success(reportId ? `Evidence PDF generated: ${reportId}` : "Evidence PDF generated");
    } catch (error) {
      toast.error(getErrorMessage(error, "Unable to generate evidence PDF"));
    } finally {
      setExportingEvidenceId(null);
    }
  };

  /* ==========================================================
   * STATS
   * ========================================================== */

  const stats = useMemo(() => {
    const active = incidents.filter(
      (item) => !isClosedStatus(item.status),
    ).length;

    const critical = incidents.filter(
      (item) =>
        String(
          item.severity || item.evaluation_severity || "INFO",
        ).toUpperCase() === "CRITICAL",
    ).length;

    const high = incidents.filter(
      (item) =>
        String(
          item.severity || item.evaluation_severity || "INFO",
        ).toUpperCase() === "HIGH",
    ).length;

    const vehicle = incidents.filter(
      (item) => getIncidentDomain(item) === "VEHICLE",
    ).length;

    const person = incidents.filter(
      (item) => getIncidentDomain(item) === "PERSON",
    ).length;

    return {
      total: incidents.length,
      active,
      critical,
      high,
      vehicle,
      person,
    };
  }, [incidents]);

  /* ==========================================================
   * FILTERED INCIDENTS
   * ========================================================== */

  const filteredIncidents = useMemo(() => {
    const query = search.trim().toLowerCase();

    return incidents.filter((incident) => {
      const severity = String(
        incident.severity || incident.evaluation_severity || "INFO",
      ).toUpperCase();

      const status = String(incident.status || "OPEN").toUpperCase();

      const domain = getIncidentDomain(incident);

      const searchable = [
        incident.id,
        incident.incident_type,
        incident.cam_id,
        incident.primary_track_id,
        incident.description,
        incident.evidence_text,

        ...(Array.isArray(incident.evidence)
          ? incident.evidence.map(
              (item) => item.object_reference || item.description || "",
            )
          : []),
      ]
        .join(" ")
        .toLowerCase();

      const matchesSearch = !query || searchable.includes(query);

      const matchesSeverity =
        severityFilter === "ALL" || severity === severityFilter;

      const matchesStatus = statusFilter === "ALL" || status === statusFilter;

      const matchesDomain = domainFilter === "ALL" || domain === domainFilter;

      return matchesSearch && matchesSeverity && matchesStatus && matchesDomain;
    });
  }, [incidents, search, severityFilter, statusFilter, domainFilter]);

  /* ==========================================================
   * RESOLVE INCIDENT
   * ========================================================== */

  const resolve = async (incident) => {
    const id = incident?.id;

    if (!id) {
      return;
    }

    setResolvingId(id);

    try {
      await updateIncident(id, {
        status: "RESOLVED",
        evaluation_status: "REVIEWED",
      });

      setIncidents((previous) =>
        previous.map((item) =>
          item.id === id
            ? {
                ...item,
                status: "RESOLVED",
                evaluation_status: "REVIEWED",
              }
            : item,
        ),
      );

      setSelectedIncident((previous) =>
        previous && previous.id === id
          ? {
              ...previous,
              status: "RESOLVED",
              evaluation_status: "REVIEWED",
            }
          : previous,
      );

      toast.success("Incident resolved");
    } catch (error) {
      toast.error(getErrorMessage(error, "Failed to resolve incident"));
    } finally {
      setResolvingId(null);
    }
  };

  /* ==========================================================
   * INCIDENT MODAL
   * ========================================================== */

  const openIncident = (incident) => {
    setSelectedIncident(incident);
  };

  const closeIncident = () => {
    setSelectedIncident(null);
  };

  /* ==========================================================
   * FILTER HELPERS
   * ========================================================== */

  const hasFilters =
    Boolean(search) ||
    severityFilter !== "ALL" ||
    statusFilter !== "ALL" ||
    domainFilter !== "ALL";

  const clearFilters = () => {
    setSearch("");

    setSeverityFilter("ALL");

    setStatusFilter("ALL");

    setDomainFilter("ALL");
  };

  /* ==========================================================
   * FETCH SNAPSHOT
   * ========================================================== */

  const fetchSnapshotBlob = async (snapshotId) => {
    if (snapshotId === null || snapshotId === undefined || snapshotId === "") {
      throw new Error("Snapshot ID is missing.");
    }

    console.log("[INTEL-I][SNAPSHOT] Requesting snapshot", snapshotId);

    const response = await api.get(
      `/snapshot/${encodeURIComponent(snapshotId)}`,
      {
        responseType: "blob",

        timeout: 30000,

        validateStatus: (status) => status >= 200 && status < 300,
      },
    );

    const contentType = String(
      response?.headers?.["content-type"] ||
        response?.headers?.["Content-Type"] ||
        response?.data?.type ||
        "",
    ).toLowerCase();

    console.log("[INTEL-I][SNAPSHOT] Response", {
      snapshotId,
      status: response?.status,
      contentType,
      blobType: response?.data?.type,
      blobSize: response?.data?.size,
    });

    const blob = await validateSnapshotBlob(response.data, contentType);

    return blob;
  };

  /* ==========================================================
   * CREATE SNAPSHOT OBJECT URL
   * ========================================================== */

  const fetchSnapshotObjectUrl = async (snapshotId) => {
    const blob = await fetchSnapshotBlob(snapshotId);

    const objectUrl = URL.createObjectURL(blob);

    console.log("[INTEL-I][SNAPSHOT] Blob URL created", {
      snapshotId,
      objectUrl,
      size: blob.size,
      type: blob.type,
    });

    return objectUrl;
  };

  /* ==========================================================
   * LOAD PRIMARY INCIDENT SNAPSHOT
   * ========================================================== */

  useEffect(() => {
    let cancelled = false;

    let objectUrl = null;

    const loadSnapshot = async () => {
      const snapshotId = selectedIncident?.snapshot_id;

      if (
        snapshotId === null ||
        snapshotId === undefined ||
        snapshotId === ""
      ) {
        setSnapshotUrl(null);

        setSnapshotError("");

        setSnapshotLoading(false);

        return;
      }

      setSnapshotLoading(true);

      setSnapshotError("");

      setSnapshotUrl(null);

      try {
        objectUrl = await fetchSnapshotObjectUrl(snapshotId);

        if (cancelled) {
          if (objectUrl) {
            URL.revokeObjectURL(objectUrl);
          }

          return;
        }

        setSnapshotUrl(objectUrl);
      } catch (error) {
        console.error("[INTEL-I][SNAPSHOT] Failed to load snapshot", {
          snapshotId,
          error,
        });

        if (!cancelled) {
          setSnapshotUrl(null);

          setSnapshotError(
            getErrorMessage(error, "Failed to load incident snapshot."),
          );
        }
      } finally {
        if (!cancelled) {
          setSnapshotLoading(false);
        }
      }
    };

    loadSnapshot();

    return () => {
      cancelled = true;

      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [selectedIncident?.snapshot_id]);

  /* ==========================================================
   * OPEN EVIDENCE SNAPSHOT
   * ========================================================== */

  const openEvidenceSnapshot = async (snapshotId) => {
    if (snapshotId === null || snapshotId === undefined || snapshotId === "") {
      toast.error("Snapshot ID is missing.");

      return;
    }

    /*
     * Create the tab synchronously inside the click event. If window.open is
     * called after the authenticated request finishes, browsers classify it
     * as an unsolicited popup. Avoid the "noopener" feature here because some
     * browsers intentionally return null for a successfully opened noopener
     * window; sever the opener explicitly instead.
     */
    const opened = window.open("", "_blank");

    if (!opened) {
      toast.error(
        "Browser blocked the snapshot window. Please allow pop-ups for this site.",
      );

      return;
    }

    opened.opener = null;
    opened.document.title = `Loading INTEL-I Snapshot #${snapshotId}`;
    opened.document.body.style.margin = "0";
    opened.document.body.style.background = "#020617";
    opened.document.body.style.color = "#cbd5e1";
    opened.document.body.style.display = "grid";
    opened.document.body.style.placeItems = "center";
    opened.document.body.style.minHeight = "100vh";
    opened.document.body.textContent = "Loading authenticated evidence image…";

    setOpeningSnapshotId(snapshotId);

    try {
      /*
       * Fetch through Axios so the request goes to
       * the FastAPI backend rather than localhost:5173.
       */
      const blob = await fetchSnapshotBlob(snapshotId);

      const objectUrl = URL.createObjectURL(blob);

      opened.document.title = `INTEL-I Snapshot #${snapshotId}`;

      opened.document.body.style.margin = "0";

      opened.document.body.style.background = "#020617";

      opened.document.body.style.display = "block";

      opened.document.body.textContent = "";

      const image = opened.document.createElement("img");

      image.src = objectUrl;

      image.alt = `INTEL-I evidence snapshot ${snapshotId}`;

      image.style.display = "block";

      image.style.width = "100vw";

      image.style.height = "100vh";

      image.style.objectFit = "contain";

      image.style.background = "#020617";

      opened.document.body.appendChild(image);

      /*
       * Keep the object URL alive while the new tab is using it.
       */
      setTimeout(() => {
        try {
          URL.revokeObjectURL(objectUrl);
        } catch {
          // Ignore cleanup errors.
        }
      }, 120000);
    } catch (error) {
      console.error("[INTEL-I][SNAPSHOT] Failed to open evidence snapshot", {
        snapshotId,
        error,
      });

      if (!opened.closed) {
        opened.document.title = `INTEL-I Snapshot #${snapshotId} - Error`;
        opened.document.body.style.display = "grid";
        opened.document.body.style.placeItems = "center";
        opened.document.body.style.padding = "24px";
        opened.document.body.style.color = "#fca5a5";
        opened.document.body.textContent = getErrorMessage(
          error,
          "Failed to load the authenticated evidence snapshot.",
        );
      }

      toast.error(getErrorMessage(error, "Failed to open snapshot."));
    } finally {
      setOpeningSnapshotId(null);
    }
  };

  /* ============================================================
   * RENDER
   * ============================================================ */

  return (
    <div className="min-h-full space-y-6 text-white">
      {/* ======================================================
       * HEADER
       * ====================================================== */}

      <div className="relative overflow-hidden rounded-2xl border border-slate-800/90 bg-gradient-to-br from-[#091426] via-[#07101d] to-[#050b14] px-5 py-6 shadow-[0_20px_60px_rgba(0,0,0,0.25)] sm:px-7">
        <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-blue-500/10 blur-3xl" />

        <div className="pointer-events-none absolute -bottom-24 left-1/3 h-56 w-56 rounded-full bg-red-500/5 blur-3xl" />

        <div className="relative flex flex-col gap-5 xl:flex-row xl:items-center xl:justify-between">
          <div className="min-w-0">
            <div className="mb-3 flex items-center gap-2">
              <div className="rounded-lg border border-blue-400/20 bg-blue-400/10 p-2 text-blue-300">
                <ClipboardList size={19} />
              </div>

              <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-blue-300/80">
                Incident Intelligence
              </span>
            </div>

            <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
              Incidents & Evidence
            </h1>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Review vehicle, person and behavioral incidents, inspect evidence,
              and close confirmed cases.
            </p>
          </div>

          <div className="shrink-0">
            <button
              type="button"
              onClick={() =>
                load({
                  silent: true,
                })
              }
              disabled={loading || refreshing}
              className="inline-flex w-full items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-3.5 py-2.5 text-sm text-slate-300 transition hover:bg-slate-800 disabled:opacity-50 sm:w-auto"
            >
              <RefreshCw
                size={16}
                className={loading || refreshing ? "animate-spin" : ""}
              />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* ======================================================
       * STATS
       * ====================================================== */}

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <StatCard
          title="Total"
          value={stats.total}
          description="Loaded incidents"
          icon={ClipboardList}
          className="text-blue-300"
        />

        <StatCard
          title="Active"
          value={stats.active}
          description="Require attention"
          icon={AlertTriangle}
          className="text-amber-300"
        />

        <StatCard
          title="Critical"
          value={stats.critical}
          description="Critical severity"
          icon={ShieldAlert}
          className="text-red-300"
        />

        <StatCard
          title="High"
          value={stats.high}
          description="High severity"
          icon={Activity}
          className="text-orange-300"
        />

        <StatCard
          title="Vehicle"
          value={stats.vehicle}
          description="Vehicle incidents"
          icon={Car}
          className="text-cyan-300"
        />

        <StatCard
          title="Person"
          value={stats.person}
          description="Person incidents"
          icon={UserRound}
          className="text-purple-300"
        />
      </div>

      {/* ======================================================
       * FILTERS
       * ====================================================== */}

      <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
        <div className="grid grid-cols-1 gap-2 xl:grid-cols-[minmax(0,1fr)_160px_160px_160px_auto]">
          <label className="relative min-w-0">
            <Search
              size={16}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-600"
            />

            <input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search incident, camera, track, reference or description..."
              className="w-full rounded-lg border border-slate-800 bg-[#08111d] py-2.5 pl-9 pr-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/50"
            />
          </label>

          <select
            value={domainFilter}
            onChange={(event) => setDomainFilter(event.target.value)}
            className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/50"
          >
            <option value="ALL">All domains</option>

            <option value="VEHICLE">Vehicle</option>

            <option value="PERSON">Person</option>

            <option value="BEHAVIOR">Behavior</option>

            <option value="GENERAL">General</option>
          </select>

          <select
            value={severityFilter}
            onChange={(event) => setSeverityFilter(event.target.value)}
            className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/50"
          >
            <option value="ALL">All severity</option>

            <option value="CRITICAL">Critical</option>

            <option value="HIGH">High</option>

            <option value="MEDIUM">Medium</option>

            <option value="LOW">Low</option>

            <option value="INFO">Info</option>
          </select>

          <select
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value)}
            className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/50"
          >
            <option value="ALL">All statuses</option>

            <option value="OPEN">Open</option>

            <option value="UNDER_REVIEW">Under review</option>

            <option value="ESCALATED">Escalated</option>

            <option value="RESOLVED">Resolved</option>

            <option value="CLOSED">Closed</option>
          </select>

          {hasFilters ? (
            <button
              type="button"
              onClick={clearFilters}
              className="rounded-lg border border-slate-800 bg-slate-900 px-3 py-2.5 text-xs font-medium text-slate-400 transition hover:bg-slate-800 hover:text-white"
            >
              Clear
            </button>
          ) : (
            <div className="hidden xl:block" />
          )}
        </div>
      </div>

      {/* ======================================================
       * INCIDENT LIST
       * ====================================================== */}

      {loading ? (
        <div className="rounded-2xl border border-slate-800 bg-[#07101b] p-12 text-center">
          <RefreshCw
            size={28}
            className="mx-auto mb-3 animate-spin text-blue-300"
          />

          <div className="text-sm text-slate-400">Loading incidents...</div>

          <div className="mt-1 text-xs text-slate-600">
            Retrieving case-level intelligence and evidence.
          </div>
        </div>
      ) : filteredIncidents.length === 0 ? (
        <div className="rounded-2xl border border-slate-800 bg-[#07101b] p-12 text-center">
          <div className="mx-auto mb-4 w-fit rounded-full border border-slate-800 bg-slate-950 p-4">
            <ClipboardList size={30} className="text-slate-600" />
          </div>

          <div className="text-sm font-semibold text-slate-300">
            No incidents found
          </div>

          <div className="mx-auto mt-1 max-w-md text-xs leading-5 text-slate-600">
            Adjust the filters or wait for the intelligence pipeline to generate
            a new incident.
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
          {filteredIncidents.map((incident) => {
            const evidence = Array.isArray(incident.evidence)
              ? incident.evidence
              : [];

            const severity = getSeverityClass(
              incident.severity || incident.evaluation_severity || "INFO",
            );

            const status = String(incident.status || "OPEN").toUpperCase();

            const domain = getIncidentDomain(incident);

            const domainMeta = getDomainMeta(domain);

            const DomainIcon = domainMeta.icon;

            return (
              <article
                key={incident.id}
                className="rounded-2xl border border-slate-800 bg-[#07101b] p-4 shadow-[0_10px_40px_rgba(0,0,0,0.14)] transition hover:border-slate-700 sm:p-5"
              >
                {/* HEADER */}

                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge className={domainMeta.className}>
                        <DomainIcon size={12} />

                        {domainMeta.label}
                      </Badge>

                      <span className="text-xs text-slate-600">
                        #{incident.id}
                      </span>
                    </div>

                    <button
                      type="button"
                      onClick={() => openIncident(incident)}
                      className="mt-3 block max-w-full text-left text-lg font-semibold text-slate-100 transition hover:text-blue-300"
                    >
                      {incident.incident_type || "GENERAL INCIDENT"}
                    </button>

                    <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-slate-600">
                      <span className="inline-flex items-center gap-1.5 break-all">
                        <Camera size={13} />
                        Camera {incident.cam_id || "N/A"}
                      </span>

                      <span>Track {incident.primary_track_id || "N/A"}</span>

                      <span className="inline-flex items-center gap-1.5">
                        <Clock3 size={13} />

                        {formatDate(incident.started_at)}
                      </span>
                    </div>
                  </div>

                  {/* BADGES */}

                  <div className="flex shrink-0 flex-wrap items-center gap-2">
                    <Badge className={severity.className}>
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${severity.dot}`}
                      />

                      {severity.label}
                    </Badge>

                    <Badge className={getStatusClass(status)}>{status}</Badge>
                  </div>
                </div>

                {/* DESCRIPTION */}

                <div className="mt-5 rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="mb-2 flex items-center gap-2">
                    <FileText size={14} className="text-slate-600" />

                    <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                      Incident Description
                    </div>
                  </div>

                  <p className="whitespace-pre-wrap break-words text-sm leading-6 text-slate-400">
                    {incident.description ||
                      incident.evidence_text ||
                      "No description provided."}
                  </p>
                </div>

                {/* EVIDENCE PREVIEW */}

                {evidence.length > 0 && (
                  <div className="mt-4">
                    <div className="mb-2 flex items-center justify-between">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                        Evidence
                      </div>

                      <span className="text-[10px] text-slate-600">
                        {evidence.length} item
                        {evidence.length !== 1 ? "s" : ""}
                      </span>
                    </div>

                    <div className="grid gap-2 md:grid-cols-2">
                      {evidence.slice(0, 4).map((item, index) => (
                        <div
                          key={item.id ?? `${incident.id}-evidence-${index}`}
                          className="rounded-xl border border-slate-800 bg-slate-950/40 p-3"
                        >
                          <div className="flex items-start justify-between gap-2">
                            <div className="min-w-0">
                              <div className="text-xs font-semibold text-slate-300">
                                {item.evidence_type || "OTHER"}
                              </div>

                              <div className="mt-1 text-[11px] text-slate-600">
                                {getEvidenceDisplayType(incident, item)}
                              </div>
                            </div>

                            {item.snapshot_id && (
                              <span className="shrink-0 rounded-md border border-cyan-500/20 bg-cyan-500/5 px-2 py-1 text-[10px] text-cyan-300">
                                SNAPSHOT
                              </span>
                            )}
                          </div>

                          <div className="mt-3 grid gap-2 sm:grid-cols-2">
                            <div className="rounded-lg border border-slate-800/80 bg-black/10 p-2.5">
                              <div className="flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-600">
                                <Fingerprint size={11} />
                                Identity
                              </div>

                              <div className="mt-1 truncate text-xs font-medium text-slate-300">
                                {item.object_reference ||
                                  incident.primary_track_id ||
                                  "Not available"}
                              </div>
                            </div>

                            <div
                              className={`rounded-lg border p-2.5 ${
                                getEvidenceCameraMeta(incident, item, index)
                                  .className
                              }`}
                            >
                              <div className="flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.12em] opacity-60">
                                {React.createElement(
                                  getEvidenceCameraMeta(incident, item, index)
                                    .icon,
                                  { size: 11 },
                                )}
                                Camera Source
                              </div>

                              <div className="mt-1 truncate text-xs font-semibold">
                                {
                                  getEvidenceCameraMeta(incident, item, index)
                                    .label
                                }
                              </div>
                            </div>
                          </div>

                          {getSafeEvidenceDescription(item.description) && (
                            <div className="mt-1 line-clamp-2 text-[11px] leading-5 text-slate-600">
                              {getSafeEvidenceDescription(item.description)}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* FOOTER */}

                <div className="mt-5 flex flex-col gap-3 border-t border-slate-800 pt-4 sm:flex-row sm:items-center sm:justify-between">
                  <div className="flex flex-wrap items-center gap-3 text-[11px] text-slate-600">
                    <span>
                      Evaluation: {incident.evaluation_status || "PENDING"}
                    </span>

                    {incident.evaluation_confidence !== null &&
                      incident.evaluation_confidence !== undefined && (
                        <span>
                          Confidence:{" "}
                          {(
                            Number(incident.evaluation_confidence) * 100
                          ).toFixed(1)}
                          %
                        </span>
                      )}

                    {incident.evaluated_by && (
                      <span>By {incident.evaluated_by}</span>
                    )}
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => openIncident(incident)}
                      className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800 hover:text-white sm:flex-none"
                    >
                      <Eye size={14} />
                      View Incident
                    </button>

                    {!isClosedStatus(status) && (
                      <button
                        type="button"
                        onClick={() => resolve(incident)}
                        disabled={resolvingId === incident.id}
                        className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg bg-emerald-600 px-3.5 py-2 text-xs font-semibold text-white hover:bg-emerald-500 disabled:opacity-50 sm:flex-none"
                      >
                        {resolvingId === incident.id ? (
                          <RefreshCw size={14} className="animate-spin" />
                        ) : (
                          <CheckCircle2 size={14} />
                        )}

                        {resolvingId === incident.id
                          ? "Resolving..."
                          : "Resolve"}
                      </button>
                    )}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}

      {/* ======================================================
       * INCIDENT DETAIL MODAL
       * ====================================================== */}

      {selectedIncident && (
        <div
          className="fixed inset-0 z-[130] flex items-center justify-center bg-black/80 p-3 backdrop-blur-md sm:p-4"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) {
              closeIncident();
            }
          }}
        >
          <div className="flex max-h-[96vh] w-[min(96vw,1440px)] max-w-none flex-col overflow-hidden rounded-2xl border border-slate-700 bg-[#07101b] shadow-[0_30px_120px_rgba(0,0,0,0.55)]">
            {/* ==================================================
             * MODAL HEADER
             * ================================================== */}

            <div className="flex shrink-0 items-center justify-between gap-3 border-b border-slate-800 px-4 py-4 sm:px-5">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge
                    className={
                      getDomainMeta(getIncidentDomain(selectedIncident))
                        .className
                    }
                  >
                    {React.createElement(
                      getDomainMeta(getIncidentDomain(selectedIncident)).icon,
                      {
                        size: 12,
                      },
                    )}

                    {getDomainMeta(getIncidentDomain(selectedIncident)).label}
                  </Badge>

                  <span className="text-xs text-slate-600">
                    Incident #{selectedIncident.id}
                  </span>
                </div>

                <h2 className="mt-2 truncate text-lg font-semibold text-white sm:text-xl">
                  {selectedIncident.incident_type || "GENERAL INCIDENT"}
                </h2>
              </div>

              <div className="flex shrink-0 items-center gap-2">
                <button
                  type="button"
                  onClick={() => exportEvidencePdf(selectedIncident)}
                  disabled={exportingEvidenceId === selectedIncident.id}
                  className="inline-flex items-center gap-2 rounded-lg border border-blue-500/30 bg-blue-500/10 px-3 py-2 text-xs font-semibold text-blue-200 transition hover:bg-blue-500/20 disabled:cursor-wait disabled:opacity-50"
                  title="Generate a restricted, watermarked, auditable evidence PDF"
                >
                  {exportingEvidenceId === selectedIncident.id ? (
                    <RefreshCw size={14} className="animate-spin" />
                  ) : (
                    <Download size={14} />
                  )}
                  <span className="hidden sm:inline">
                    {exportingEvidenceId === selectedIncident.id
                      ? "Generating..."
                      : "Evidence PDF"}
                  </span>
                  <ShieldCheck size={13} className="text-cyan-300" />
                </button>

                <button
                  type="button"
                  onClick={closeIncident}
                  className="rounded-lg p-2 text-slate-500 transition hover:bg-slate-800 hover:text-white"
                  aria-label="Close incident details"
                >
                  <X size={19} />
                </button>
              </div>
            </div>

            {/* ==================================================
             * MODAL CONTENT
             * ================================================== */}

            <div className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-5">
              {/* TOP INFORMATION */}

              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                    Camera
                  </div>

                  <div className="flex min-w-0 items-center gap-2 break-all text-sm text-slate-300">
                    <Camera size={15} className="shrink-0 text-blue-300" />

                    {getIncidentCameraLabel(selectedIncident)}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                    Track
                  </div>

                  <div className="break-all text-sm font-mono text-slate-300">
                    {selectedIncident.primary_track_id || "N/A"}
                  </div>
                </div>

                <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4 sm:col-span-2 lg:col-span-1">
                  <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                    Started
                  </div>

                  <div className="text-sm text-slate-300">
                    {formatDate(selectedIncident.started_at)}
                  </div>
                </div>
              </div>

              {/* FULL-WIDTH WATCHLIST JOURNEY MAP */}

              {(getIncidentDomain(selectedIncident) === "VEHICLE" ||
                getIncidentDomain(selectedIncident) === "PERSON") && (
                <section className="mt-4 min-w-0">
                  <div className="mb-3 flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
                    <div>
                      <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-cyan-300">
                        <MapPin size={13} />
                        Evidence Journey Map
                      </div>

                      <p className="mt-1 text-xs text-slate-500">
                        Confirmed camera observations and movement path
                      </p>
                    </div>

                    <span className="text-[10px] text-slate-600">
                      Map refreshes automatically
                    </span>
                  </div>

                  <div className="min-w-0 overflow-hidden rounded-2xl [&>*]:!w-full [&_.leaflet-container]:!min-h-[380px] [&_.leaflet-container]:!w-full [&_.leaflet-container]:!rounded-xl [&_.leaflet-container]:!bg-slate-950 [&_.leaflet-container]:!z-0 sm:[&_.leaflet-container]:!min-h-[420px]">
                    <WatchlistJourneyEvidence
                      incidentId={selectedIncident.id}
                      onOpenSnapshot={openEvidenceSnapshot}
                    />
                  </div>
                </section>
              )}

              {/* MAIN CONTENT */}

              <div className="mt-4 grid gap-4 xl:grid-cols-[minmax(0,1fr)_300px]">
                {/* ==================================================
                 * LEFT COLUMN
                 * ================================================== */}

                <div className="min-w-0 space-y-4">
                  {/* INCIDENT SNAPSHOT */}

                  {selectedIncident.snapshot_id && (
                    <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/[0.04] p-4">
                      <div className="mb-3 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-cyan-300">
                            Incident Snapshot
                          </div>

                          <div className="mt-1 text-[11px] text-slate-500">
                            Primary evidence frame
                          </div>
                        </div>

                        {snapshotUrl && (
                          <a
                            href={snapshotUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex w-fit items-center justify-center rounded-lg border border-cyan-500/20 px-2.5 py-1.5 text-[10px] font-semibold text-cyan-300 hover:bg-cyan-500/10"
                          >
                            Open Full
                          </a>
                        )}
                      </div>

                      <div className="flex min-h-[220px] items-center justify-center overflow-hidden rounded-lg border border-slate-800 bg-black sm:min-h-[300px]">
                        {snapshotLoading ? (
                          <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
                            <RefreshCw
                              size={24}
                              className="animate-spin text-cyan-300"
                            />

                            <div className="mt-3 text-xs text-slate-500">
                              Loading incident snapshot...
                            </div>
                          </div>
                        ) : snapshotError ? (
                          <div className="px-6 py-12 text-center">
                            <AlertTriangle
                              size={26}
                              className="mx-auto mb-3 text-red-400"
                            />

                            <div className="text-sm font-medium text-red-300">
                              Snapshot unavailable
                            </div>

                            <div className="mx-auto mt-2 max-w-lg break-words text-xs leading-5 text-slate-600">
                              {snapshotError}
                            </div>
                          </div>
                        ) : snapshotUrl ? (
                          <img
                            src={snapshotUrl}
                            alt="Incident evidence snapshot"
                            className="block max-h-[55vh] w-full object-contain"
                            onError={(event) => {
                              console.error(
                                "[INTEL-I][SNAPSHOT] Browser failed to decode image",
                                {
                                  snapshotId: selectedIncident?.snapshot_id,
                                  src: event.currentTarget.src,
                                },
                              );

                              setSnapshotUrl(null);

                              setSnapshotError(
                                "Snapshot data was received, but the browser could not decode it as an image.",
                              );
                            }}
                          />
                        ) : (
                          <div className="py-16 text-xs text-slate-600">
                            No snapshot available.
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* DESCRIPTION */}

                  <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                    <div className="mb-2 flex items-center gap-2">
                      <FileText size={15} className="text-slate-600" />

                      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                        Incident Description
                      </div>
                    </div>

                    <div className="whitespace-pre-wrap break-words text-sm leading-6 text-slate-300">
                      {selectedIncident.description ||
                        selectedIncident.evidence_text ||
                        "No description provided."}
                    </div>
                  </div>

                  {/* EVIDENCE */}

                  <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                    <div className="mb-3 flex items-center justify-between gap-3">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                        Evidence
                      </div>

                      <span className="text-[10px] text-slate-600">
                        {Array.isArray(selectedIncident.evidence)
                          ? selectedIncident.evidence.length
                          : 0}{" "}
                        items
                      </span>
                    </div>

                    <div className="space-y-2">
                      {(Array.isArray(selectedIncident.evidence)
                        ? selectedIncident.evidence
                        : []
                      ).map((item, index) => (
                        <div
                          key={item.id ?? `selected-evidence-${index}`}
                          className="rounded-xl border border-slate-800 bg-[#08111d] p-3"
                        >
                          <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                            <div className="min-w-0">
                              <div className="text-xs font-semibold text-slate-300">
                                Evidence {String(index + 1).padStart(2, "0")}
                              </div>

                              <div className="mt-1 text-[11px] text-slate-600">
                                Confirmed incident evidence
                              </div>
                            </div>

                            {item.snapshot_id && (
                              <button
                                type="button"
                                onClick={() =>
                                  openEvidenceSnapshot(item.snapshot_id)
                                }
                                disabled={
                                  openingSnapshotId === item.snapshot_id
                                }
                                className="inline-flex w-fit shrink-0 items-center gap-1.5 rounded-lg border border-cyan-500/20 bg-cyan-500/5 px-2.5 py-1.5 text-[10px] text-cyan-300 transition hover:bg-cyan-500/10 disabled:cursor-wait disabled:opacity-50"
                              >
                                {openingSnapshotId === item.snapshot_id ? (
                                  <RefreshCw
                                    size={12}
                                    className="animate-spin"
                                  />
                                ) : (
                                  <Eye size={12} />
                                )}
                                Snapshot #{item.snapshot_id}
                              </button>
                            )}
                          </div>

                          <div className="mt-3 grid gap-2 sm:grid-cols-3">
                            <div className="rounded-lg border border-slate-800/80 bg-black/10 p-3">
                              <div className="text-[9px] font-semibold uppercase tracking-[0.13em] text-slate-600">
                                Evidence Type
                              </div>

                              <div className="mt-1.5 truncate text-xs font-semibold text-slate-300">
                                {item.evidence_type ||
                                  getEvidenceDisplayType(
                                    selectedIncident,
                                    item,
                                  ) ||
                                  "OTHER"}
                              </div>
                            </div>

                            <div className="rounded-lg border border-slate-800/80 bg-black/10 p-3">
                              <div className="flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.13em] text-slate-600">
                                <Fingerprint size={11} />
                                Identity
                              </div>

                              <div className="mt-1.5 truncate text-xs font-semibold text-slate-300">
                                {item.object_reference ||
                                  selectedIncident.primary_track_id ||
                                  "Not available"}
                              </div>
                            </div>

                            <div
                              className={`rounded-lg border p-3 ${
                                getEvidenceCameraMeta(
                                  selectedIncident,
                                  item,
                                  index,
                                ).className
                              }`}
                            >
                              <div className="flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.13em] opacity-60">
                                {React.createElement(
                                  getEvidenceCameraMeta(
                                    selectedIncident,
                                    item,
                                    index,
                                  ).icon,
                                  { size: 11 },
                                )}
                                Camera Source
                              </div>

                              <div className="mt-1.5 truncate text-xs font-semibold">
                                {
                                  getEvidenceCameraMeta(
                                    selectedIncident,
                                    item,
                                    index,
                                  ).label
                                }
                              </div>

                              <div className="mt-1 truncate text-[10px] opacity-60">
                                {
                                  getEvidenceCameraMeta(
                                    selectedIncident,
                                    item,
                                    index,
                                  ).helper
                                }
                              </div>
                            </div>
                          </div>

                          {getSafeEvidenceDescription(item.description) && (
                            <div className="mt-2 whitespace-pre-wrap break-words text-xs leading-5 text-slate-500">
                              {getSafeEvidenceDescription(item.description)}
                            </div>
                          )}
                        </div>
                      ))}

                      {(!Array.isArray(selectedIncident.evidence) ||
                        selectedIncident.evidence.length === 0) && (
                        <div className="rounded-lg border border-dashed border-slate-800 px-4 py-8 text-center text-xs text-slate-600">
                          No evidence items available.
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* ==================================================
                 * RIGHT COLUMN
                 * ================================================== */}

                <div className="min-w-0 space-y-4">
                  {/* CASE STATUS */}

                  <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                    <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                      Case Status
                    </div>

                    <div className="flex flex-wrap gap-2">
                      <Badge
                        className={
                          getSeverityClass(
                            selectedIncident.severity ||
                              selectedIncident.evaluation_severity ||
                              "INFO",
                          ).className
                        }
                      >
                        {
                          getSeverityClass(
                            selectedIncident.severity ||
                              selectedIncident.evaluation_severity ||
                              "INFO",
                          ).label
                        }
                      </Badge>

                      <Badge
                        className={getStatusClass(selectedIncident.status)}
                      >
                        {String(
                          selectedIncident.status || "OPEN",
                        ).toUpperCase()}
                      </Badge>
                    </div>
                  </div>

                  {/* EVALUATION */}

                  <div className="rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                    <div className="mb-3 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                      Evaluation
                    </div>

                    <div className="space-y-3 text-xs">
                      <div className="flex items-start justify-between gap-3">
                        <span className="text-slate-600">Status</span>

                        <span className="max-w-[180px] text-right text-slate-300">
                          {selectedIncident.evaluation_status || "PENDING"}
                        </span>
                      </div>

                      <div className="flex items-start justify-between gap-3">
                        <span className="text-slate-600">Score</span>

                        <span className="text-right text-slate-300">
                          {selectedIncident.evaluation_score !== null &&
                          selectedIncident.evaluation_score !== undefined
                            ? Number(selectedIncident.evaluation_score).toFixed(
                                3,
                              )
                            : "—"}
                        </span>
                      </div>

                      <div className="flex items-start justify-between gap-3">
                        <span className="text-slate-600">Confidence</span>

                        <span className="text-right text-slate-300">
                          {selectedIncident.evaluation_confidence !== null &&
                          selectedIncident.evaluation_confidence !== undefined
                            ? `${(
                                Number(selectedIncident.evaluation_confidence) *
                                100
                              ).toFixed(1)}%`
                            : "—"}
                        </span>
                      </div>

                      <div className="flex items-start justify-between gap-3">
                        <span className="text-slate-600">Evaluated by</span>

                        <span className="max-w-[180px] break-words text-right text-slate-300">
                          {selectedIncident.evaluated_by || "—"}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* EVALUATION REASON */}

                  {selectedIncident.evaluation_reason && (
                    <div className="rounded-xl border border-blue-500/10 bg-blue-500/[0.03] p-4">
                      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-blue-300/70">
                        Evaluation Reason
                      </div>

                      <div className="whitespace-pre-wrap break-words text-xs leading-5 text-slate-400">
                        {selectedIncident.evaluation_reason}
                      </div>
                    </div>
                  )}

                  {/* RESOLVE */}

                  {!isClosedStatus(selectedIncident.status) && (
                    <button
                      type="button"
                      onClick={() => resolve(selectedIncident)}
                      disabled={resolvingId === selectedIncident.id}
                      className="inline-flex w-full items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-wait disabled:opacity-50"
                    >
                      {resolvingId === selectedIncident.id ? (
                        <RefreshCw size={16} className="animate-spin" />
                      ) : (
                        <CheckCircle2 size={16} />
                      )}

                      {resolvingId === selectedIncident.id
                        ? "Resolving..."
                        : "Resolve Incident"}
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

export default IncidentCenter;