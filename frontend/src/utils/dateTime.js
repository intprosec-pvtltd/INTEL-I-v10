export const INDIA_TIME_ZONE = "Asia/Kolkata";

export const parseBackendDate = (value) => {
  if (value === null || value === undefined || value === "") return null;

  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }

  if (typeof value === "number" || /^\d+(\.\d+)?$/.test(String(value).trim())) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return null;
    const milliseconds = Math.abs(numeric) < 10_000_000_000 ? numeric * 1000 : numeric;
    const date = new Date(milliseconds);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  let normalized = String(value).trim();
  if (!normalized) return null;
  normalized = normalized.replace(" ", "T");

  // INTEL-I stores canonical UTC. Legacy API values may omit their offset.
  if (!/(Z|[+-]\d{2}:?\d{2})$/i.test(normalized)) normalized += "Z";
  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? null : date;
};

export const formatIndianDateTime = (value, fallback = "N/A") => {
  const date = parseBackendDate(value);
  if (!date) return fallback;
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: INDIA_TIME_ZONE,
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
};

export const formatIndianDate = (value, fallback = "—") => {
  const date = parseBackendDate(value);
  if (!date) return fallback;
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: INDIA_TIME_ZONE,
    year: "numeric",
    month: "short",
    day: "2-digit",
  }).format(date);
};

export const formatIndianTime = (value, fallback = "—") => {
  const date = parseBackendDate(value);
  if (!date) return fallback;
  return new Intl.DateTimeFormat("en-IN", {
    timeZone: INDIA_TIME_ZONE,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  }).format(date);
};
