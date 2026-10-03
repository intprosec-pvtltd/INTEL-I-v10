import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Database,
  Edit3,
  Eye,
  EyeOff,
  FileJson2,
  KeyRound,
  Link2,
  Loader2,
  LockKeyhole,
  Plus,
  RefreshCw,
  RotateCcw,
  Save,
  Search,
  ServerCog,
  Shield,
  ShieldCheck,
  Trash2,
  X,
  Zap,
} from "lucide-react";
import toast from "react-hot-toast";
import api from "../api/axios";

/* -------------------------------------------------------------------------- */
/*                                 CONSTANTS                                  */
/* -------------------------------------------------------------------------- */

const API_ROOT = "/api/watchlist-integrations";

const DEFAULT_FIELD_MAPPING = {
  external_id: "id",
  entity_type: "entity_type",
  category: "category",
  active: "active",
  name: "name",
  plate: "registration_number",
  priority: "priority",
  case_reference: "case_reference",
  image_data: "image_base64",
  image_content_type: "image_content_type",
  make: "make",
  model: "model",
  color: "color",
  description: "description",
};

const createInitialForm = () => ({
  name: "",
  description: "",
  base_url: "",
  auth_type: "NONE",
  credential: "",
  api_key_header: "",
  entity_types: ["PERSON", "VEHICLE"],
  records_path: "records",
  sync_mode: "INTERVAL",
  sync_interval_seconds: 300,
  tls_verify: true,
  pagination: {},
  field_mapping: { ...DEFAULT_FIELD_MAPPING },
});

const inputClass =
  "mt-1.5 w-full rounded-lg border border-[#283957] bg-[#08111f] px-3.5 py-2.5 text-sm text-slate-100 outline-none transition placeholder:text-slate-600 focus:border-cyan-400/70 focus:ring-2 focus:ring-cyan-400/10 disabled:cursor-not-allowed disabled:opacity-60";

const labelClass = "text-xs font-semibold text-slate-200";

const buttonBase =
  "inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2.5 text-xs font-bold transition disabled:cursor-not-allowed disabled:opacity-50";

const secondaryButtonClass =
  `${buttonBase} border border-[#2a3957] bg-[#0b1525] text-slate-200 hover:border-cyan-400/40 hover:bg-[#101d31]`;

const primaryButtonClass =
  `${buttonBase} border border-cyan-400/30 bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/15`;

const accentButtonClass =
  `${buttonBase} border border-blue-500/40 bg-blue-600 text-white shadow-lg shadow-blue-600/10 hover:bg-blue-500`;

/* -------------------------------------------------------------------------- */
/*                                  HELPERS                                   */
/* -------------------------------------------------------------------------- */

function normalizeDate(value) {
  if (!value) return null;

  const raw = String(value);
  const hasTimezone = /(?:Z|[+-]\d{2}:\d{2})$/i.test(raw);
  const parsed = new Date(hasTimezone ? raw : `${raw}Z`);

  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function formatDate(value) {
  const date = normalizeDate(value);
  if (!date) return "—";

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function statusTone(status) {
  switch (status) {
    case "CONNECTED":
      return {
        outer: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
        dot: "bg-emerald-400",
      };
    case "DEGRADED":
      return {
        outer: "border-amber-500/30 bg-amber-500/10 text-amber-300",
        dot: "bg-amber-400",
      };
    case "ERROR":
      return {
        outer: "border-rose-500/30 bg-rose-500/10 text-rose-300",
        dot: "bg-rose-400",
      };
    case "DISABLED":
      return {
        outer: "border-slate-600 bg-slate-700/20 text-slate-400",
        dot: "bg-slate-500",
      };
    default:
      return {
        outer: "border-slate-600 bg-slate-700/10 text-slate-400",
        dot: "bg-slate-500",
      };
  }
}

function StatusBadge({ status }) {
  const normalized = status || "NEVER_SYNCED";
  const tone = statusTone(normalized);

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-[10px] font-extrabold uppercase tracking-[0.12em] ${tone.outer}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${tone.dot}`} />
      {normalized.replaceAll("_", " ")}
    </span>
  );
}

function MetricCard({ icon: Icon, label, value, helper, tone = "blue" }) {
  const toneMap = {
    blue: {
      border: "border-blue-500/25",
      bg: "bg-blue-500/[0.045]",
      icon: "bg-blue-500/10 text-blue-300 border-blue-400/20",
    },
    emerald: {
      border: "border-emerald-500/25",
      bg: "bg-emerald-500/[0.045]",
      icon: "bg-emerald-500/10 text-emerald-300 border-emerald-400/20",
    },
    amber: {
      border: "border-amber-500/25",
      bg: "bg-amber-500/[0.045]",
      icon: "bg-amber-500/10 text-amber-300 border-amber-400/20",
    },
    rose: {
      border: "border-rose-500/25",
      bg: "bg-rose-500/[0.045]",
      icon: "bg-rose-500/10 text-rose-300 border-rose-400/20",
    },
  };

  const selected = toneMap[tone] || toneMap.blue;

  return (
    <div
      className={`rounded-xl border ${selected.border} ${selected.bg} px-4 py-3.5`}
    >
      <div className="flex items-center justify-between gap-4">
        <div className="min-w-0">
          <p className="text-[10px] font-medium text-slate-400">{label}</p>
          <p className="mt-0.5 text-xl font-extrabold text-white">{value}</p>
          <p className="mt-1 truncate text-[10px] text-slate-500">{helper}</p>
        </div>

        <div
          className={`grid h-9 w-9 shrink-0 place-items-center rounded-xl border ${selected.icon}`}
        >
          <Icon size={17} />
        </div>
      </div>
    </div>
  );
}

function SectionHeader({ icon: Icon, title, description }) {
  return (
    <div className="flex items-start gap-3">
      <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-cyan-500/15 bg-cyan-500/10 text-cyan-300">
        <Icon size={17} />
      </div>

      <div>
        <h3 className="text-sm font-bold text-white">{title}</h3>
        {description && (
          <p className="mt-0.5 text-[11px] leading-5 text-slate-500">
            {description}
          </p>
        )}
      </div>
    </div>
  );
}

/* -------------------------------------------------------------------------- */
/*                                  COMPONENT                                 */
/* -------------------------------------------------------------------------- */

export default function ExternalWatchlists() {
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState("");
  const [formOpen, setFormOpen] = useState(true);
  const [editingId, setEditingId] = useState(null);
  const [showCredential, setShowCredential] = useState(false);
  const [history, setHistory] = useState(null);
  const [historySourceName, setHistorySourceName] = useState("");
  const [searchTerm, setSearchTerm] = useState("");

  const [form, setForm] = useState(createInitialForm);
  const [mappingText, setMappingText] = useState(
    JSON.stringify(DEFAULT_FIELD_MAPPING, null, 2)
  );

  const load = useCallback(async () => {
    setLoading(true);

    try {
      const { data } = await api.get(API_ROOT);
      setSources(data?.sources || []);
    } catch (error) {
      toast.error(
        error.response?.data?.detail || "Could not load external watchlists"
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const stats = useMemo(() => {
    const enabled = sources.filter((source) => source.enabled).length;
    const connected = sources.filter(
      (source) => source.enabled && source.status === "CONNECTED"
    ).length;
    const personRecords = sources.reduce(
      (total, source) => total + Number(source.person_records || 0),
      0
    );
    const vehicleRecords = sources.reduce(
      (total, source) => total + Number(source.vehicle_records || 0),
      0
    );

    return {
      enabled,
      connected,
      personRecords,
      vehicleRecords,
      totalRecords: personRecords + vehicleRecords,
    };
  }, [sources]);

  const filteredSources = useMemo(() => {
    const q = searchTerm.trim().toLowerCase();
    if (!q) return sources;

    return sources.filter((source) => {
      const haystack = [
        source.name,
        source.description,
        source.base_url,
        source.status,
        source.auth_type,
        ...(source.entity_types || []),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return haystack.includes(q);
    });
  }, [sources, searchTerm]);

  const update = (key, value) => {
    setForm((current) => ({ ...current, [key]: value }));
  };

  const resetForm = () => {
    const next = createInitialForm();
    setForm(next);
    setMappingText(JSON.stringify(next.field_mapping, null, 2));
    setEditingId(null);
    setShowCredential(false);
  };

  const startNewSource = () => {
    resetForm();
    setFormOpen(true);

    window.requestAnimationFrame(() => {
      document
        .getElementById("external-watchlist-form")
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  const editSource = (source) => {
    const next = {
      ...createInitialForm(),
      ...source,
      credential: "",
      entity_types: source.entity_types || ["PERSON", "VEHICLE"],
      pagination: source.pagination || {},
    };

    setForm(next);
    setMappingText(
      JSON.stringify(source.field_mapping || DEFAULT_FIELD_MAPPING, null, 2)
    );
    setEditingId(source.id);
    setShowCredential(false);
    setFormOpen(true);

    window.requestAnimationFrame(() => {
      document
        .getElementById("external-watchlist-form")
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  const toggleEntityType = (type) => {
    setForm((current) => {
      const selected = current.entity_types || [];
      const next = selected.includes(type)
        ? selected.filter((item) => item !== type)
        : [...selected, type];

      return { ...current, entity_types: next };
    });
  };

  const validateForm = () => {
    if (!form.name.trim()) {
      toast.error("Source name is required");
      return false;
    }

    if (!form.base_url.trim()) {
      toast.error("Endpoint URL is required");
      return false;
    }

    try {
      const parsed = new URL(form.base_url.trim());

      if (
        parsed.protocol !== "https:" &&
        parsed.hostname !== "127.0.0.1" &&
        parsed.hostname !== "localhost"
      ) {
        toast.error("External watchlist endpoint must use HTTPS");
        return false;
      }
    } catch {
      toast.error("Enter a valid endpoint URL");
      return false;
    }

    if (!form.entity_types?.length) {
      toast.error("Select at least one entity type");
      return false;
    }

    if (
      form.auth_type === "API_KEY_HEADER" &&
      !form.api_key_header.trim()
    ) {
      toast.error("API key header name is required");
      return false;
    }

    if (
      !editingId &&
      form.auth_type !== "NONE" &&
      !String(form.credential || "").trim()
    ) {
      toast.error("Credential is required for the selected authentication type");
      return false;
    }

    return true;
  };

  const save = async (saveAndSync = false) => {
    if (!validateForm()) return;

    let parsedMapping;

    try {
      parsedMapping = JSON.parse(mappingText);

      if (
        !parsedMapping ||
        Array.isArray(parsedMapping) ||
        typeof parsedMapping !== "object"
      ) {
        throw new Error("Invalid mapping");
      }
    } catch {
      toast.error("Field mapping must be a valid JSON object");
      return;
    }

    setBusy("save");

    try {
      const body = {
        ...form,
        name: form.name.trim(),
        description: form.description?.trim() || "",
        base_url: form.base_url.trim(),
        records_path: form.records_path?.trim() || "records",
        api_key_header: form.api_key_header?.trim() || "",
        field_mapping: parsedMapping,
      };

      if (body.auth_type === "NONE") {
        body.credential = "";
        body.api_key_header = "";
      }

      if (editingId) {
        if (!body.credential) {
          delete body.credential;
        }

        await api.patch(`${API_ROOT}/${editingId}`, body);

        toast.success("External source updated");

        const sourceId = editingId;
        resetForm();
        await load();

        if (saveAndSync) {
          await runAction(sourceId, "sync");
        }
      } else {
        const { data } = await api.post(API_ROOT, body);
        const sourceId = data?.source?.id;

        toast.success("External source saved");
        resetForm();
        await load();

        if (saveAndSync && sourceId) {
          await runAction(sourceId, "sync");
        }
      }
    } catch (error) {
      toast.error(error.response?.data?.detail || "Could not save source");
    } finally {
      setBusy("");
    }
  };

  const runAction = async (id, action) => {
    setBusy(`${action}-${id}`);

    try {
      const { data } = await api.post(`${API_ROOT}/${id}/${action}`);

      if (action === "test") {
        if (data?.connected) {
          toast.success(
            `Connected · ${Number(data.records_available || 0).toLocaleString()} records available`
          );
        } else {
          toast.error(data?.error || "Connection test failed");
        }
      } else {
        toast.success(
          `Sync ${data?.status?.toLowerCase() || "completed"} · ${
            data?.created ?? 0
          } created · ${data?.updated ?? 0} updated · ${
            data?.rejected ?? 0
          } rejected`
        );
      }

      await load();
    } catch (error) {
      toast.error(
        error.response?.data?.detail || `Could not ${action} source`
      );
    } finally {
      setBusy("");
    }
  };

  const toggle = async (source) => {
    setBusy(`toggle-${source.id}`);

    try {
      await api.patch(`${API_ROOT}/${source.id}`, {
        enabled: !source.enabled,
      });

      toast.success(source.enabled ? "Source disabled" : "Source enabled");
      await load();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Could not update source");
    } finally {
      setBusy("");
    }
  };

  const showHistory = async (source) => {
    setBusy(`history-${source.id}`);

    try {
      const { data } = await api.get(
        `${API_ROOT}/${source.id}/sync-history`
      );

      setHistory(data?.history || []);
      setHistorySourceName(source.name);
    } catch (error) {
      toast.error(error.response?.data?.detail || "Could not load sync history");
    } finally {
      setBusy("");
    }
  };

  const remove = async (source) => {
    const confirmed = window.confirm(
      `Delete "${source.name}" and its synchronized external watchlist records?`
    );

    if (!confirmed) return;

    setBusy(`delete-${source.id}`);

    try {
      await api.delete(`${API_ROOT}/${source.id}`);
      toast.success("Source deleted");

      if (editingId === source.id) {
        resetForm();
      }

      await load();
    } catch (error) {
      toast.error(error.response?.data?.detail || "Could not delete source");
    } finally {
      setBusy("");
    }
  };

  const isSaving = busy === "save";

  return (
    <section className="mx-auto w-full max-w-[1440px] space-y-4 px-4 py-4 text-slate-100 md:px-5 md:py-5">
      {/* PAGE HEADER */}
      <div className="rounded-2xl border border-[#22314b] bg-[#0a1423] px-5 py-5 shadow-[0_10px_30px_rgba(0,0,0,0.12)]">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-start gap-4">
            <div className="grid h-12 w-12 shrink-0 place-items-center rounded-xl border border-cyan-500/30 bg-cyan-500/10 text-cyan-300 shadow-inner">
              <ShieldCheck size={24} />
            </div>

            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[10px] font-extrabold uppercase tracking-[0.18em] text-cyan-400">
                  Watchlist Integration
                </span>
              </div>

              <h1 className="mt-1 text-2xl font-extrabold tracking-tight text-white">
                External Watchlists
              </h1>

              <p className="mt-1 max-w-3xl text-xs leading-5 text-slate-400">
                Connect authorised government or external REST/HTTPS watchlist
                sources and securely synchronize person and vehicle records into
                INTEL-I.
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-2 rounded-lg border border-amber-500/25 bg-amber-500/10 px-3 py-2 text-[10px] font-bold text-amber-300">
              <Shield size={14} />
              Read-only integration
            </span>

            <button
              type="button"
              onClick={load}
              disabled={loading || !!busy}
              className={secondaryButtonClass}
            >
              <RefreshCw
                size={15}
                className={loading ? "animate-spin" : ""}
              />
              Refresh
            </button>

            <button
              type="button"
              onClick={startNewSource}
              className={accentButtonClass}
            >
              <Plus size={15} />
              Add source
            </button>
          </div>
        </div>
      </div>

      {/* METRICS */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          icon={ServerCog}
          label="Configured sources"
          value={sources.length}
          helper={`${stats.enabled} enabled`}
          tone="blue"
        />

        <MetricCard
          icon={CheckCircle2}
          label="Connected"
          value={stats.connected}
          helper="Healthy active integrations"
          tone="emerald"
        />

        <MetricCard
          icon={Activity}
          label="Person records"
          value={stats.personRecords.toLocaleString()}
          helper="Synchronized FRS watchlist entries"
          tone="blue"
        />

        <MetricCard
          icon={Database}
          label="Vehicle records"
          value={stats.vehicleRecords.toLocaleString()}
          helper={`${stats.totalRecords.toLocaleString()} total synchronized`}
          tone="amber"
        />
      </div>

      {/* FORM */}
      <div
        id="external-watchlist-form"
        className="overflow-hidden rounded-2xl border border-[#22314b] bg-[#0a1423]"
      >
        <button
          type="button"
          onClick={() => setFormOpen((current) => !current)}
          className="flex w-full items-center justify-between gap-4 border-b border-[#1f2d45] px-5 py-4 text-left transition hover:bg-[#0d192a]"
        >
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg border border-cyan-500/20 bg-cyan-500/10 text-cyan-300">
              {editingId ? <Edit3 size={17} /> : <Plus size={17} />}
            </div>

            <div>
              <h2 className="text-sm font-bold text-white">
                {editingId ? "Edit external REST source" : "Add external REST source"}
              </h2>
              <p className="mt-0.5 text-[10px] text-slate-500">
                Configure endpoint, authentication, synchronization and field
                mapping.
              </p>
            </div>
          </div>

          <ChevronDown
            size={18}
            className={`text-slate-500 transition ${
              formOpen ? "rotate-180" : ""
            }`}
          />
        </button>

        {formOpen && (
          <div className="space-y-5 p-5">
            <div className="grid gap-4 xl:grid-cols-2">
              {/* SOURCE DETAILS */}
              <div className="rounded-xl border border-[#23314b] bg-[#09121f] p-4">
                <SectionHeader
                  icon={Link2}
                  title="Source details"
                  description="Identify the integration and define its authorised API endpoint."
                />

                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  <label className={labelClass}>
                    Source name
                    <span className="ml-1 text-rose-400">*</span>
                    <input
                      className={inputClass}
                      value={form.name}
                      onChange={(event) => update("name", event.target.value)}
                      placeholder="Example: State Police Watchlist"
                    />
                  </label>

                  <label className={labelClass}>
                    Endpoint URL
                    <span className="ml-1 text-rose-400">*</span>
                    <input
                      className={inputClass}
                      value={form.base_url}
                      onChange={(event) =>
                        update("base_url", event.target.value)
                      }
                      placeholder="https://authorized.example.gov/api/watchlist"
                    />
                  </label>

                  <label className={`${labelClass} md:col-span-2`}>
                    Description
                    <input
                      className={inputClass}
                      value={form.description || ""}
                      onChange={(event) =>
                        update("description", event.target.value)
                      }
                      placeholder="Short operational description of this source"
                    />
                  </label>

                  <label className={labelClass}>
                    JSON records path
                    <input
                      className={inputClass}
                      value={form.records_path}
                      onChange={(event) =>
                        update("records_path", event.target.value)
                      }
                      placeholder="records"
                    />
                    <span className="mt-1.5 block text-[10px] leading-4 text-slate-500">
                      Use a nested path only when the API wraps records.
                    </span>
                  </label>

                  <div>
                    <p className={labelClass}>Entity types</p>

                    <div className="mt-1.5 grid grid-cols-2 gap-2">
                      {[
                        ["PERSON", "Person"],
                        ["VEHICLE", "Vehicle"],
                      ].map(([value, label]) => {
                        const selected = form.entity_types?.includes(value);

                        return (
                          <button
                            key={value}
                            type="button"
                            onClick={() => toggleEntityType(value)}
                            className={`flex items-center justify-between rounded-lg border px-3 py-2.5 text-xs font-semibold transition ${
                              selected
                                ? "border-blue-500/60 bg-blue-500/10 text-blue-300"
                                : "border-[#283957] bg-[#08111f] text-slate-400 hover:border-slate-500"
                            }`}
                          >
                            <span>{label}</span>

                            <span
                              className={`grid h-4 w-4 place-items-center rounded-full border ${
                                selected
                                  ? "border-blue-400 text-blue-300"
                                  : "border-slate-600 text-transparent"
                              }`}
                            >
                              <Check size={10} />
                            </span>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>

              {/* AUTHENTICATION */}
              <div className="rounded-xl border border-[#23314b] bg-[#09121f] p-4">
                <SectionHeader
                  icon={KeyRound}
                  title="Authentication & transport"
                  description="Credentials are submitted securely and never displayed after saving."
                />

                <div className="mt-4 grid gap-4 md:grid-cols-2">
                  <label
                    className={`${labelClass} ${
                      form.auth_type === "NONE" ? "md:col-span-2" : ""
                    }`}
                  >
                    Authentication
                    <select
                      className={inputClass}
                      value={form.auth_type}
                      onChange={(event) => {
                        update("auth_type", event.target.value);
                        setShowCredential(false);
                      }}
                    >
                      <option value="NONE">None</option>
                      <option value="BEARER">Bearer token</option>
                      <option value="API_KEY_HEADER">API key header</option>
                    </select>
                  </label>

                  {form.auth_type === "API_KEY_HEADER" && (
                    <label className={labelClass}>
                      API key header name
                      <input
                        className={inputClass}
                        value={form.api_key_header || ""}
                        onChange={(event) =>
                          update("api_key_header", event.target.value)
                        }
                        placeholder="X-API-Key"
                      />
                    </label>
                  )}

                  {form.auth_type !== "NONE" && (
                    <label
                      className={`${labelClass} ${
                        form.auth_type === "BEARER"
                          ? "md:col-span-2"
                          : ""
                      }`}
                    >
                      <span className="flex items-center justify-between gap-3">
                        <span>
                          {form.auth_type === "BEARER"
                            ? "Bearer token"
                            : "API key"}
                        </span>

                        {editingId && (
                          <span className="text-[9px] font-normal text-slate-500">
                            leave blank to keep current
                          </span>
                        )}
                      </span>

                      <div className="relative">
                        <input
                          type={showCredential ? "text" : "password"}
                          autoComplete="new-password"
                          className={`${inputClass} pr-11`}
                          value={form.credential || ""}
                          onChange={(event) =>
                            update("credential", event.target.value)
                          }
                          placeholder={
                            editingId ? "••••••••••••••••" : "Enter credential"
                          }
                        />

                        <button
                          type="button"
                          onClick={() =>
                            setShowCredential((current) => !current)
                          }
                          className="absolute right-2 top-[10px] grid h-8 w-8 place-items-center rounded-md text-slate-500 transition hover:bg-[#101d31] hover:text-slate-200"
                          aria-label={
                            showCredential
                              ? "Hide credential"
                              : "Show credential"
                          }
                        >
                          {showCredential ? (
                            <EyeOff size={15} />
                          ) : (
                            <Eye size={15} />
                          )}
                        </button>
                      </div>
                    </label>
                  )}

                  <label className="md:col-span-2 flex cursor-pointer items-start gap-3 rounded-lg border border-[#283957] bg-[#08111f] p-3.5">
                    <input
                      type="checkbox"
                      checked={form.tls_verify}
                      onChange={(event) =>
                        update("tls_verify", event.target.checked)
                      }
                      className="mt-0.5 h-4 w-4 rounded border-slate-600 bg-[#0b1525] text-cyan-500 focus:ring-cyan-400"
                    />

                    <span>
                      <span className="block text-xs font-semibold text-slate-200">
                        Verify TLS certificate
                      </span>

                      <span className="mt-1 block text-[10px] leading-4 text-slate-500">
                        Recommended for production government and enterprise
                        HTTPS endpoints.
                      </span>
                    </span>
                  </label>
                </div>
              </div>
            </div>

            <div className="grid gap-4 xl:grid-cols-[0.82fr_1.18fr]">
              {/* SYNC */}
              <div className="rounded-xl border border-[#23314b] bg-[#09121f] p-4">
                <SectionHeader
                  icon={Clock3}
                  title="Synchronization"
                  description="Choose manual synchronization or a recurring polling interval."
                />

                <div className="mt-4 space-y-4">
                  <label className={labelClass}>
                    Sync mode
                    <select
                      className={inputClass}
                      value={form.sync_mode}
                      onChange={(event) =>
                        update("sync_mode", event.target.value)
                      }
                    >
                      <option value="MANUAL">Manual only</option>
                      <option value="INTERVAL">Scheduled interval</option>
                    </select>
                  </label>

                  {form.sync_mode === "INTERVAL" && (
                    <label className={labelClass}>
                      Sync interval
                      <select
                        className={inputClass}
                        value={form.sync_interval_seconds}
                        onChange={(event) =>
                          update(
                            "sync_interval_seconds",
                            Number(event.target.value)
                          )
                        }
                      >
                        <option value={30}>30 seconds</option>
                        <option value={60}>1 minute</option>
                        <option value={300}>5 minutes</option>
                        <option value={900}>15 minutes</option>
                        <option value={1800}>30 minutes</option>
                        <option value={3600}>1 hour</option>
                        <option value={21600}>6 hours</option>
                        <option value={43200}>12 hours</option>
                        <option value={86400}>24 hours</option>
                      </select>
                    </label>
                  )}

                  <div className="rounded-lg border border-cyan-500/15 bg-cyan-500/[0.06] p-3 text-[10px] leading-5 text-cyan-200">
                    <div className="flex items-start gap-2">
                      <Zap size={14} className="mt-0.5 shrink-0" />
                      <span>
                        Save &amp; sync validates and synchronizes immediately
                        after the source is saved.
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* MAPPING */}
              <div className="rounded-xl border border-[#23314b] bg-[#09121f] p-4">
                <SectionHeader
                  icon={FileJson2}
                  title="Field mapping"
                  description="Map external JSON fields to INTEL-I's canonical watchlist schema."
                />

                <textarea
                  spellCheck={false}
                  value={mappingText}
                  onChange={(event) => setMappingText(event.target.value)}
                  className="mt-4 min-h-[280px] w-full resize-y rounded-xl border border-[#23314b] bg-[#030816] p-4 font-mono text-[11px] leading-5 text-cyan-100 outline-none transition focus:border-cyan-500/50 focus:ring-2 focus:ring-cyan-500/10"
                />

                <p className="mt-2 text-[10px] leading-5 text-slate-500">
                  Person matching requires valid base64 image data mapped to{" "}
                  <code className="rounded bg-[#101d31] px-1 py-0.5 text-cyan-300">
                    image_data
                  </code>
                  . Vehicle registration values should map to{" "}
                  <code className="rounded bg-[#101d31] px-1 py-0.5 text-cyan-300">
                    plate
                  </code>
                  .
                </p>
              </div>
            </div>

            <div className="flex flex-col gap-3 border-t border-[#1f2d45] pt-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="flex items-center gap-2 text-[10px] text-slate-500">
                <LockKeyhole size={13} className="text-emerald-400" />
                Credentials remain encrypted server-side. External integrations
                are read-only.
              </div>

              <div className="flex flex-wrap justify-end gap-2">
                <button
                  type="button"
                  onClick={resetForm}
                  disabled={isSaving}
                  className={secondaryButtonClass}
                >
                  <RotateCcw size={14} />
                  Reset
                </button>

                <button
                  type="button"
                  disabled={!!busy}
                  onClick={() => save(false)}
                  className={primaryButtonClass}
                >
                  {isSaving ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Save size={14} />
                  )}
                  {editingId ? "Update source" : "Save source"}
                </button>

                <button
                  type="button"
                  disabled={!!busy}
                  onClick={() => save(true)}
                  className={accentButtonClass}
                >
                  {isSaving ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <RefreshCw size={14} />
                  )}
                  {editingId ? "Update & sync" : "Save & sync"}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* SOURCE TABLE */}
      <div className="overflow-hidden rounded-2xl border border-[#22314b] bg-[#0a1423]">
        <div className="flex flex-col gap-3 border-b border-[#1f2d45] px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex items-center gap-3">
            <div className="grid h-9 w-9 place-items-center rounded-lg border border-blue-500/20 bg-blue-500/10 text-blue-300">
              <Database size={17} />
            </div>

            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-sm font-bold text-white">
                  Configured sources
                </h2>

                <span className="grid min-w-5 place-items-center rounded-full bg-[#1a2940] px-1.5 py-0.5 text-[9px] font-bold text-slate-300">
                  {sources.length}
                </span>
              </div>

              <p className="mt-0.5 text-[10px] text-slate-500">
                Test, synchronize, review history, enable or disable external
                watchlist sources.
              </p>
            </div>
          </div>

          <div className="relative w-full lg:w-[260px]">
            <Search
              size={14}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
            />

            <input
              value={searchTerm}
              onChange={(event) => setSearchTerm(event.target.value)}
              placeholder="Search source, URL or status"
              className="w-full rounded-lg border border-[#283957] bg-[#08111f] py-2.5 pl-9 pr-3 text-xs text-slate-200 outline-none placeholder:text-slate-600 focus:border-cyan-400/60"
            />
          </div>
        </div>

        {loading ? (
          <div className="grid min-h-[220px] place-items-center">
            <div className="text-center">
              <Loader2
                size={28}
                className="mx-auto animate-spin text-cyan-400"
              />
              <p className="mt-3 text-xs text-slate-500">
                Loading external sources…
              </p>
            </div>
          </div>
        ) : filteredSources.length === 0 ? (
          <div className="grid min-h-[230px] place-items-center px-6 py-8 text-center">
            <div>
              <div className="mx-auto grid h-12 w-12 place-items-center rounded-xl border border-[#24334e] bg-[#0d192a] text-slate-500">
                <ServerCog size={21} />
              </div>

              <p className="mt-4 text-sm font-bold text-slate-200">
                {sources.length === 0
                  ? "No external sources yet"
                  : "No matching sources"}
              </p>

              <p className="mt-1 text-[10px] leading-5 text-slate-500">
                {sources.length === 0
                  ? "Add the first authorised watchlist source using the form above."
                  : "Try another source name, URL, entity type or status."}
              </p>
            </div>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[1180px] text-left text-xs">
              <thead className="bg-[#07101d] text-[9px] font-bold uppercase tracking-[0.12em] text-slate-500">
                <tr>
                  <th className="px-5 py-3">Source</th>
                  <th className="px-5 py-3">Status</th>
                  <th className="px-5 py-3">Records</th>
                  <th className="px-5 py-3">Synchronization</th>
                  <th className="px-5 py-3">Actions</th>
                </tr>
              </thead>

              <tbody className="divide-y divide-[#1b2940]">
                {filteredSources.map((source) => {
                  const testBusy = busy === `test-${source.id}`;
                  const syncBusy = busy === `sync-${source.id}`;
                  const toggleBusy = busy === `toggle-${source.id}`;
                  const deleteBusy = busy === `delete-${source.id}`;
                  const historyBusy = busy === `history-${source.id}`;

                  return (
                    <tr
                      key={source.id}
                      className="align-top transition hover:bg-[#0d192a]/70"
                    >
                      <td className="px-5 py-4">
                        <div className="flex items-start gap-3">
                          <div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg border border-blue-500/20 bg-blue-500/10 text-blue-300">
                            <Link2 size={15} />
                          </div>

                          <div className="min-w-0">
                            <p className="font-bold text-slate-100">
                              {source.name}
                            </p>

                            <p
                              className="mt-1 max-w-[360px] truncate text-[10px] text-slate-500"
                              title={source.base_url}
                            >
                              {source.base_url}
                            </p>

                            <div className="mt-2 flex flex-wrap gap-1.5">
                              {(source.entity_types || []).map((type) => (
                                <span
                                  key={type}
                                  className="rounded-md border border-[#293a59] bg-[#101d31] px-2 py-1 text-[9px] font-bold text-slate-300"
                                >
                                  {type}
                                </span>
                              ))}

                              <span className="rounded-md border border-[#293a59] bg-[#101d31] px-2 py-1 text-[9px] font-bold text-slate-400">
                                {source.auth_type || "NONE"}
                              </span>

                              <span className="rounded-md border border-emerald-500/20 bg-emerald-500/10 px-2 py-1 text-[9px] font-bold text-emerald-300">
                                READ ONLY
                              </span>
                            </div>
                          </div>
                        </div>
                      </td>

                      <td className="px-5 py-4">
                        <StatusBadge
                          status={source.enabled ? source.status : "DISABLED"}
                        />

                        {source.last_error && (
                          <div className="mt-2 flex max-w-[250px] items-start gap-1.5 text-[10px] leading-4 text-amber-300">
                            <AlertTriangle
                              size={12}
                              className="mt-0.5 shrink-0"
                            />
                            <span>{source.last_error}</span>
                          </div>
                        )}
                      </td>

                      <td className="px-5 py-4">
                        <div className="space-y-1.5">
                          <div className="flex min-w-[140px] items-center justify-between gap-4 rounded-md bg-[#08111f] px-2.5 py-2">
                            <span className="text-[10px] text-slate-500">
                              Person
                            </span>
                            <span className="text-[10px] font-bold text-slate-200">
                              {Number(
                                source.person_records || 0
                              ).toLocaleString()}
                            </span>
                          </div>

                          <div className="flex min-w-[140px] items-center justify-between gap-4 rounded-md bg-[#08111f] px-2.5 py-2">
                            <span className="text-[10px] text-slate-500">
                              Vehicle
                            </span>
                            <span className="text-[10px] font-bold text-slate-200">
                              {Number(
                                source.vehicle_records || 0
                              ).toLocaleString()}
                            </span>
                          </div>
                        </div>
                      </td>

                      <td className="px-5 py-4">
                        <div className="space-y-1.5 text-[10px]">
                          <p>
                            <span className="text-slate-600">Last:</span>{" "}
                            <span className="text-slate-300">
                              {source.last_sync_at
                                ? formatDate(source.last_sync_at)
                                : "Never"}
                            </span>
                          </p>

                          <p>
                            <span className="text-slate-600">Success:</span>{" "}
                            <span className="text-slate-300">
                              {source.last_success_at
                                ? formatDate(source.last_success_at)
                                : "—"}
                            </span>
                          </p>

                          <p>
                            <span className="text-slate-600">Next:</span>{" "}
                            <span className="text-slate-300">
                              {source.next_sync_at
                                ? formatDate(source.next_sync_at)
                                : source.sync_mode === "MANUAL"
                                ? "Manual"
                                : "—"}
                            </span>
                          </p>
                        </div>
                      </td>

                      <td className="px-5 py-4">
                        <div className="flex max-w-[390px] flex-wrap gap-1.5">
                          <button
                            type="button"
                            onClick={() => editSource(source)}
                            disabled={!!busy}
                            className={secondaryButtonClass}
                          >
                            <Edit3 size={13} />
                            Edit
                          </button>

                          <button
                            type="button"
                            onClick={() => runAction(source.id, "test")}
                            disabled={!!busy}
                            className={secondaryButtonClass}
                          >
                            {testBusy ? (
                              <Loader2 size={13} className="animate-spin" />
                            ) : (
                              <Activity size={13} />
                            )}
                            Test
                          </button>

                          <button
                            type="button"
                            onClick={() => runAction(source.id, "sync")}
                            disabled={!!busy || !source.enabled}
                            className={secondaryButtonClass}
                          >
                            {syncBusy ? (
                              <Loader2 size={13} className="animate-spin" />
                            ) : (
                              <RefreshCw size={13} />
                            )}
                            Sync
                          </button>

                          <button
                            type="button"
                            onClick={() => toggle(source)}
                            disabled={!!busy}
                            className={secondaryButtonClass}
                          >
                            {toggleBusy ? (
                              <Loader2 size={13} className="animate-spin" />
                            ) : source.enabled ? (
                              <EyeOff size={13} />
                            ) : (
                              <Eye size={13} />
                            )}
                            {source.enabled ? "Disable" : "Enable"}
                          </button>

                          <button
                            type="button"
                            onClick={() => showHistory(source)}
                            disabled={!!busy}
                            className={secondaryButtonClass}
                          >
                            {historyBusy ? (
                              <Loader2 size={13} className="animate-spin" />
                            ) : (
                              <Clock3 size={13} />
                            )}
                            History
                          </button>

                          <button
                            type="button"
                            onClick={() => remove(source)}
                            disabled={!!busy}
                            className={`${buttonBase} border border-rose-500/25 bg-rose-500/10 text-rose-300 hover:bg-rose-500/15`}
                            aria-label={`Delete ${source.name}`}
                          >
                            {deleteBusy ? (
                              <Loader2 size={13} className="animate-spin" />
                            ) : (
                              <Trash2 size={13} />
                            )}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* HISTORY */}
      {history && (
        <div className="overflow-hidden rounded-2xl border border-[#22314b] bg-[#0a1423]">
          <div className="flex items-center justify-between gap-3 border-b border-[#1f2d45] px-5 py-4">
            <div>
              <h2 className="text-sm font-bold text-white">Sync history</h2>
              <p className="mt-0.5 text-[10px] text-slate-500">
                {historySourceName || "External source"}
              </p>
            </div>

            <button
              type="button"
              onClick={() => {
                setHistory(null);
                setHistorySourceName("");
              }}
              className="grid h-8 w-8 place-items-center rounded-lg border border-[#293a59] text-slate-500 transition hover:bg-[#101d31] hover:text-slate-200"
              aria-label="Close sync history"
            >
              <X size={15} />
            </button>
          </div>

          {history.length === 0 ? (
            <p className="px-5 py-6 text-xs text-slate-500">
              No sync history yet.
            </p>
          ) : (
            <div className="divide-y divide-[#1b2940]">
              {history.map((item) => (
                <div
                  key={item.id}
                  className="grid gap-3 px-5 py-3.5 md:grid-cols-[185px_145px_1fr] md:items-center"
                >
                  <div className="text-[10px] text-slate-400">
                    {formatDate(item.created_at)}
                  </div>

                  <div>
                    <StatusBadge status={item.result} />
                  </div>

                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-[10px] text-slate-500">
                    <span>
                      <strong className="text-slate-200">
                        {item.records_received ?? 0}
                      </strong>{" "}
                      received
                    </span>

                    <span>
                      <strong className="text-slate-200">
                        {item.records_created ?? 0}
                      </strong>{" "}
                      created
                    </span>

                    <span>
                      <strong className="text-slate-200">
                        {item.records_updated ?? 0}
                      </strong>{" "}
                      updated
                    </span>

                    <span>
                      <strong className="text-slate-200">
                        {item.records_rejected ?? 0}
                      </strong>{" "}
                      rejected
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* INFO */}
      <div className="grid gap-3 lg:grid-cols-2">
        <div className="flex items-start gap-3 rounded-xl border border-blue-500/20 bg-blue-500/[0.055] p-4">
          <Activity
            size={17}
            className="mt-0.5 shrink-0 text-blue-300"
          />

          <div>
            <p className="text-xs font-bold text-blue-200">
              Person / FRS integration
            </p>

            <p className="mt-1 text-[10px] leading-5 text-slate-400">
              Person matching requires valid base64 face image data mapped to{" "}
              <code className="rounded bg-[#101d31] px-1 py-0.5 text-cyan-300">
                image_data
              </code>
              .
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3 rounded-xl border border-emerald-500/20 bg-emerald-500/[0.055] p-4">
          <ShieldCheck
            size={17}
            className="mt-0.5 shrink-0 text-emerald-300"
          />

          <div>
            <p className="text-xs font-bold text-emerald-200">
              Vehicle watchlist safety
            </p>

            <p className="mt-1 text-[10px] leading-5 text-slate-400">
              Vehicle registrations are normalized into INTEL-I&apos;s existing
              vehicle watchlist. External records do not create synthetic
              detections.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
