import React, {
  useEffect,
  useMemo,
  useState,
} from "react";

import toast from "react-hot-toast";

import {
  Plus,
  Search,
  ShieldAlert,
  Trash2,
  Edit3,
  RefreshCw,
  X,
  CheckCircle2,
  Car,
  AlertTriangle,
  List,
  Activity,
  Clock3,
  MoreVertical,
  Eye,
  FileText,
  ChevronRight,
  Filter,
  Save,
  Power,
} from "lucide-react";

import {
  getWatchlists,
  createWatchlist,
  updateWatchlist,
  deleteWatchlist,
  getWatchlistEntries,
  createWatchlistEntry,
  updateWatchlistEntry,
  deleteWatchlistEntry,
  matchWatchlistPlate,
} from "../api/axios.js";
import WatchlistActiveJourneys from "../components/WatchlistActiveJourneys.jsx";


/* ============================================================
 * CONSTANTS
 * ============================================================ */

const CATEGORIES = [
  "STOLEN",
  "BLACKLISTED",
  "SUSPICIOUS",
  "WANTED",
  "OTHER",
];

const STATUSES = [
  "ACTIVE",
  "CLEARED",
  "EXPIRED",
  "DISABLED",
];

const PRIORITIES = [
  "LOW",
  "MEDIUM",
  "HIGH",
  "CRITICAL",
];

const emptyEntry = {
  plate: "",
  category: "SUSPICIOUS",
  status: "ACTIVE",
  priority: "HIGH",
  description: "",
  source: "",
};

const emptyList = {
  name: "Police Vehicle Watchlist",
  description: "",
  is_active: true,
};


/* ============================================================
 * HELPERS
 * ============================================================ */

const getErrorMessage = (
  error,
  fallback = "Request failed"
) => {
  const detail =
    error?.response?.data?.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (
          typeof item === "string"
        ) {
          return item;
        }

        if (
          item &&
          typeof item.msg === "string"
        ) {
          const location =
            Array.isArray(item.loc)
              ? item.loc.join(".")
              : "";

          return location
            ? `${location}: ${item.msg}`
            : item.msg;
        }

        return "Validation error";
      })
      .join(", ");
  }

  if (
    detail &&
    typeof detail === "object"
  ) {
    if (
      typeof detail.msg === "string"
    ) {
      return detail.msg;
    }

    try {
      return JSON.stringify(detail);
    } catch {
      return fallback;
    }
  }

  if (
    typeof error?.message === "string"
  ) {
    return error.message;
  }

  return fallback;
};


const getCategoryMeta = (category) => {
  const value =
    String(
      category || "OTHER"
    ).toUpperCase();

  switch (value) {
    case "STOLEN":
      return {
        label: "STOLEN",
        className:
          "border-red-500/30 bg-red-500/10 text-red-300",
        dot: "bg-red-400",
      };

    case "BLACKLISTED":
      return {
        label: "BLACKLISTED",
        className:
          "border-purple-500/30 bg-purple-500/10 text-purple-300",
        dot: "bg-purple-400",
      };

    case "SUSPICIOUS":
      return {
        label: "SUSPICIOUS",
        className:
          "border-amber-500/30 bg-amber-500/10 text-amber-300",
        dot: "bg-amber-400",
      };

    case "WANTED":
      return {
        label: "WANTED",
        className:
          "border-orange-500/30 bg-orange-500/10 text-orange-300",
        dot: "bg-orange-400",
      };

    default:
      return {
        label: "OTHER",
        className:
          "border-slate-500/30 bg-slate-500/10 text-slate-300",
        dot: "bg-slate-400",
      };
  }
};


const getPriorityMeta = (priority) => {
  const value =
    String(
      priority || "LOW"
    ).toUpperCase();

  switch (value) {
    case "CRITICAL":
      return {
        label: "CRITICAL",
        className:
          "border-red-500/30 bg-red-500/10 text-red-300",
        dot: "bg-red-400",
      };

    case "HIGH":
      return {
        label: "HIGH",
        className:
          "border-orange-500/30 bg-orange-500/10 text-orange-300",
        dot: "bg-orange-400",
      };

    case "MEDIUM":
      return {
        label: "MEDIUM",
        className:
          "border-amber-500/30 bg-amber-500/10 text-amber-300",
        dot: "bg-amber-400",
      };

    default:
      return {
        label: "LOW",
        className:
          "border-slate-500/30 bg-slate-500/10 text-slate-300",
        dot: "bg-slate-400",
      };
  }
};


const getStatusMeta = (status) => {
  const value =
    String(
      status || "UNKNOWN"
    ).toUpperCase();

  switch (value) {
    case "ACTIVE":
      return {
        label: "ACTIVE",
        className:
          "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
        dot: "bg-emerald-400",
      };

    case "CLEARED":
      return {
        label: "CLEARED",
        className:
          "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
        dot: "bg-cyan-400",
      };

    case "EXPIRED":
      return {
        label: "EXPIRED",
        className:
          "border-orange-500/30 bg-orange-500/10 text-orange-300",
        dot: "bg-orange-400",
      };

    default:
      return {
        label: value,
        className:
          "border-slate-500/30 bg-slate-500/10 text-slate-300",
        dot: "bg-slate-400",
      };
  }
};


const formatDate = (value) => {
  if (!value) {
    return "—";
  }

  const date =
    new Date(value);

  if (
    Number.isNaN(
      date.getTime()
    )
  ) {
    return "—";
  }

  return new Intl.DateTimeFormat(
    "en-IN",
    {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    }
  ).format(date);
};


const normalizePlate = (plate) =>
  String(
    plate || ""
  )
    .trim()
    .toUpperCase();


const Badge = ({
  meta,
}) => (
  <span
    className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[10px] font-semibold tracking-wide ${meta.className}`}
  >
    <span
      className={`h-1.5 w-1.5 rounded-full ${meta.dot}`}
    />
    {meta.label}
  </span>
);


const SectionLabel = ({
  children,
}) => (
  <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
    {children}
  </div>
);


const StatCard = ({
  title,
  value,
  description,
  icon: Icon,
  iconClass,
}) => (
  <div className="rounded-xl border border-slate-800/90 bg-slate-950/60 p-4 shadow-[0_10px_35px_rgba(0,0,0,0.16)]">
    <div className="flex items-start justify-between">
      <div>
        <div className="text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
          {title}
        </div>

        <div className="mt-2 text-2xl font-bold tracking-tight text-white">
          {value}
        </div>
      </div>

      <div
        className={`rounded-xl border border-white/5 bg-white/[0.03] p-2.5 ${
          iconClass || "text-blue-300"
        }`}
      >
        <Icon size={18} />
      </div>
    </div>

    <div className="mt-2 text-xs text-slate-500">
      {description}
    </div>
  </div>
);


/* ============================================================
 * COMPONENT
 * ============================================================ */

const Watchlist = () => {
  const [lists, setLists] =
    useState([]);

  const [selectedId, setSelectedId] =
    useState(null);

  const [entries, setEntries] =
    useState([]);

  const [loading, setLoading] =
    useState(true);

  const [entriesLoading, setEntriesLoading] =
    useState(false);

  const [search, setSearch] =
    useState("");

  const [category, setCategory] =
    useState("");

  const [status, setStatus] =
    useState("");

  const [priorityFilter, setPriorityFilter] =
    useState("");

  const [showListForm, setShowListForm] =
    useState(false);

  const [showEntryForm, setShowEntryForm] =
    useState(false);

  const [showEntryPreview, setShowEntryPreview] =
    useState(false);

  const [editingEntry, setEditingEntry] =
    useState(null);

  const [selectedEntry, setSelectedEntry] =
    useState(null);

  const [listForm, setListForm] =
    useState({
      ...emptyList,
    });

  const [entryForm, setEntryForm] =
    useState({
      ...emptyEntry,
    });

  const [testPlate, setTestPlate] =
    useState("");

  const [matchResult, setMatchResult] =
    useState(null);

  const [saving, setSaving] =
    useState(false);

  const [deletingEntryId, setDeletingEntryId] =
    useState(null);

  const [deletingList, setDeletingList] =
    useState(false);

  const [matchLoading, setMatchLoading] =
    useState(false);

  const [actionMenuId, setActionMenuId] =
    useState(null);


  /* ==========================================================
   * SELECTED WATCHLIST
   * ========================================================== */

  const selected = useMemo(
    () =>
      lists.find(
        (item) =>
          Number(item.id) ===
          Number(selectedId)
      ) || null,
    [lists, selectedId]
  );


  /* ==========================================================
   * WATCHLIST STATS
   * ========================================================== */

  const listStats = useMemo(() => {
    const active =
      entries.filter(
        (entry) =>
          String(
            entry.status || ""
          ).toUpperCase() ===
          "ACTIVE"
      ).length;

    const critical =
      entries.filter(
        (entry) =>
          String(
            entry.priority || ""
          ).toUpperCase() ===
          "CRITICAL"
      ).length;

    const high =
      entries.filter(
        (entry) =>
          String(
            entry.priority || ""
          ).toUpperCase() ===
          "HIGH"
      ).length;

    const stolen =
      entries.filter(
        (entry) =>
          String(
            entry.category || ""
          ).toUpperCase() ===
          "STOLEN"
      ).length;

    return {
      total: entries.length,
      active,
      critical,
      high,
      stolen,
    };
  }, [entries]);


  /* ==========================================================
   * LOAD WATCHLISTS
   * ========================================================== */

  const loadLists = async () => {
    setLoading(true);

    try {
      const data =
        await getWatchlists();

      const next =
        Array.isArray(
          data?.watchlists
        )
          ? data.watchlists
          : [];

      setLists(next);

      if (
        !selectedId &&
        next.length
      ) {
        setSelectedId(
          next[0].id
        );
      }

      if (
        selectedId &&
        !next.some(
          (item) =>
            Number(item.id) ===
            Number(selectedId)
        )
      ) {
        setSelectedId(
          next[0]?.id || null
        );
      }
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Failed to load vehicle watchlists"
        )
      );
    } finally {
      setLoading(false);
    }
  };


  /* ==========================================================
   * LOAD ENTRIES
   * ========================================================== */

  const loadEntries = async () => {
    if (!selectedId) {
      setEntries([]);
      return;
    }

    setEntriesLoading(true);

    try {
      const data =
        await getWatchlistEntries(
          selectedId,
          {
            search:
              search ||
              undefined,

            category:
              category ||
              undefined,

            status:
              status ||
              undefined,

            priority:
              priorityFilter ||
              undefined,
          }
        );

      setEntries(
        Array.isArray(
          data?.entries
        )
          ? data.entries
          : []
      );
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Failed to load watchlist entries"
        )
      );
    } finally {
      setEntriesLoading(false);
    }
  };


  /* ==========================================================
   * INITIAL LOAD
   * ========================================================== */

  useEffect(() => {
    loadLists();
  }, []);

  useEffect(() => {
    loadEntries();
  }, [
    selectedId,
    search,
    category,
    status,
    priorityFilter,
  ]);


  /* ==========================================================
   * CREATE / UPDATE WATCHLIST
   * ========================================================== */

  const submitList = async (
    event
  ) => {
    event.preventDefault();

    const name =
      String(
        listForm.name || ""
      ).trim();

    if (!name) {
      toast.error(
        "Watchlist name is required"
      );
      return;
    }

    setSaving(true);

    try {
      if (listForm.id) {
        await updateWatchlist(
          listForm.id,
          {
            ...listForm,
            name,
          }
        );

        toast.success(
          "Watchlist updated"
        );
      } else {
        await createWatchlist(
          {
            ...listForm,
            name,
          }
        );

        toast.success(
          "Watchlist created"
        );
      }

      setShowListForm(false);

      setListForm({
        ...emptyList,
      });

      await loadLists();
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Could not save watchlist"
        )
      );
    } finally {
      setSaving(false);
    }
  };


  /* ==========================================================
   * DELETE WATCHLIST
   * ========================================================== */

  const removeList = async () => {
    if (!selected) {
      return;
    }

    const confirmed =
      window.confirm(
        `Delete "${selected.name}" and all its entries?`
      );

    if (!confirmed) {
      return;
    }

    setDeletingList(true);

    try {
      await deleteWatchlist(
        selected.id
      );

      toast.success(
        "Watchlist deleted"
      );

      setSelectedId(null);

      await loadLists();
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Could not delete watchlist"
        )
      );
    } finally {
      setDeletingList(false);
    }
  };


  /* ==========================================================
   * CREATE / UPDATE ENTRY
   * ========================================================== */

  const submitEntry = async (
    event
  ) => {
    event.preventDefault();

    if (!selectedId) {
      toast.error(
        "Create or select a watchlist first"
      );
      return;
    }

    const plate =
      normalizePlate(
        entryForm.plate
      );

    if (!plate) {
      toast.error(
        "Registration number is required"
      );
      return;
    }

    setSaving(true);

    try {
      const payload = {
        ...entryForm,
        plate,
        category:
          String(
            entryForm.category ||
              "OTHER"
          ).toUpperCase(),
        status:
          String(
            entryForm.status ||
              "ACTIVE"
          ).toUpperCase(),
        priority:
          String(
            entryForm.priority ||
              "LOW"
          ).toUpperCase(),
      };

      if (editingEntry) {
        await updateWatchlistEntry(
          editingEntry.id,
          payload
        );

        toast.success(
          "Vehicle watchlist entry updated"
        );
      } else {
        await createWatchlistEntry(
          selectedId,
          payload
        );

        toast.success(
          "Vehicle added to watchlist"
        );
      }

      setShowEntryForm(false);

      setEditingEntry(null);

      setEntryForm({
        ...emptyEntry,
      });

      await loadEntries();
      await loadLists();
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Could not save vehicle entry"
        )
      );
    } finally {
      setSaving(false);
    }
  };


  /* ==========================================================
   * EDIT ENTRY
   * ========================================================== */

  const editEntry = (
    entry
  ) => {
    setEditingEntry(entry);

    setEntryForm({
      plate:
        entry.plate ||
        "",

      category:
        entry.category ||
        "SUSPICIOUS",

      status:
        entry.status ||
        "ACTIVE",

      priority:
        entry.priority ||
        "HIGH",

      description:
        entry.description ||
        "",

      source:
        entry.source ||
        "",
    });

    setActionMenuId(null);

    setShowEntryForm(true);
  };


  /* ==========================================================
   * PREVIEW ENTRY
   * ========================================================== */

  const previewEntry = (
    entry
  ) => {
    setSelectedEntry(
      entry
    );

    setActionMenuId(null);

    setShowEntryPreview(true);
  };


  /* ==========================================================
   * DELETE ENTRY
   * ========================================================== */

  const removeEntry = async (
    entry
  ) => {
    if (!entry?.id) {
      return;
    }

    const confirmed =
      window.confirm(
        `Remove ${entry.plate} from the watchlist?`
      );

    if (!confirmed) {
      return;
    }

    setDeletingEntryId(
      entry.id
    );

    try {
      await deleteWatchlistEntry(
        entry.id
      );

      toast.success(
        "Vehicle removed from watchlist"
      );

      await loadEntries();
      await loadLists();
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Could not remove vehicle entry"
        )
      );
    } finally {
      setDeletingEntryId(null);
    }
  };


  /* ==========================================================
   * TEST MATCH
   * ========================================================== */

  const testMatch = async (
    event
  ) => {
    event.preventDefault();

    const normalized =
      normalizePlate(
        testPlate
      );

    if (!normalized) {
      toast.error(
        "Enter a registration number"
      );
      return;
    }

    setMatchLoading(true);

    try {
      const result =
        await matchWatchlistPlate(
          normalized,
          true
        );

      setMatchResult(
        result
      );
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Watchlist lookup failed"
        )
      );
    } finally {
      setMatchLoading(false);
    }
  };


  /* ==========================================================
   * OPEN NEW WATCHLIST
   * ========================================================== */

  const openNewList = () => {
    setListForm({
      ...emptyList,
    });

    setShowListForm(true);
  };


  /* ==========================================================
   * OPEN EDIT WATCHLIST
   * ========================================================== */

  const openEditList = () => {
    if (!selected) {
      return;
    }

    setListForm({
      id: selected.id,
      name:
        selected.name ||
        "",
      description:
        selected.description ||
        "",
      is_active:
        Boolean(
          selected.is_active
        ),
    });

    setShowListForm(true);
  };


  /* ==========================================================
   * OPEN NEW ENTRY
   * ========================================================== */

  const openNewEntry = () => {
    if (!selectedId) {
      toast.error(
        "Create or select a watchlist first"
      );
      return;
    }

    setEditingEntry(null);

    setEntryForm({
      ...emptyEntry,
    });

    setShowEntryForm(true);
  };


  /* ==========================================================
   * CLEAR FILTERS
   * ========================================================== */

  const clearFilters = () => {
    setSearch("");
    setCategory("");
    setStatus("");
    setPriorityFilter("");
  };


  const hasFilters =
    Boolean(search) ||
    Boolean(category) ||
    Boolean(status) ||
    Boolean(priorityFilter);


  /* ==========================================================
   * RENDER
   * ========================================================== */

  return (
    <div className="min-h-full space-y-6 text-white">

      {/* ======================================================
       * PAGE HEADER
       * ====================================================== */}

      <div className="relative overflow-hidden rounded-2xl border border-slate-800/90 bg-gradient-to-br from-[#091426] via-[#07101d] to-[#050b14] px-5 py-6 shadow-[0_20px_60px_rgba(0,0,0,0.25)] sm:px-7">

        <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-blue-500/10 blur-3xl" />

        <div className="pointer-events-none absolute -bottom-24 left-1/3 h-56 w-56 rounded-full bg-cyan-500/5 blur-3xl" />

        <div className="relative flex flex-col gap-5 xl:flex-row xl:items-center xl:justify-between">

          <div>
            <div className="mb-3 flex items-center gap-2">
              <div className="rounded-lg border border-blue-400/20 bg-blue-400/10 p-2 text-blue-300">
                <Car size={19} />
              </div>

              <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-blue-300/80">
                Vehicle Intelligence
              </span>
            </div>

            <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
              Vehicle Watchlist
            </h1>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Manage stolen, blacklisted, suspicious and wanted
              vehicle registrations used for exact ANPR watchlist
              matching and persistent alerting.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={loadLists}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-3.5 py-2.5 text-sm font-medium text-slate-300 transition hover:border-slate-600 hover:bg-slate-800 disabled:opacity-50"
            >
              <RefreshCw
                size={16}
                className={
                  loading
                    ? "animate-spin"
                    : ""
                }
              />
              Refresh
            </button>

            <button
              type="button"
              onClick={openNewList}
              className="inline-flex items-center gap-2 rounded-lg border border-blue-400/30 bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-[0_8px_25px_rgba(59,130,246,0.18)] hover:bg-blue-500"
            >
              <Plus size={17} />
              New Watchlist
            </button>
          </div>
        </div>
      </div>


      {/* ======================================================
       * STATS
       * ====================================================== */}

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-5">
        <StatCard
          title="Entries"
          value={listStats.total}
          description="Current list entries"
          icon={List}
          iconClass="text-blue-300"
        />

        <StatCard
          title="Active"
          value={listStats.active}
          description="Operationally active"
          icon={Activity}
          iconClass="text-emerald-300"
        />

        <StatCard
          title="Critical"
          value={listStats.critical}
          description="Critical priority"
          icon={ShieldAlert}
          iconClass="text-red-300"
        />

        <StatCard
          title="High Priority"
          value={listStats.high}
          description="High priority"
          icon={AlertTriangle}
          iconClass="text-orange-300"
        />

        <StatCard
          title="Stolen"
          value={listStats.stolen}
          description="Stolen vehicle records"
          icon={Car}
          iconClass="text-purple-300"
        />
      </div>


      {/* ======================================================
       * WATCHLIST / ENTRY AREA
       * ====================================================== */}

      <div className="grid min-w-0 grid-cols-1 gap-5 xl:grid-cols-[330px_minmax(0,1fr)]">

        {/* WATCHLIST SIDEBAR */}

        <section className="rounded-2xl border border-slate-800 bg-[#07101b] p-4 shadow-[0_15px_50px_rgba(0,0,0,0.2)]">

          <div className="mb-4 flex items-center justify-between">
            <div>
              <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                Watchlists
              </div>

              <div className="mt-1 text-sm font-semibold text-slate-200">
                Vehicle collections
              </div>
            </div>

            <span className="rounded-full border border-slate-800 bg-slate-900 px-2.5 py-1 text-xs text-slate-500">
              {lists.length}
            </span>
          </div>


          {loading ? (
            <div className="flex min-h-[220px] items-center justify-center text-slate-500">
              <div className="text-center">
                <RefreshCw
                  size={22}
                  className="mx-auto mb-2 animate-spin text-blue-300"
                />
                <div className="text-xs">
                  Loading watchlists...
                </div>
              </div>
            </div>
          ) : lists.length === 0 ? (
            <div className="rounded-xl border border-dashed border-slate-800 bg-slate-950/50 p-7 text-center">
              <List
                size={28}
                className="mx-auto mb-3 text-slate-700"
              />

              <div className="text-sm font-medium text-slate-400">
                No watchlist created
              </div>

              <div className="mt-1 text-xs leading-5 text-slate-600">
                Create a vehicle watchlist to begin.
              </div>

              <button
                type="button"
                onClick={openNewList}
                className="mt-4 rounded-lg bg-blue-600 px-3.5 py-2 text-xs font-semibold text-white hover:bg-blue-500"
              >
                Create Watchlist
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              {lists.map((item) => {
                const isSelected =
                  Number(
                    selectedId
                  ) ===
                  Number(item.id);

                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() =>
                      setSelectedId(
                        item.id
                      )
                    }
                    className={`group w-full rounded-xl border p-3.5 text-left transition ${
                      isSelected
                        ? "border-blue-500/50 bg-blue-500/[0.08] shadow-[0_8px_25px_rgba(37,99,235,0.08)]"
                        : "border-slate-800 bg-slate-950/40 hover:bg-slate-900/70"
                    }`}
                  >
                    <div className="flex items-start justify-between gap-3">

                      <div className="min-w-0">
                        <div className="truncate text-sm font-semibold text-slate-200">
                          {item.name}
                        </div>

                        <div className="mt-1 text-xs text-slate-600">
                          {item.entry_count || 0} vehicle entries
                        </div>
                      </div>

                      <span
                        className={`shrink-0 rounded-full px-2 py-1 text-[9px] font-semibold ${
                          item.is_active
                            ? "bg-emerald-500/10 text-emerald-300"
                            : "bg-slate-700 text-slate-400"
                        }`}
                      >
                        {item.is_active
                          ? "ACTIVE"
                          : "OFF"}
                      </span>
                    </div>

                    <div className="mt-3 flex items-center justify-between text-[10px] text-slate-600">
                      <span>
                        List ID #{item.id}
                      </span>

                      <ChevronRight
                        size={14}
                        className={
                          isSelected
                            ? "text-blue-400"
                            : "text-slate-700"
                        }
                      />
                    </div>
                  </button>
                );
              })}
            </div>
          )}


          {selected && (
            <div className="mt-4 border-t border-slate-800 pt-4">

              <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">
                Selected Watchlist
              </div>

              <div className="rounded-xl border border-slate-800 bg-slate-950/50 p-3">
                <div className="text-sm font-semibold text-slate-200">
                  {selected.name}
                </div>

                {selected.description && (
                  <div className="mt-1 text-xs leading-5 text-slate-500">
                    {selected.description}
                  </div>
                )}

                <div className="mt-3 flex gap-2">
                  <button
                    type="button"
                    onClick={openEditList}
                    className="flex-1 rounded-lg border border-slate-700 bg-slate-900 px-3 py-2 text-xs font-medium text-slate-300 hover:bg-slate-800"
                  >
                    <Edit3
                      size={14}
                      className="mr-1 inline"
                    />
                    Edit
                  </button>

                  <button
                    type="button"
                    onClick={removeList}
                    disabled={deletingList}
                    className="rounded-lg border border-red-500/20 bg-red-500/5 px-3 py-2 text-red-300 hover:bg-red-500/10 disabled:opacity-50"
                  >
                    {deletingList ? (
                      <RefreshCw
                        size={14}
                        className="animate-spin"
                      />
                    ) : (
                      <Trash2 size={14} />
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}
        </section>


        {/* ENTRIES */}

        <section className="min-w-0 rounded-2xl border border-slate-800 bg-[#07101b] p-4 shadow-[0_15px_50px_rgba(0,0,0,0.2)]">

          <div className="mb-4 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">

            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-lg font-semibold text-slate-100">
                  {selected?.name ||
                    "Select a watchlist"}
                </h2>

                {selected && (
                  <span className="rounded-full border border-slate-800 bg-slate-950 px-2 py-0.5 text-[10px] text-slate-500">
                    {entries.length} records
                  </span>
                )}
              </div>

              <p className="mt-1 text-xs leading-5 text-slate-600">
                Exact ANPR matches generate persistent alerts.
                Possible matches remain non-alerting until confirmed.
              </p>
            </div>

            {selected && (
              <button
                type="button"
                onClick={openNewEntry}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-500"
              >
                <Plus size={17} />
                Add Vehicle
              </button>
            )}
          </div>


          {!selected ? (
            <div className="flex min-h-[420px] items-center justify-center rounded-xl border border-dashed border-slate-800 bg-slate-950/40 text-center">
              <div>
                <Car
                  size={34}
                  className="mx-auto mb-3 text-slate-700"
                />

                <div className="text-sm font-semibold text-slate-400">
                  Select a vehicle watchlist
                </div>

                <div className="mt-1 text-xs text-slate-600">
                  Choose a watchlist from the left panel.
                </div>
              </div>
            </div>
          ) : (
            <>
              {/* FILTER BAR */}

              <div className="mb-4 rounded-xl border border-slate-800 bg-slate-950/40 p-3">
                <div className="grid grid-cols-1 gap-2 xl:grid-cols-[minmax(0,1fr)_160px_150px_150px_auto]">

                  <label className="relative">
                    <Search
                      size={16}
                      className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-600"
                    />

                    <input
                      value={search}
                      onChange={(event) =>
                        setSearch(
                          event.target.value
                        )
                      }
                      placeholder="Search registration, source or description..."
                      className="w-full rounded-lg border border-slate-800 bg-[#08111d] py-2.5 pl-9 pr-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/50"
                    />
                  </label>


                  <select
                    value={category}
                    onChange={(event) =>
                      setCategory(
                        event.target.value
                      )
                    }
                    className="rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/50"
                  >
                    <option value="">
                      All categories
                    </option>

                    {CATEGORIES.map(
                      (value) => (
                        <option
                          key={value}
                          value={value}
                        >
                          {value}
                        </option>
                      )
                    )}
                  </select>


                  <select
                    value={priorityFilter}
                    onChange={(event) =>
                      setPriorityFilter(
                        event.target.value
                      )
                    }
                    className="rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/50"
                  >
                    <option value="">
                      All priorities
                    </option>

                    {PRIORITIES.map(
                      (value) => (
                        <option
                          key={value}
                          value={value}
                        >
                          {value}
                        </option>
                      )
                    )}
                  </select>


                  <select
                    value={status}
                    onChange={(event) =>
                      setStatus(
                        event.target.value
                      )
                    }
                    className="rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/50"
                  >
                    <option value="">
                      All statuses
                    </option>

                    {STATUSES.map(
                      (value) => (
                        <option
                          key={value}
                          value={value}
                        >
                          {value}
                        </option>
                      )
                    )}
                  </select>


                  {hasFilters ? (
                    <button
                      type="button"
                      onClick={clearFilters}
                      className="rounded-lg border border-slate-800 bg-slate-900 px-3 text-xs font-medium text-slate-400 hover:bg-slate-800 hover:text-white"
                    >
                      Clear
                    </button>
                  ) : (
                    <div className="hidden xl:block" />
                  )}
                </div>
              </div>


              {/* TABLE */}

              <div className="overflow-hidden rounded-xl border border-slate-800">

                <div className="overflow-x-auto">
                  <table className="min-w-[1050px] w-full text-sm">

                    <thead className="bg-slate-900/80">
                      <tr className="border-b border-slate-800 text-left text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                        <th className="px-4 py-3.5">
                          Vehicle
                        </th>

                        <th className="px-4 py-3.5">
                          Classification
                        </th>

                        <th className="px-4 py-3.5">
                          Priority
                        </th>

                        <th className="px-4 py-3.5">
                          Status
                        </th>

                        <th className="px-4 py-3.5">
                          Case / Description
                        </th>

                        <th className="px-4 py-3.5">
                          Source
                        </th>

                        <th className="px-4 py-3.5 text-right">
                          Actions
                        </th>
                      </tr>
                    </thead>

                    <tbody className="divide-y divide-slate-800/80">

                      {entriesLoading ? (
                        <tr>
                          <td
                            colSpan="7"
                            className="p-10 text-center"
                          >
                            <RefreshCw
                              size={24}
                              className="mx-auto mb-3 animate-spin text-blue-300"
                            />

                            <div className="text-xs text-slate-500">
                              Loading vehicle intelligence...
                            </div>
                          </td>
                        </tr>
                      ) : entries.length === 0 ? (
                        <tr>
                          <td
                            colSpan="7"
                            className="p-10 text-center"
                          >
                            <Car
                              size={30}
                              className="mx-auto mb-3 text-slate-700"
                            />

                            <div className="text-sm font-medium text-slate-500">
                              No vehicle entries found
                            </div>

                            <div className="mt-1 text-xs text-slate-700">
                              Adjust filters or add a vehicle.
                            </div>
                          </td>
                        </tr>
                      ) : (
                        entries.map(
                          (entry) => {
                            const categoryMeta =
                              getCategoryMeta(
                                entry.category
                              );

                            const priorityMeta =
                              getPriorityMeta(
                                entry.priority
                              );

                            const statusMeta =
                              getStatusMeta(
                                entry.status
                              );

                            return (
                              <tr
                                key={entry.id}
                                className="group hover:bg-white/[0.02]"
                              >

                                {/* VEHICLE */}

                                <td className="px-4 py-4 align-top">
                                  <button
                                    type="button"
                                    onClick={() =>
                                      previewEntry(
                                        entry
                                      )
                                    }
                                    className="group/plate flex items-start gap-3 text-left"
                                  >
                                    <div className="flex h-11 w-14 shrink-0 items-center justify-center rounded-lg border border-slate-700 bg-slate-950 text-slate-400 shadow-inner transition group-hover/plate:border-blue-500/40">
                                      <Car
                                        size={20}
                                      />
                                    </div>

                                    <div>
                                      <div className="font-mono text-base font-bold tracking-wide text-slate-100 group-hover/plate:text-blue-300">
                                        {entry.plate}
                                      </div>

                                      <div className="mt-1 text-[10px] uppercase tracking-[0.13em] text-slate-600">
                                        Vehicle ID #{entry.id}
                                      </div>
                                    </div>
                                  </button>
                                </td>


                                {/* CATEGORY */}

                                <td className="px-4 py-4 align-top">
                                  <Badge
                                    meta={
                                      categoryMeta
                                    }
                                  />
                                </td>


                                {/* PRIORITY */}

                                <td className="px-4 py-4 align-top">
                                  <Badge
                                    meta={
                                      priorityMeta
                                    }
                                  />
                                </td>


                                {/* STATUS */}

                                <td className="px-4 py-4 align-top">
                                  <Badge
                                    meta={
                                      statusMeta
                                    }
                                  />
                                </td>


                                {/* DESCRIPTION */}

                                <td className="max-w-[340px] px-4 py-4 align-top">
                                  <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-slate-600">
                                    Case / Intelligence
                                  </div>

                                  <div className="line-clamp-3 text-sm leading-5 text-slate-400">
                                    {entry.description ||
                                      "No case description provided."}
                                  </div>
                                </td>


                                {/* SOURCE */}

                                <td className="px-4 py-4 align-top">
                                  <div className="flex items-start gap-2">
                                    <FileText
                                      size={14}
                                      className="mt-0.5 shrink-0 text-slate-600"
                                    />

                                    <div>
                                      <div className="max-w-[140px] truncate text-sm text-slate-300">
                                        {entry.source ||
                                          "—"}
                                      </div>

                                      <div className="mt-1 text-[10px] text-slate-600">
                                        Department / source
                                      </div>
                                    </div>
                                  </div>
                                </td>


                                {/* ACTIONS */}

                                <td className="px-4 py-4 align-top text-right">
                                  <div className="relative inline-flex items-center gap-1">

                                    <button
                                      type="button"
                                      onClick={() =>
                                        previewEntry(
                                          entry
                                        )
                                      }
                                      className="rounded-lg p-2 text-cyan-300 hover:bg-cyan-500/10"
                                      title="Preview"
                                    >
                                      <Eye
                                        size={16}
                                      />
                                    </button>

                                    <button
                                      type="button"
                                      onClick={() =>
                                        editEntry(
                                          entry
                                        )
                                      }
                                      className="rounded-lg p-2 text-amber-300 hover:bg-amber-500/10"
                                      title="Edit"
                                    >
                                      <Edit3
                                        size={16}
                                      />
                                    </button>

                                    <button
                                      type="button"
                                      onClick={() =>
                                        setActionMenuId(
                                          actionMenuId ===
                                            entry.id
                                            ? null
                                            : entry.id
                                        )
                                      }
                                      className="rounded-lg p-2 text-slate-400 hover:bg-slate-800 hover:text-white"
                                      title="More actions"
                                    >
                                      <MoreVertical
                                        size={16}
                                      />
                                    </button>

                                    {actionMenuId ===
                                      entry.id && (
                                      <div className="absolute right-0 top-10 z-30 w-48 overflow-hidden rounded-xl border border-slate-700 bg-[#08111d] p-1.5 text-left shadow-[0_20px_60px_rgba(0,0,0,0.45)]">

                                        <button
                                          type="button"
                                          onClick={() =>
                                            previewEntry(
                                              entry
                                            )
                                          }
                                          className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-xs text-slate-300 hover:bg-white/[0.04] hover:text-white"
                                        >
                                          <Eye
                                            size={14}
                                          />
                                          View details
                                        </button>

                                        <button
                                          type="button"
                                          onClick={() =>
                                            editEntry(
                                              entry
                                            )
                                          }
                                          className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-xs text-slate-300 hover:bg-white/[0.04] hover:text-white"
                                        >
                                          <Edit3
                                            size={14}
                                          />
                                          Edit vehicle
                                        </button>

                                        <button
                                          type="button"
                                          onClick={() =>
                                            removeEntry(
                                              entry
                                            )
                                          }
                                          disabled={
                                            deletingEntryId ===
                                            entry.id
                                          }
                                          className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-xs text-red-300 hover:bg-red-500/10 disabled:opacity-50"
                                        >
                                          {deletingEntryId ===
                                          entry.id ? (
                                            <RefreshCw
                                              size={14}
                                              className="animate-spin"
                                            />
                                          ) : (
                                            <Trash2
                                              size={14}
                                            />
                                          )}

                                          Remove entry
                                        </button>
                                      </div>
                                    )}

                                    <button
                                      type="button"
                                      onClick={() =>
                                        removeEntry(
                                          entry
                                        )
                                      }
                                      disabled={
                                        deletingEntryId ===
                                        entry.id
                                      }
                                      className="rounded-lg p-2 text-red-300 hover:bg-red-500/10 disabled:opacity-50"
                                      title="Remove entry"
                                    >
                                      {deletingEntryId ===
                                      entry.id ? (
                                        <RefreshCw
                                          size={16}
                                          className="animate-spin"
                                        />
                                      ) : (
                                        <Trash2
                                          size={16}
                                        />
                                      )}
                                    </button>

                                  </div>
                                </td>
                              </tr>
                            );
                          }
                        )
                      )}

                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </section>
      </div>


      {/* ======================================================
       * TEST MATCH
       * ====================================================== */}

      <section className="rounded-2xl border border-slate-800 bg-[#07101b] p-5 shadow-[0_15px_50px_rgba(0,0,0,0.2)]">

        <div className="mb-4 flex items-start gap-3">
          <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2.5 text-cyan-300">
            <Search size={18} />
          </div>

          <div>
            <h2 className="text-base font-semibold text-slate-200">
              Test Vehicle Watchlist Match
            </h2>

            <p className="mt-1 text-xs leading-5 text-slate-600">
              Test exact registration matching before
              integrating a live ANPR feed.
            </p>
          </div>
        </div>


        <form
          onSubmit={testMatch}
          className="flex max-w-3xl flex-col gap-2 sm:flex-row"
        >
          <div className="relative flex-1">
            <Car
              size={16}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-600"
            />

            <input
              value={testPlate}
              onChange={(event) =>
                setTestPlate(
                  event.target.value.toUpperCase()
                )
              }
              placeholder="Enter registration e.g. KA01AB1234"
              className="w-full rounded-lg border border-slate-800 bg-[#08111d] py-2.5 pl-9 pr-3 font-mono text-sm text-white outline-none placeholder:font-sans placeholder:text-slate-600 focus:border-blue-500/50"
            />
          </div>

          <button
            type="submit"
            disabled={matchLoading}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-800 px-5 py-2.5 text-sm font-semibold text-slate-200 hover:bg-slate-700 disabled:opacity-50"
          >
            {matchLoading ? (
              <RefreshCw
                size={15}
                className="animate-spin"
              />
            ) : (
              <Search size={15} />
            )}

            {matchLoading
              ? "Checking..."
              : "Check Vehicle"}
          </button>
        </form>


        {matchResult && (
          <div className="mt-5 border-t border-slate-800 pt-5">

            <div className="mb-3 flex items-center justify-between gap-3">
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">
                  Match Result
                </div>

                <div className="mt-1 text-sm text-slate-300">
                  Normalized registration{" "}
                  <span className="font-mono font-semibold text-white">
                    {matchResult.plate_normalized}
                  </span>
                </div>
              </div>

              <button
                type="button"
                onClick={() =>
                  setMatchResult(null)
                }
                className="rounded-lg p-2 text-slate-600 hover:bg-slate-800 hover:text-white"
                title="Close result"
              >
                <X size={16} />
              </button>
            </div>


            {matchResult.exact_matches?.length > 0 ? (
              <div className="space-y-2">
                {matchResult.exact_matches.map(
                  (match) => (
                    <button
                      key={
                        match.entry.id
                      }
                      type="button"
                      onClick={() =>
                        previewEntry(
                          match.entry
                        )
                      }
                      className="w-full rounded-xl border border-red-500/20 bg-red-500/[0.05] p-4 text-left transition hover:bg-red-500/[0.09]"
                    >
                      <div className="flex items-center gap-3">
                        <div className="rounded-lg bg-red-500/10 p-2 text-red-300">
                          <CheckCircle2
                            size={18}
                          />
                        </div>

                        <div className="min-w-0 flex-1">
                          <div className="text-sm font-semibold text-red-200">
                            EXACT MATCH
                          </div>

                          <div className="mt-1 flex flex-wrap items-center gap-2 text-xs text-slate-500">
                            <span className="font-mono text-slate-200">
                              {match.entry.plate}
                            </span>

                            <span>
                              •
                            </span>

                            <span>
                              {match.entry.category}
                            </span>

                            <span>
                              •
                            </span>

                            <span>
                              {match.entry.priority}
                            </span>

                            <span>
                              •
                            </span>

                            <span>
                              confidence{" "}
                              {(Number(
                                match.confidence
                              ) * 100).toFixed(1)}
                              %
                            </span>
                          </div>
                        </div>

                        <ChevronRight
                          size={16}
                          className="text-slate-600"
                        />
                      </div>
                    </button>
                  )
                )}
              </div>
            ) : (
              <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/[0.04] p-4">
                <div className="flex items-center gap-2 text-sm font-semibold text-emerald-300">
                  <CheckCircle2
                    size={17}
                  />
                  No exact active watchlist match
                </div>

                <div className="mt-1 text-xs leading-5 text-slate-600">
                  No automatic watchlist alert would be created
                  from this exact-match result.
                </div>
              </div>
            )}


            {matchResult.possible_matches?.length > 0 && (
              <div className="mt-3 space-y-2">
                <div className="text-[10px] font-semibold uppercase tracking-[0.15em] text-slate-600">
                  Possible Matches
                </div>

                {matchResult.possible_matches.map(
                  (match) => (
                    <button
                      key={
                        match.entry.id
                      }
                      type="button"
                      onClick={() =>
                        previewEntry(
                          match.entry
                        )
                      }
                      className="w-full rounded-xl border border-amber-500/20 bg-amber-500/[0.03] p-3 text-left hover:bg-amber-500/[0.06]"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <span className="text-xs font-semibold text-amber-300">
                            POSSIBLE MATCH
                          </span>

                          <span className="ml-2 font-mono text-sm text-slate-300">
                            {match.entry.plate}
                          </span>

                          <div className="mt-1 text-[11px] text-slate-600">
                            {(Number(
                              match.confidence
                            ) * 100).toFixed(1)}
                            % similarity • does not create an automatic alert
                          </div>
                        </div>

                        <ChevronRight
                          size={15}
                          className="text-slate-700"
                        />
                      </div>
                    </button>
                  )
                )}
              </div>
            )}
          </div>
        )}
      </section>


      {/* ======================================================
       * WATCHLIST MODAL
       * ====================================================== */}

      {showListForm && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm"
          onMouseDown={(event) => {
            if (
              event.target ===
                event.currentTarget &&
              !saving
            ) {
              setShowListForm(false);
            }
          }}
        >
          <form
            onSubmit={submitList}
            className="w-full max-w-xl rounded-2xl border border-slate-700 bg-[#07101b] p-6 shadow-[0_30px_100px_rgba(0,0,0,0.55)]"
          >
            <div className="mb-5 flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-lg font-semibold text-white">
                  {listForm.id
                    ? "Edit Watchlist"
                    : "Create Vehicle Watchlist"}
                </h2>

                <p className="mt-1 text-xs text-slate-500">
                  Configure the operational vehicle collection.
                </p>
              </div>

              <button
                type="button"
                onClick={() =>
                  setShowListForm(false)
                }
                disabled={saving}
                className="rounded-lg p-2 text-slate-500 hover:bg-slate-800 hover:text-white"
              >
                <X size={18} />
              </button>
            </div>


            <div className="space-y-4">

              <div>
                <SectionLabel>
                  Watchlist Name
                </SectionLabel>

                <input
                  required
                  maxLength={120}
                  value={listForm.name}
                  onChange={(event) =>
                    setListForm(
                      {
                        ...listForm,
                        name: event.target.value,
                      }
                    )
                  }
                  placeholder="Police Vehicle Watchlist"
                  disabled={saving}
                  className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-2.75 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/50"
                />
              </div>


              <div>
                <SectionLabel>
                  Description
                </SectionLabel>

                <textarea
                  rows={4}
                  value={
                    listForm.description
                  }
                  onChange={(event) =>
                    setListForm(
                      {
                        ...listForm,
                        description:
                          event.target.value,
                      }
                    )
                  }
                  placeholder="Describe the operational purpose of this watchlist..."
                  disabled={saving}
                  className="w-full resize-none rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-3 text-sm leading-6 text-white outline-none placeholder:text-slate-600 focus:border-blue-500/50"
                />
              </div>


              <label className="flex cursor-pointer items-center gap-3 rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                <input
                  type="checkbox"
                  checked={
                    Boolean(
                      listForm.is_active
                    )
                  }
                  onChange={(event) =>
                    setListForm(
                      {
                        ...listForm,
                        is_active:
                          event.target.checked,
                      }
                    )
                  }
                  disabled={saving}
                  className="h-4 w-4 rounded border-slate-700 bg-slate-900"
                />

                <div>
                  <div className="text-sm font-medium text-slate-300">
                    Watchlist active
                  </div>

                  <div className="mt-0.5 text-xs text-slate-600">
                    Active lists participate in operational matching.
                  </div>
                </div>

                <Power
                  size={16}
                  className="ml-auto text-emerald-300"
                />
              </label>


              <div className="flex justify-end gap-2 border-t border-slate-800 pt-4">

                <button
                  type="button"
                  onClick={() =>
                    setShowListForm(false)
                  }
                  disabled={saving}
                  className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-slate-300 hover:bg-slate-800"
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  disabled={saving}
                  className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-50"
                >
                  {saving ? (
                    <RefreshCw
                      size={15}
                      className="animate-spin"
                    />
                  ) : (
                    <Save size={15} />
                  )}

                  {saving
                    ? "Saving..."
                    : listForm.id
                    ? "Save Changes"
                    : "Create Watchlist"}
                </button>
              </div>
            </div>
          </form>
        </div>
      )}


      {/* ======================================================
       * ENTRY MODAL
       * ====================================================== */}

      {showEntryForm && (
        <div
          className="fixed inset-0 z-[110] flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm"
          onMouseDown={(event) => {
            if (
              event.target ===
                event.currentTarget &&
              !saving
            ) {
              setShowEntryForm(false);
            }
          }}
        >
          <form
            onSubmit={submitEntry}
            className="w-full max-w-2xl rounded-2xl border border-slate-700 bg-[#07101b] p-6 shadow-[0_30px_100px_rgba(0,0,0,0.55)]"
          >

            <div className="mb-5 flex items-center justify-between border-b border-slate-800 pb-4">
              <div>
                <h2 className="text-lg font-semibold text-white">
                  {editingEntry
                    ? "Edit Vehicle Watchlist Entry"
                    : "Add Vehicle to Watchlist"}
                </h2>

                <p className="mt-1 text-xs text-slate-500">
                  Store a registration number with its operational
                  classification and case intelligence.
                </p>
              </div>

              <button
                type="button"
                onClick={() =>
                  setShowEntryForm(false)
                }
                disabled={saving}
                className="rounded-lg p-2 text-slate-500 hover:bg-slate-800 hover:text-white"
              >
                <X size={18} />
              </button>
            </div>


            <div className="space-y-4">

              <div>
                <SectionLabel>
                  Registration Number
                </SectionLabel>

                <input
                  required
                  maxLength={32}
                  value={
                    entryForm.plate
                  }
                  onChange={(event) =>
                    setEntryForm(
                      {
                        ...entryForm,
                        plate:
                          event.target.value.toUpperCase(),
                      }
                    )
                  }
                  placeholder="KA01AB1234"
                  disabled={saving}
                  className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-3 font-mono text-base font-semibold tracking-wider text-white outline-none placeholder:font-sans placeholder:font-normal placeholder:text-slate-600 focus:border-blue-500/50"
                />
              </div>


              <div className="grid grid-cols-1 gap-4 md:grid-cols-3">

                <div>
                  <SectionLabel>
                    Category
                  </SectionLabel>

                  <select
                    value={
                      entryForm.category
                    }
                    onChange={(event) =>
                      setEntryForm(
                        {
                          ...entryForm,
                          category:
                            event.target.value,
                        }
                      )
                    }
                    disabled={saving}
                    className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.75 text-sm text-slate-300 outline-none focus:border-blue-500/50"
                  >
                    {CATEGORIES.map(
                      (value) => (
                        <option
                          key={value}
                          value={value}
                        >
                          {value}
                        </option>
                      )
                    )}
                  </select>
                </div>


                <div>
                  <SectionLabel>
                    Priority
                  </SectionLabel>

                  <select
                    value={
                      entryForm.priority
                    }
                    onChange={(event) =>
                      setEntryForm(
                        {
                          ...entryForm,
                          priority:
                            event.target.value,
                        }
                      )
                    }
                    disabled={saving}
                    className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.75 text-sm text-slate-300 outline-none focus:border-blue-500/50"
                  >
                    {PRIORITIES.map(
                      (value) => (
                        <option
                          key={value}
                          value={value}
                        >
                          {value}
                        </option>
                      )
                    )}
                  </select>
                </div>


                <div>
                  <SectionLabel>
                    Status
                  </SectionLabel>

                  <select
                    value={
                      entryForm.status
                    }
                    onChange={(event) =>
                      setEntryForm(
                        {
                          ...entryForm,
                          status:
                            event.target.value,
                        }
                      )
                    }
                    disabled={saving}
                    className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.75 text-sm text-slate-300 outline-none focus:border-blue-500/50"
                  >
                    {STATUSES.map(
                      (value) => (
                        <option
                          key={value}
                          value={value}
                        >
                          {value}
                        </option>
                      )
                    )}
                  </select>
                </div>
              </div>


              <div>
                <SectionLabel>
                  Source / Department
                </SectionLabel>

                <input
                  maxLength={150}
                  value={
                    entryForm.source
                  }
                  onChange={(event) =>
                    setEntryForm(
                      {
                        ...entryForm,
                        source:
                          event.target.value,
                      }
                    )
                  }
                  placeholder="Police station / department / source"
                  disabled={saving}
                  className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-2.75 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/50"
                />
              </div>


              <div>
                <SectionLabel>
                  Case / Intelligence Description
                </SectionLabel>

                <textarea
                  rows={6}
                  maxLength={5000}
                  value={
                    entryForm.description
                  }
                  onChange={(event) =>
                    setEntryForm(
                      {
                        ...entryForm,
                        description:
                          event.target.value,
                      }
                    )
                  }
                  placeholder="Crime, FIR, vehicle identifiers, incident notes or other relevant intelligence..."
                  disabled={saving}
                  className="w-full resize-none rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-3 text-sm leading-6 text-white outline-none placeholder:text-slate-600 focus:border-blue-500/50"
                />

                <div className="mt-1 text-right text-[10px] text-slate-600">
                  {String(
                    entryForm.description ||
                      ""
                  ).length}
                  /5000
                </div>
              </div>


              <div className="flex items-center gap-2 rounded-xl border border-blue-500/10 bg-blue-500/[0.03] p-3">
                <AlertTriangle
                  size={15}
                  className="shrink-0 text-blue-300"
                />

                <div className="text-[11px] leading-5 text-slate-500">
                  Exact plate matches may generate operational
                  alerts. Verify case data and classification before
                  activating a high-priority record.
                </div>
              </div>


              <div className="flex justify-end gap-2 border-t border-slate-800 pt-4">

                <button
                  type="button"
                  onClick={() =>
                    setShowEntryForm(false)
                  }
                  disabled={saving}
                  className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm text-slate-300 hover:bg-slate-800"
                >
                  Cancel
                </button>

                <button
                  type="submit"
                  disabled={saving}
                  className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-50"
                >
                  {saving ? (
                    <RefreshCw
                      size={15}
                      className="animate-spin"
                    />
                  ) : (
                    <Save size={15} />
                  )}

                  {saving
                    ? "Saving..."
                    : editingEntry
                    ? "Save Changes"
                    : "Add Vehicle"}
                </button>
              </div>
            </div>
          </form>
        </div>
      )}


      {/* ======================================================
       * ENTRY PREVIEW
       * ====================================================== */}

      <WatchlistActiveJourneys
        identityType="VEHICLE"
        entryIds={entries.map((entry) => entry.id)}
      />

      {showEntryPreview && selectedEntry && (
          <div
            className="fixed inset-0 z-[120] flex items-center justify-center bg-black/80 p-4 backdrop-blur-md"
            onMouseDown={(event) => {
              if (
                event.target ===
                event.currentTarget
              ) {
                setShowEntryPreview(
                  false
                );
              }
            }}
          >
            <div className="w-full max-w-3xl overflow-hidden rounded-2xl border border-slate-700 bg-[#07101b] shadow-[0_30px_120px_rgba(0,0,0,0.55)]">

              <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
                <div>
                  <div className="flex items-center gap-2">
                    <div className="rounded-lg border border-blue-500/20 bg-blue-500/10 p-2 text-blue-300">
                      <Car size={17} />
                    </div>

                    <div>
                      <h2 className="font-mono text-lg font-bold tracking-wide text-white">
                        {selectedEntry.plate}
                      </h2>

                      <p className="text-xs text-slate-500">
                        Vehicle watchlist record
                      </p>
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() =>
                    setShowEntryPreview(
                      false
                    )
                  }
                  className="rounded-lg p-2 text-slate-500 hover:bg-slate-800 hover:text-white"
                >
                  <X size={18} />
                </button>
              </div>


              <div className="grid gap-0 lg:grid-cols-[1fr_320px]">

                <div className="bg-black p-6">
                  <div className="flex min-h-[300px] items-center justify-center rounded-xl border border-slate-800 bg-[#03070c]">
                    <div className="text-center">
                      <div className="mx-auto mb-4 flex h-20 w-20 items-center justify-center rounded-2xl border border-blue-500/20 bg-blue-500/10 text-blue-300">
                        <Car size={38} />
                      </div>

                      <div className="font-mono text-2xl font-bold tracking-[0.18em] text-white">
                        {selectedEntry.plate}
                      </div>

                      <div className="mt-2 text-xs uppercase tracking-[0.18em] text-slate-600">
                        ANPR Watchlist Identifier
                      </div>
                    </div>
                  </div>
                </div>


                <div className="border-t border-slate-800 p-5 lg:border-l lg:border-t-0">
                  <div className="space-y-5">

                    <div className="flex flex-wrap gap-2">
                      <Badge
                        meta={getCategoryMeta(
                          selectedEntry.category
                        )}
                      />

                      <Badge
                        meta={getPriorityMeta(
                          selectedEntry.priority
                        )}
                      />

                      <Badge
                        meta={getStatusMeta(
                          selectedEntry.status
                        )}
                      />
                    </div>


                    <div>
                      <SectionLabel>
                        Source / Department
                      </SectionLabel>

                      <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3 text-sm text-slate-300">
                        {selectedEntry.source ||
                          "—"}
                      </div>
                    </div>


                    <div>
                      <SectionLabel>
                        Description / Intelligence
                      </SectionLabel>

                      <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3 text-sm leading-6 text-slate-300">
                        {selectedEntry.description ||
                          "No case description provided."}
                      </div>
                    </div>


                    <div className="rounded-xl border border-blue-500/10 bg-blue-500/[0.03] p-3">
                      <div className="flex items-start gap-2">
                        <ShieldAlert
                          size={15}
                          className="mt-0.5 shrink-0 text-blue-300"
                        />

                        <div className="text-[11px] leading-5 text-slate-500">
                          Exact registration matches can be
                          correlated with live ANPR detections
                          and converted into persistent alerts.
                        </div>
                      </div>
                    </div>


                    <div className="grid grid-cols-2 gap-2 border-t border-slate-800 pt-4">

                      <button
                        type="button"
                        onClick={() => {
                          setShowEntryPreview(
                            false
                          );

                          editEntry(
                            selectedEntry
                          );
                        }}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2.5 text-sm font-medium text-amber-200 hover:bg-amber-500/15"
                      >
                        <Edit3 size={15} />
                        Edit
                      </button>

                      <button
                        type="button"
                        onClick={() =>
                          removeEntry(
                            selectedEntry
                          )
                        }
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2.5 text-sm font-medium text-red-200 hover:bg-red-500/15"
                      >
                        <Trash2 size={15} />
                        Remove
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
    </div>
  );
};


export default Watchlist;
