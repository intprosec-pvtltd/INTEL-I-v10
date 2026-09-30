import React, {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import toast from "react-hot-toast";
import WatchlistActiveJourneys from "../components/WatchlistActiveJourneys.jsx";
import { formatIndianDateTime as formatDate } from "../utils/dateTime";

import {
  Plus,
  RefreshCw,
  UserRound,
  Trash2,
  X,
  Eye,
  Pencil,
  Upload,
  Save,
  Image as ImageIcon,
  Search,
  Filter,
  ShieldAlert,
  UserX,
  Users,
  Clock3,
  ChevronDown,
  MoreVertical,
  FileText,
  MapPin,
  Fingerprint,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";

import {
  getPersonWatchlist,
  enrollPersonWatchlist,
  deletePersonWatchlist,
  updatePersonWatchlist,
  getPersonWatchlistImage,
  uploadPersonWatchlistImage,
} from "../api/axios.js";


/* ============================================================
 * CONSTANTS
 * ============================================================ */

const MAX_IMAGE_SIZE = 10 * 1024 * 1024;

const initialForm = {
  full_name: "",
  category: "MISSING",
  status: "ACTIVE",
  description: "",
  source: "",
  external_reference: "",
  image: null,
};


/* ============================================================
 * HELPERS
 * ============================================================ */

const getErrorMessage = (
  error,
  fallback = "Request failed"
) => {
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
          const location = Array.isArray(item.loc)
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

  if (detail && typeof detail === "object") {
    if (typeof detail.msg === "string") {
      return detail.msg;
    }

    try {
      return JSON.stringify(detail);
    } catch {
      return fallback;
    }
  }

  if (typeof error?.message === "string") {
    return error.message;
  }

  return fallback;
};


const revokeObjectUrl = (url) => {
  if (
    typeof url === "string" &&
    url.startsWith("blob:")
  ) {
    URL.revokeObjectURL(url);
  }
};


const validateImage = (file) => {
  if (!file) {
    return "Reference image is required.";
  }

  const allowedTypes = [
    "image/jpeg",
    "image/png",
    "image/webp",
  ];

  const allowedExtensions = [
    "jpg",
    "jpeg",
    "png",
    "webp",
  ];

  const extension = (
    file.name?.split(".").pop() || ""
  ).toLowerCase();

  if (
    !allowedTypes.includes(file.type) &&
    !allowedExtensions.includes(extension)
  ) {
    return "Only JPG, JPEG, PNG and WEBP images are allowed.";
  }

  if (!Number.isFinite(file.size) || file.size <= 0) {
    return "Selected image is empty.";
  }

  if (file.size > MAX_IMAGE_SIZE) {
    return "Image must not exceed 10 MB.";
  }

  return null;
};


const getCategoryMeta = (category) => {
  const value = String(category || "OTHER").toUpperCase();

  if (value === "WANTED") {
    return {
      label: "WANTED",
      icon: ShieldAlert,
      badge: "border-red-500/30 bg-red-500/10 text-red-300",
      dot: "bg-red-400",
    };
  }

  if (value === "MISSING") {
    return {
      label: "MISSING",
      icon: UserX,
      badge: "border-amber-500/30 bg-amber-500/10 text-amber-300",
      dot: "bg-amber-400",
    };
  }

  return {
    label: "OTHER",
    icon: Users,
    badge: "border-slate-500/30 bg-slate-500/10 text-slate-300",
    dot: "bg-slate-400",
  };
};


const getStatusMeta = (status) => {
  const value = String(status || "UNKNOWN").toUpperCase();

  if (value === "ACTIVE") {
    return {
      label: "ACTIVE",
      badge: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
      dot: "bg-emerald-400",
    };
  }

  if (value === "CLEARED") {
    return {
      label: "CLEARED",
      badge: "border-cyan-500/30 bg-cyan-500/10 text-cyan-300",
      dot: "bg-cyan-400",
    };
  }

  if (value === "EXPIRED") {
    return {
      label: "EXPIRED",
      badge: "border-orange-500/30 bg-orange-500/10 text-orange-300",
      dot: "bg-orange-400",
    };
  }

  return {
    label: value,
    badge: "border-slate-500/30 bg-slate-500/10 text-slate-300",
    dot: "bg-slate-400",
  };
};


const getDescriptionLabel = (entry) => {
  return String(entry?.category || "").toUpperCase() === "MISSING"
    ? "Physical Appearance"
    : "Crime / Case Description";
};


const getDescriptionText = (entry) => {
  const description = String(entry?.description || "").trim();

  if (description) {
    return description;
  }

  return String(entry?.category || "").toUpperCase() === "MISSING"
    ? "Physical appearance details not provided."
    : "Crime / case description not provided.";
};



/* ============================================================
 * SMALL UI COMPONENTS
 * ============================================================ */

const SectionLabel = ({ children }) => (
  <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
    {children}
  </div>
);


const StatusBadge = ({ status }) => {
  const meta = getStatusMeta(status);

  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-2.5 py-1 text-[11px] font-semibold tracking-wide ${meta.badge}`}
    >
      <span className={`h-1.5 w-1.5 rounded-full ${meta.dot}`} />
      {meta.label}
    </span>
  );
};


const CategoryBadge = ({ category }) => {
  const meta = getCategoryMeta(category);
  const Icon = meta.icon;

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold tracking-wide ${meta.badge}`}
    >
      <Icon size={12} />
      {meta.label}
    </span>
  );
};


const StatCard = ({
  title,
  value,
  icon: Icon,
  description,
  iconClassName,
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
          iconClassName || "text-blue-300"
        }`}
      >
        <Icon size={19} />
      </div>
    </div>

    {description && (
      <div className="mt-2 text-xs text-slate-500">
        {description}
      </div>
    )}
  </div>
);


/* ============================================================
 * COMPONENT
 * ============================================================ */

const PersonWatchlist = () => {
  const fileInputRef = useRef(null);

  const [entries, setEntries] = useState([]);
  const [loading, setLoading] = useState(true);

  const [search, setSearch] = useState("");
  const [categoryFilter, setCategoryFilter] = useState("ALL");
  const [statusFilter, setStatusFilter] = useState("ALL");

  const [formOpen, setFormOpen] = useState(false);
  const [formMode, setFormMode] = useState("create");
  const [editingEntry, setEditingEntry] = useState(null);
  const [saving, setSaving] = useState(false);

  const [form, setForm] = useState({ ...initialForm });
  const [formImagePreview, setFormImagePreview] = useState(null);

  const [selectedPreview, setSelectedPreview] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const [imageUrls, setImageUrls] = useState({});
  const [imageLoading, setImageLoading] = useState({});
  const [imageUploadingId, setImageUploadingId] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const [actionMenuId, setActionMenuId] = useState(null);


  /* ==========================================================
   * FORM HELPERS
   * ========================================================== */

  const updateField = (field, value) => {
    setForm((previous) => ({
      ...previous,
      [field]: value,
    }));
  };


  const resetForm = () => {
    setForm({ ...initialForm });
    setEditingEntry(null);

    if (formImagePreview) {
      revokeObjectUrl(formImagePreview);
    }

    setFormImagePreview(null);
  };


  const openCreate = () => {
    resetForm();
    setFormMode("create");
    setFormOpen(true);
  };


  const openEdit = (entry) => {
    if (!entry) {
      return;
    }

    if (formImagePreview) {
      revokeObjectUrl(formImagePreview);
    }

    setFormMode("edit");
    setEditingEntry(entry);
    setFormImagePreview(null);

    setForm({
      full_name: entry.full_name || "",
      category: entry.category || "OTHER",
      status: entry.status || "ACTIVE",
      description: entry.description || "",
      source: entry.source || "",
      external_reference: entry.external_reference || "",
      image: null,
    });

    setActionMenuId(null);
    setFormOpen(true);
  };


  const closeForm = () => {
    if (saving) {
      return;
    }

    setFormOpen(false);
    resetForm();
  };


  /* ==========================================================
   * IMAGE SELECTOR
   * ========================================================== */

  const handleImageChange = (event) => {
    const file = event.target.files?.[0] || null;

    if (event.target) {
      event.target.value = "";
    }

    if (!file) {
      return;
    }

    const validation = validateImage(file);

    if (validation) {
      toast.error(validation);
      return;
    }

    if (formImagePreview) {
      revokeObjectUrl(formImagePreview);
    }

    const nextPreview = URL.createObjectURL(file);

    setFormImagePreview(nextPreview);
    updateField("image", file);
  };


  /* ==========================================================
   * LOAD IMAGE CACHE
   * ========================================================== */

  const loadPersonImages = async (list) => {
    const nextUrls = {};
    const nextLoading = {};

    for (const entry of list) {
      if (!entry?.id || !entry.reference_image_available) {
        continue;
      }

      nextLoading[entry.id] = true;

      try {
        const blob = await getPersonWatchlistImage(entry.id);

        if (blob instanceof Blob && blob.size > 0) {
          nextUrls[entry.id] = URL.createObjectURL(blob);
        }
      } catch (error) {
        console.warn(
          `Unable to load person image ${entry.id}`,
          error
        );
      } finally {
        nextLoading[entry.id] = false;
      }
    }

    setImageUrls((previous) => {
      Object.values(previous).forEach(revokeObjectUrl);
      return nextUrls;
    });

    setImageLoading(nextLoading);
  };


  /* ==========================================================
   * LOAD LIST
   * ========================================================== */

  const load = async () => {
    setLoading(true);
    setActionMenuId(null);

    try {
      const response = await getPersonWatchlist();

      const list = Array.isArray(response?.entries)
        ? response.entries
        : [];

      const activeEntries = list.filter(
        (entry) =>
          String(entry?.status || "").toUpperCase() !== "DISABLED"
      );

      setEntries(activeEntries);

      await loadPersonImages(activeEntries);
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Failed to load person watchlist"
        )
      );
    } finally {
      setLoading(false);
    }
  };


  /* ==========================================================
   * INITIAL LOAD
   * ========================================================== */

  useEffect(() => {
    load();

    return () => {
      Object.values(imageUrls).forEach(revokeObjectUrl);
      revokeObjectUrl(previewUrl);
      revokeObjectUrl(formImagePreview);
    };

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);


  /* ==========================================================
   * FILTERED DATA
   * ========================================================== */

  const filteredEntries = useMemo(() => {
    const query = search.trim().toLowerCase();

    return entries.filter((entry) => {
      const category = String(entry?.category || "").toUpperCase();
      const status = String(entry?.status || "").toUpperCase();

      const matchesSearch =
        !query ||
        String(entry?.full_name || "").toLowerCase().includes(query) ||
        String(entry?.external_reference || "")
          .toLowerCase()
          .includes(query) ||
        String(entry?.source || "").toLowerCase().includes(query) ||
        String(entry?.description || "").toLowerCase().includes(query);

      const matchesCategory =
        categoryFilter === "ALL" ||
        category === categoryFilter;

      const matchesStatus =
        statusFilter === "ALL" ||
        status === statusFilter;

      return (
        matchesSearch &&
        matchesCategory &&
        matchesStatus
      );
    });
  }, [
    entries,
    search,
    categoryFilter,
    statusFilter,
  ]);


  /* ==========================================================
   * STATS
   * ========================================================== */

  const stats = useMemo(() => {
    const active = entries.filter(
      (entry) =>
        String(entry?.status || "").toUpperCase() === "ACTIVE"
    ).length;

    const missing = entries.filter(
      (entry) =>
        String(entry?.category || "").toUpperCase() === "MISSING"
    ).length;

    const wanted = entries.filter(
      (entry) =>
        String(entry?.category || "").toUpperCase() === "WANTED"
    ).length;

    const withImages = entries.filter(
      (entry) => Boolean(entry?.reference_image_available)
    ).length;

    return {
      active,
      missing,
      wanted,
      withImages,
    };
  }, [entries]);


  /* ==========================================================
   * SAVE
   * ========================================================== */

  const save = async (event) => {
    event.preventDefault();

    const fullName = form.full_name.trim();

    if (!fullName) {
      toast.error("Full name is required.");
      return;
    }

    if (formMode === "create" && !form.image) {
      toast.error("Reference face image is required.");
      return;
    }

    if (form.image) {
      const imageError = validateImage(form.image);

      if (imageError) {
        toast.error(imageError);
        return;
      }
    }

    setSaving(true);

    try {
      if (formMode === "create") {
        await enrollPersonWatchlist({
          ...form,
          full_name: fullName,
        });

        toast.success(
          "Person added to the watchlist."
        );
      } else {
        if (!editingEntry?.id) {
          throw new Error("Person ID is missing.");
        }

        await updatePersonWatchlist(
          editingEntry.id,
          {
            full_name: fullName,
            category: form.category,
            status: form.status,
            description: form.description,
            source: form.source,
            external_reference: form.external_reference,
          }
        );

        if (form.image) {
          await uploadPersonWatchlistImage(
            editingEntry.id,
            form.image
          );
        }

        toast.success(
          "Person details updated successfully."
        );
      }

      closeForm();
      await load();
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          formMode === "create"
            ? "Person enrollment failed"
            : "Person update failed"
        )
      );
    } finally {
      setSaving(false);
    }
  };


  /* ==========================================================
   * DISABLE
   * ========================================================== */

  const disablePerson = async (entry) => {
    if (!entry?.id) {
      return;
    }

    const confirmed = window.confirm(
      `Disable ${entry.full_name || "this person"} from the active watchlist?`
    );

    if (!confirmed) {
      return;
    }

    setDeletingId(entry.id);
    setActionMenuId(null);

    try {
      await deletePersonWatchlist(entry.id);

      setEntries((previous) =>
        previous.filter(
          (item) => item.id !== entry.id
        )
      );

      setImageUrls((previous) => {
        const next = { ...previous };

        if (next[entry.id]) {
          revokeObjectUrl(next[entry.id]);
          delete next[entry.id];
        }

        return next;
      });

      toast.success(
        "Person removed from the active watchlist."
      );
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Failed to disable person"
        )
      );
    } finally {
      setDeletingId(null);
    }
  };


  /* ==========================================================
   * REPLACE REFERENCE IMAGE
   * ========================================================== */

  const uploadReferenceImage = async (entry, event) => {
    const file = event.target.files?.[0] || null;

    if (event.target) {
      event.target.value = "";
    }

    if (!file) {
      return;
    }

    const validation = validateImage(file);

    if (validation) {
      toast.error(validation);
      return;
    }

    setImageUploadingId(entry.id);
    setActionMenuId(null);

    try {
      await uploadPersonWatchlistImage(
        entry.id,
        file
      );

      toast.success(
        "Reference photograph updated."
      );

      await load();
    } catch (error) {
      toast.error(
        getErrorMessage(
          error,
          "Failed to update reference photograph"
        )
      );
    } finally {
      setImageUploadingId(null);
    }
  };


  /* ==========================================================
   * PREVIEW
   * ========================================================== */

  const openPreview = async (entry) => {
    if (!entry) {
      return;
    }

    setSelectedPreview(entry);
    setPreviewLoading(true);
    setActionMenuId(null);

    if (previewUrl && !imageUrls[entry.id]) {
      revokeObjectUrl(previewUrl);
    }

    const cached = imageUrls[entry.id];

    if (cached) {
      setPreviewUrl(cached);
      setPreviewLoading(false);
      return;
    }

    try {
      const blob = await getPersonWatchlistImage(entry.id);

      if (!(blob instanceof Blob) || blob.size === 0) {
        throw new Error(
          "Reference photograph is not available."
        );
      }

      const url = URL.createObjectURL(blob);

      setPreviewUrl(url);
    } catch (error) {
      setPreviewUrl(null);

      toast.error(
        getErrorMessage(
          error,
          "Failed to load reference photograph"
        )
      );
    } finally {
      setPreviewLoading(false);
    }
  };


  const closePreview = () => {
    if (previewUrl && !selectedPreview?.id) {
      revokeObjectUrl(previewUrl);
    }

    setSelectedPreview(null);
    setPreviewUrl(null);
    setPreviewLoading(false);
  };


  /* ==========================================================
   * RENDER
   * ========================================================== */

  return (
    <div className="min-h-full space-y-6 text-white">

      {/* ======================================================
       * HERO / PAGE HEADER
       * ====================================================== */}

      <div className="relative overflow-hidden rounded-2xl border border-slate-800/90 bg-gradient-to-br from-[#091426] via-[#07101d] to-[#050b14] px-5 py-6 shadow-[0_20px_60px_rgba(0,0,0,0.25)] sm:px-7">

        <div className="pointer-events-none absolute -right-20 -top-20 h-64 w-64 rounded-full bg-blue-500/10 blur-3xl" />
        <div className="pointer-events-none absolute -bottom-28 left-1/3 h-56 w-56 rounded-full bg-cyan-500/5 blur-3xl" />

        <div className="relative flex flex-col gap-5 xl:flex-row xl:items-center xl:justify-between">

          <div>
            <div className="mb-3 flex items-center gap-2">
              <div className="rounded-lg border border-blue-400/20 bg-blue-400/10 p-2 text-blue-300">
                <Fingerprint size={19} />
              </div>

              <span className="text-[11px] font-semibold uppercase tracking-[0.2em] text-blue-300/80">
                Person Intelligence
              </span>
            </div>

            <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
              Person Watchlist
            </h1>

            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-400">
              Manage missing, wanted and watchlist persons,
              reference photographs, case information and
              encrypted face-recognition templates.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={load}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-3.5 py-2.5 text-sm font-medium text-slate-300 transition hover:border-slate-600 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RefreshCw
                size={16}
                className={loading ? "animate-spin" : ""}
              />
              Refresh
            </button>

            <button
              type="button"
              onClick={openCreate}
              className="inline-flex items-center gap-2 rounded-lg border border-blue-400/30 bg-blue-500 px-4 py-2.5 text-sm font-semibold text-white shadow-[0_8px_25px_rgba(59,130,246,0.18)] transition hover:bg-blue-400"
            >
              <Plus size={17} />
              Add Person
            </button>
          </div>
        </div>
      </div>


      {/* ======================================================
       * STATISTICS
       * ====================================================== */}

      <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
        <StatCard
          title="Active Records"
          value={stats.active}
          icon={CheckCircle2}
          description="Currently active"
          iconClassName="text-emerald-300"
        />

        <StatCard
          title="Missing"
          value={stats.missing}
          icon={UserX}
          description="Missing-person cases"
          iconClassName="text-amber-300"
        />

        <StatCard
          title="Wanted"
          value={stats.wanted}
          icon={ShieldAlert}
          description="Wanted / priority cases"
          iconClassName="text-red-300"
        />

        <StatCard
          title="Reference Photos"
          value={stats.withImages}
          icon={ImageIcon}
          description="Records with photograph"
          iconClassName="text-cyan-300"
        />
      </div>


      {/* ======================================================
       * SEARCH / FILTER BAR
       * ====================================================== */}

      <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3 shadow-[0_10px_35px_rgba(0,0,0,0.16)]">
        <div className="flex flex-col gap-3 xl:flex-row">

          <div className="relative flex-1">
            <Search
              size={16}
              className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
            />

            <input
              value={search}
              onChange={(event) =>
                setSearch(event.target.value)
              }
              placeholder="Search by name, case reference, source or description..."
              className="w-full rounded-lg border border-slate-800 bg-[#08111d] py-2.5 pl-9 pr-4 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/60"
            />
          </div>

          <div className="flex items-center gap-2">
            <div className="hidden items-center gap-2 text-xs text-slate-500 sm:flex">
              <Filter size={14} />
              Filters
            </div>

            <select
              value={categoryFilter}
              onChange={(event) =>
                setCategoryFilter(event.target.value)
              }
              className="rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/60"
            >
              <option value="ALL">All Categories</option>
              <option value="MISSING">Missing</option>
              <option value="WANTED">Wanted</option>
              <option value="OTHER">Other</option>
            </select>

            <select
              value={statusFilter}
              onChange={(event) =>
                setStatusFilter(event.target.value)
              }
              className="rounded-lg border border-slate-800 bg-[#08111d] px-3 py-2.5 text-sm text-slate-300 outline-none focus:border-blue-500/60"
            >
              <option value="ALL">All Status</option>
              <option value="ACTIVE">Active</option>
              <option value="CLEARED">Cleared</option>
              <option value="EXPIRED">Expired</option>
            </select>
          </div>
        </div>

        <div className="mt-3 flex items-center justify-between border-t border-slate-800/80 pt-3 text-xs text-slate-500">
          <span>
            Showing{" "}
            <span className="font-semibold text-slate-300">
              {filteredEntries.length}
            </span>{" "}
            of{" "}
            <span className="font-semibold text-slate-300">
              {entries.length}
            </span>{" "}
            active records
          </span>

          {(search ||
            categoryFilter !== "ALL" ||
            statusFilter !== "ALL") && (
            <button
              type="button"
              onClick={() => {
                setSearch("");
                setCategoryFilter("ALL");
                setStatusFilter("ALL");
              }}
              className="text-blue-300 hover:text-blue-200"
            >
              Clear filters
            </button>
          )}
        </div>
      </div>


      {/* ======================================================
       * MAIN TABLE
       * ====================================================== */}

      <div className="overflow-hidden rounded-2xl border border-slate-800 bg-[#07101b] shadow-[0_15px_50px_rgba(0,0,0,0.2)]">

        {loading ? (
          <div className="flex min-h-[360px] flex-col items-center justify-center text-center">
            <div className="mb-4 rounded-full border border-blue-500/20 bg-blue-500/10 p-4">
              <RefreshCw
                size={25}
                className="animate-spin text-blue-300"
              />
            </div>

            <div className="text-sm font-medium text-slate-300">
              Loading person intelligence...
            </div>

            <div className="mt-1 text-xs text-slate-600">
              Retrieving secure watchlist records
            </div>
          </div>
        ) : filteredEntries.length === 0 ? (
          <div className="flex min-h-[360px] flex-col items-center justify-center px-6 text-center">
            <div className="mb-4 rounded-full border border-slate-700 bg-slate-900 p-4">
              <UserRound
                size={28}
                className="text-slate-600"
              />
            </div>

            <div className="text-base font-semibold text-slate-300">
              No matching watchlist records
            </div>

            <div className="mt-1 max-w-md text-sm text-slate-500">
              Add a person or adjust your search and
              classification filters.
            </div>

            <button
              type="button"
              onClick={openCreate}
              className="mt-5 inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-500"
            >
              <Plus size={16} />
              Add Person
            </button>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-[1180px] w-full text-sm">

              <thead className="bg-slate-900/80">
                <tr className="border-b border-slate-800 text-left text-[11px] font-semibold uppercase tracking-[0.12em] text-slate-500">
                  <th className="px-5 py-3.5">
                    Person
                  </th>

                  <th className="px-4 py-3.5">
                    Classification
                  </th>

                  <th className="px-4 py-3.5">
                    Status
                  </th>

                  <th className="px-4 py-3.5">
                    Case / Reference
                  </th>

                  <th className="max-w-[380px] px-4 py-3.5">
                    Description
                  </th>

                  <th className="px-4 py-3.5">
                    Updated
                  </th>

                  <th className="px-5 py-3.5 text-right">
                    Actions
                  </th>
                </tr>
              </thead>

              <tbody className="divide-y divide-slate-800/80">
                {filteredEntries.map((entry) => {
                  const categoryMeta = getCategoryMeta(
                    entry.category
                  );

                  return (
                    <tr
                      key={entry.id}
                      className="group transition hover:bg-white/[0.02]"
                    >

                      {/* PERSON */}

                      <td className="px-5 py-4 align-top">
                        <div className="flex items-center gap-3">

                          <button
                            type="button"
                            onClick={() =>
                              openPreview(entry)
                            }
                            className="group/photo relative h-14 w-14 shrink-0 overflow-hidden rounded-xl border border-slate-700 bg-slate-900 shadow-inner transition hover:border-blue-400/60"
                            title="Open person profile"
                          >
                            {imageLoading[entry.id] ? (
                              <div className="flex h-full w-full items-center justify-center">
                                <RefreshCw
                                  size={16}
                                  className="animate-spin text-blue-300"
                                />
                              </div>
                            ) : imageUrls[entry.id] ? (
                              <>
                                <img
                                  src={imageUrls[entry.id]}
                                  alt={
                                    entry.full_name ||
                                    "Person"
                                  }
                                  className="h-full w-full object-cover"
                                />

                                <div className="absolute inset-0 flex items-center justify-center bg-black/50 opacity-0 transition group-hover/photo:opacity-100">
                                  <Eye
                                    size={17}
                                    className="text-white"
                                  />
                                </div>
                              </>
                            ) : (
                              <div className="flex h-full w-full items-center justify-center">
                                <UserRound
                                  size={23}
                                  className="text-slate-600"
                                />
                              </div>
                            )}
                          </button>

                          <div className="min-w-0">
                            <button
                              type="button"
                              onClick={() =>
                                openPreview(entry)
                              }
                              className="block max-w-[230px] truncate text-left font-semibold text-slate-100 hover:text-blue-300"
                              title={
                                entry.full_name ||
                                "Unknown"
                              }
                            >
                              {entry.full_name ||
                                "Unknown"}
                            </button>

                            <div className="mt-1 flex items-center gap-2 text-xs text-slate-500">
                              <span>
                                ID #{entry.id}
                              </span>

                              {entry.source && (
                                <>
                                  <span className="text-slate-700">
                                    •
                                  </span>
                                  <span className="truncate">
                                    {entry.source}
                                  </span>
                                </>
                              )}
                            </div>
                          </div>
                        </div>
                      </td>


                      {/* CLASSIFICATION */}

                      <td className="px-4 py-4 align-top">
                        <CategoryBadge
                          category={entry.category}
                        />

                        <div className="mt-2 text-[11px] text-slate-600">
                          {String(
                            entry.category || "OTHER"
                          ).toUpperCase() === "MISSING"
                            ? "Missing person"
                            : String(
                                entry.category || ""
                              ).toUpperCase() ===
                              "WANTED"
                            ? "Priority watchlist"
                            : "Other watchlist"}
                        </div>
                      </td>


                      {/* STATUS */}

                      <td className="px-4 py-4 align-top">
                        <StatusBadge
                          status={entry.status}
                        />
                      </td>


                      {/* REFERENCE */}

                      <td className="px-4 py-4 align-top">
                        <div className="flex items-start gap-2">
                          <FileText
                            size={15}
                            className="mt-0.5 shrink-0 text-slate-600"
                          />

                          <div className="min-w-0">
                            <div className="max-w-[180px] truncate text-sm text-slate-300">
                              {entry.external_reference ||
                                "No reference ID"}
                            </div>

                            <div className="mt-1 text-[11px] text-slate-600">
                              Case / reference
                            </div>
                          </div>
                        </div>
                      </td>


                      {/* DESCRIPTION */}

                      <td className="px-4 py-4 align-top">
                        <div className="max-w-[360px]">
                          <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                            {getDescriptionLabel(
                              entry
                            )}
                          </div>

                          <div className="line-clamp-3 whitespace-pre-wrap text-sm leading-5 text-slate-400">
                            {getDescriptionText(
                              entry
                            )}
                          </div>
                        </div>
                      </td>


                      {/* UPDATED */}

                      <td className="px-4 py-4 align-top">
                        <div className="flex items-start gap-2">
                          <Clock3
                            size={14}
                            className="mt-0.5 shrink-0 text-slate-600"
                          />

                          <div className="text-xs text-slate-500">
                            {formatDate(
                              entry.updated_at
                            )}
                          </div>
                        </div>
                      </td>


                      {/* ACTIONS */}

                      <td className="px-5 py-4 align-top text-right">
                        <div className="relative inline-flex items-center gap-1">

                          <button
                            type="button"
                            onClick={() =>
                              openPreview(entry)
                            }
                            className="rounded-lg p-2 text-cyan-300 transition hover:bg-cyan-500/10 hover:text-cyan-200"
                            title="Preview"
                          >
                            <Eye size={16} />
                          </button>

                          <button
                            type="button"
                            onClick={() =>
                              openEdit(entry)
                            }
                            className="rounded-lg p-2 text-amber-300 transition hover:bg-amber-500/10 hover:text-amber-200"
                            title="Edit"
                          >
                            <Pencil size={16} />
                          </button>

                          <label
                            className={`rounded-lg p-2 text-blue-300 transition hover:bg-blue-500/10 hover:text-blue-200 ${
                              imageUploadingId === entry.id
                                ? "pointer-events-none opacity-50"
                                : "cursor-pointer"
                            }`}
                            title="Replace reference photograph"
                          >
                            {imageUploadingId ===
                            entry.id ? (
                              <RefreshCw
                                size={16}
                                className="animate-spin"
                              />
                            ) : (
                              <Upload size={16} />
                            )}

                            <input
                              type="file"
                              accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                              className="hidden"
                              disabled={
                                imageUploadingId ===
                                entry.id
                              }
                              onChange={(event) =>
                                uploadReferenceImage(
                                  entry,
                                  event
                                )
                              }
                            />
                          </label>

                          <button
                            type="button"
                            onClick={() => {
                              setActionMenuId(
                                actionMenuId === entry.id
                                  ? null
                                  : entry.id
                              );
                            }}
                            className="rounded-lg p-2 text-slate-400 transition hover:bg-slate-800 hover:text-white"
                            title="More actions"
                          >
                            <MoreVertical size={16} />
                          </button>

                          {actionMenuId === entry.id && (
                            <div className="absolute right-0 top-10 z-30 w-44 overflow-hidden rounded-xl border border-slate-700 bg-[#08111d] p-1.5 text-left shadow-[0_20px_50px_rgba(0,0,0,0.45)]">

                              <button
                                type="button"
                                onClick={() =>
                                  openPreview(entry)
                                }
                                className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-xs text-slate-300 hover:bg-white/[0.04] hover:text-white"
                              >
                                <Eye size={14} />
                                Open profile
                              </button>

                              <button
                                type="button"
                                onClick={() =>
                                  openEdit(entry)
                                }
                                className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-xs text-slate-300 hover:bg-white/[0.04] hover:text-white"
                              >
                                <Pencil size={14} />
                                Edit details
                              </button>

                              <button
                                type="button"
                                onClick={() =>
                                  disablePerson(entry)
                                }
                                disabled={
                                  deletingId ===
                                  entry.id
                                }
                                className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-xs text-red-300 hover:bg-red-500/10 hover:text-red-200 disabled:opacity-50"
                              >
                                {deletingId ===
                                entry.id ? (
                                  <RefreshCw
                                    size={14}
                                    className="animate-spin"
                                  />
                                ) : (
                                  <Trash2 size={14} />
                                )}
                                Disable record
                              </button>
                            </div>
                          )}

                          <button
                            type="button"
                            onClick={() =>
                              disablePerson(entry)
                            }
                            disabled={
                              deletingId === entry.id
                            }
                            className="rounded-lg p-2 text-red-300 transition hover:bg-red-500/10 hover:text-red-200 disabled:opacity-50"
                            title="Disable person"
                          >
                            {deletingId ===
                            entry.id ? (
                              <RefreshCw
                                size={16}
                                className="animate-spin"
                              />
                            ) : (
                              <Trash2 size={16} />
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


      {/* ======================================================
       * ADD / EDIT MODAL
       * ====================================================== */}

      {formOpen && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center bg-black/75 p-4 backdrop-blur-sm"
          onMouseDown={(event) => {
            if (
              event.target ===
                event.currentTarget &&
              !saving
            ) {
              closeForm();
            }
          }}
        >
          <form
            onSubmit={save}
            className="flex max-h-[94vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-slate-700 bg-[#07101b] shadow-[0_30px_100px_rgba(0,0,0,0.55)]"
          >

            {/* MODAL HEADER */}

            <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4 sm:px-6">
              <div>
                <div className="flex items-center gap-2">
                  <div className="rounded-lg bg-blue-500/10 p-2 text-blue-300">
                    {formMode === "create" ? (
                      <Plus size={17} />
                    ) : (
                      <Pencil size={17} />
                    )}
                  </div>

                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      {formMode === "create"
                        ? "Add Person to Watchlist"
                        : "Edit Watchlist Person"}
                    </h2>

                    <p className="mt-0.5 text-xs text-slate-500">
                      {formMode === "create"
                        ? "Create a secure person-intelligence record."
                        : "Update case details or replace the reference photograph."}
                    </p>
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={closeForm}
                disabled={saving}
                className="rounded-lg p-2 text-slate-500 hover:bg-slate-800 hover:text-white"
              >
                <X size={19} />
              </button>
            </div>


            {/* MODAL CONTENT */}

            <div className="grid min-h-0 flex-1 overflow-y-auto lg:grid-cols-[360px_minmax(0,1fr)]">

              {/* LEFT PHOTO PANEL */}

              <div className="border-b border-slate-800 bg-[#06101b] p-5 lg:border-b-0 lg:border-r">
                <SectionLabel>
                  Reference Photograph
                </SectionLabel>

                <label className="group relative flex min-h-[310px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed border-slate-700 bg-slate-950/80 transition hover:border-blue-400/50 hover:bg-blue-500/[0.03]">

                  {formImagePreview ? (
                    <>
                      <img
                        src={formImagePreview}
                        alt="Selected reference"
                        className="h-full max-h-[310px] w-full object-contain p-4"
                      />

                      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent px-4 pb-4 pt-14">
                        <div className="truncate text-xs font-medium text-white">
                          {form.image?.name}
                        </div>
                        <div className="mt-1 text-[11px] text-slate-400">
                          Click to replace photograph
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <div className="mb-4 rounded-2xl border border-blue-500/20 bg-blue-500/10 p-4 text-blue-300">
                        <Upload size={28} />
                      </div>

                      <div className="text-center">
                        <div className="text-sm font-semibold text-slate-200">
                          Upload reference photograph
                        </div>

                        <div className="mx-auto mt-2 max-w-[240px] text-xs leading-5 text-slate-500">
                          Use a clear image containing one
                          identifiable face. JPG, PNG or WEBP.
                        </div>
                      </div>

                      <div className="mt-4 rounded-lg border border-slate-800 bg-slate-900/60 px-3 py-2 text-[11px] text-slate-600">
                        Maximum size: 10 MB
                      </div>
                    </>
                  )}

                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                    className="hidden"
                    disabled={saving}
                    onChange={handleImageChange}
                  />
                </label>

                <div className="mt-4 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <div className="flex items-start gap-2">
                    <AlertTriangle
                      size={15}
                      className="mt-0.5 shrink-0 text-amber-300"
                    />

                    <div className="text-[11px] leading-5 text-slate-500">
                      Enrollment requires exactly one clear face.
                      The face template is encrypted before database
                      storage.
                    </div>
                  </div>
                </div>
              </div>


              {/* RIGHT FORM PANEL */}

              <div className="p-5 sm:p-6">

                <div className="grid gap-5">

                  {/* NAME */}

                  <div>
                    <SectionLabel>
                      Person Information
                    </SectionLabel>

                    <label className="mb-1.5 block text-xs font-medium text-slate-400">
                      Full Name <span className="text-red-400">*</span>
                    </label>

                    <input
                      required
                      maxLength={150}
                      value={form.full_name}
                      onChange={(event) =>
                        updateField(
                          "full_name",
                          event.target.value
                        )
                      }
                      placeholder="Enter full legal name"
                      disabled={saving}
                      className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-2.75 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/60"
                    />
                  </div>


                  {/* CLASSIFICATION */}

                  <div className="grid gap-4 md:grid-cols-2">

                    <div>
                      <label className="mb-1.5 block text-xs font-medium text-slate-400">
                        Classification
                      </label>

                      <select
                        value={form.category}
                        onChange={(event) =>
                          updateField(
                            "category",
                            event.target.value
                          )
                        }
                        disabled={saving}
                        className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-2.75 text-sm text-slate-200 outline-none focus:border-blue-500/60"
                      >
                        <option value="MISSING">
                          MISSING
                        </option>
                        <option value="WANTED">
                          WANTED
                        </option>
                        <option value="OTHER">
                          OTHER
                        </option>
                      </select>
                    </div>


                    <div>
                      <label className="mb-1.5 block text-xs font-medium text-slate-400">
                        Operational Status
                      </label>

                      <select
                        value={form.status}
                        onChange={(event) =>
                          updateField(
                            "status",
                            event.target.value
                          )
                        }
                        disabled={saving}
                        className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-2.75 text-sm text-slate-200 outline-none focus:border-blue-500/60"
                      >
                        <option value="ACTIVE">
                          ACTIVE
                        </option>
                        <option value="CLEARED">
                          CLEARED
                        </option>
                        <option value="EXPIRED">
                          EXPIRED
                        </option>
                        <option value="DISABLED">
                          DISABLED
                        </option>
                      </select>
                    </div>
                  </div>


                  {/* CASE / SOURCE */}

                  <div className="grid gap-4 md:grid-cols-2">

                    <div>
                      <label className="mb-1.5 block text-xs font-medium text-slate-400">
                        Case / Reference ID
                      </label>

                      <input
                        maxLength={150}
                        value={form.external_reference}
                        onChange={(event) =>
                          updateField(
                            "external_reference",
                            event.target.value
                          )
                        }
                        placeholder="FIR / case / reference number"
                        disabled={saving}
                        className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-2.75 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/60"
                      />
                    </div>

                    <div>
                      <label className="mb-1.5 block text-xs font-medium text-slate-400">
                        Source / Department
                      </label>

                      <input
                        maxLength={150}
                        value={form.source}
                        onChange={(event) =>
                          updateField(
                            "source",
                            event.target.value
                          )
                        }
                        placeholder="Police station / department"
                        disabled={saving}
                        className="w-full rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-2.75 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/60"
                      />
                    </div>
                  </div>


                  {/* DESCRIPTION */}

                  <div>
                    <label className="mb-1.5 block text-xs font-medium text-slate-400">
                      {form.category === "MISSING"
                        ? "Physical Appearance"
                        : "Crime / Case Description"}
                    </label>

                    <textarea
                      maxLength={5000}
                      rows={7}
                      value={form.description}
                      onChange={(event) =>
                        updateField(
                          "description",
                          event.target.value
                        )
                      }
                      placeholder={
                        form.category === "MISSING"
                          ? "Example: Male, 25–30 years, medium build, black hair, brown eyes, approximately 5'8\", wearing a blue shirt..."
                          : "Describe the crime, incident, case details, identifiers or other relevant intelligence..."
                      }
                      disabled={saving}
                      className="w-full resize-none rounded-lg border border-slate-800 bg-[#08111d] px-3.5 py-3 text-sm leading-6 text-white outline-none placeholder:text-slate-600 focus:border-blue-500/60"
                    />

                    <div className="mt-1.5 text-right text-[10px] text-slate-600">
                      {form.description.length}/5000
                    </div>
                  </div>


                  {/* FOOTER */}

                  <div className="mt-1 flex flex-col-reverse gap-2 border-t border-slate-800 pt-5 sm:flex-row sm:justify-end">

                    <button
                      type="button"
                      onClick={closeForm}
                      disabled={saving}
                      className="rounded-lg border border-slate-700 bg-slate-900 px-4 py-2.5 text-sm font-medium text-slate-300 hover:bg-slate-800 disabled:opacity-50"
                    >
                      Cancel
                    </button>

                    <button
                      type="submit"
                      disabled={saving}
                      className="inline-flex items-center justify-center gap-2 rounded-lg border border-blue-400/20 bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white shadow-lg shadow-blue-950/30 hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                      {saving ? (
                        <RefreshCw
                          size={16}
                          className="animate-spin"
                        />
                      ) : formMode === "create" ? (
                        <UserRound size={16} />
                      ) : (
                        <Save size={16} />
                      )}

                      {saving
                        ? "Saving..."
                        : formMode === "create"
                        ? "Enroll Person"
                        : "Save Changes"}
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </form>
        </div>
      )}


      {/* ======================================================
       * PROFILE PREVIEW
       * ====================================================== */}

      <WatchlistActiveJourneys
        identityType="PERSON"
        entryIds={entries.map((entry) => entry.id)}
      />

      {selectedPreview && (
        <div
          className="fixed inset-0 z-[120] flex items-center justify-center bg-black/80 p-4 backdrop-blur-md"
          onMouseDown={(event) => {
            if (
              event.target ===
              event.currentTarget
            ) {
              closePreview();
            }
          }}
        >
          <div className="w-full max-w-5xl overflow-hidden rounded-2xl border border-slate-700 bg-[#07101b] shadow-[0_30px_120px_rgba(0,0,0,0.55)]">

            {/* HEADER */}

            <div className="flex items-center justify-between border-b border-slate-800 px-5 py-4">
              <div>
                <div className="flex items-center gap-2">
                  <div className="rounded-lg border border-cyan-500/20 bg-cyan-500/10 p-2 text-cyan-300">
                    <Fingerprint size={17} />
                  </div>

                  <div>
                    <h2 className="text-lg font-semibold text-white">
                      {selectedPreview.full_name ||
                        "Person Profile"}
                    </h2>

                    <p className="text-xs text-slate-500">
                      Person intelligence record
                    </p>
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={closePreview}
                className="rounded-lg p-2 text-slate-500 hover:bg-slate-800 hover:text-white"
              >
                <X size={19} />
              </button>
            </div>


            {/* BODY */}

            <div className="grid lg:grid-cols-[minmax(0,1.2fr)_380px]">

              {/* IMAGE */}

              <div className="min-h-[480px] bg-black p-6 lg:min-h-[580px]">
                <div className="flex h-full min-h-[440px] items-center justify-center rounded-xl border border-slate-800 bg-[#02060b]">

                  {previewLoading ? (
                    <div className="text-center">
                      <RefreshCw
                        size={28}
                        className="mx-auto animate-spin text-blue-300"
                      />
                      <div className="mt-3 text-sm text-slate-400">
                        Loading reference photograph...
                      </div>
                    </div>
                  ) : previewUrl ? (
                    <img
                      src={previewUrl}
                      alt={
                        selectedPreview.full_name ||
                        "Person reference"
                      }
                      className="max-h-[70vh] max-w-full rounded-lg object-contain"
                    />
                  ) : (
                    <div className="text-center text-slate-600">
                      <ImageIcon
                        size={46}
                        className="mx-auto mb-3"
                      />
                      <div className="text-sm">
                        Reference photograph not available
                      </div>
                    </div>
                  )}
                </div>
              </div>


              {/* DETAILS */}

              <div className="border-t border-slate-800 p-5 lg:border-l lg:border-t-0">

                <div className="space-y-5">

                  <div className="flex flex-wrap gap-2">
                    <CategoryBadge
                      category={
                        selectedPreview.category
                      }
                    />

                    <StatusBadge
                      status={
                        selectedPreview.status
                      }
                    />
                  </div>


                  <div>
                    <SectionLabel>
                      Case / Reference
                    </SectionLabel>

                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3.5 py-3 text-sm text-slate-300">
                      {selectedPreview.external_reference ||
                        "No reference number"}
                    </div>
                  </div>


                  <div>
                    <SectionLabel>
                      {getDescriptionLabel(
                        selectedPreview
                      )}
                    </SectionLabel>

                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 px-3.5 py-3 text-sm leading-6 text-slate-300">
                      {getDescriptionText(
                        selectedPreview
                      )}
                    </div>
                  </div>


                  <div className="grid grid-cols-2 gap-3">

                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                      <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                        Source
                      </div>
                      <div className="truncate text-sm text-slate-300">
                        {selectedPreview.source ||
                          "—"}
                      </div>
                    </div>

                    <div className="rounded-lg border border-slate-800 bg-slate-950/50 p-3">
                      <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-600">
                        Updated
                      </div>
                      <div className="text-xs text-slate-300">
                        {formatDate(
                          selectedPreview.updated_at
                        )}
                      </div>
                    </div>

                  </div>


                  <div className="rounded-lg border border-blue-500/10 bg-blue-500/[0.03] p-3">
                    <div className="flex items-start gap-2">
                      <Fingerprint
                        size={15}
                        className="mt-0.5 shrink-0 text-blue-300"
                      />

                      <div>
                        <div className="text-xs font-semibold text-slate-300">
                          Face Recognition
                        </div>

                        <div className="mt-1 text-[11px] leading-5 text-slate-500">
                          Secure recognition template is
                          stored encrypted on the backend.
                          Technical model details are retained
                          for system audit rather than shown
                          as the primary operational field.
                        </div>
                      </div>
                    </div>
                  </div>


                  {/* PROFILE ACTIONS */}

                  <div className="border-t border-slate-800 pt-4">

                    <div className="grid grid-cols-2 gap-2">

                      <button
                        type="button"
                        onClick={() => {
                          closePreview();
                          openEdit(
                            selectedPreview
                          );
                        }}
                        className="inline-flex items-center justify-center gap-2 rounded-lg border border-amber-500/20 bg-amber-500/10 px-3 py-2.5 text-sm font-medium text-amber-200 hover:bg-amber-500/15"
                      >
                        <Pencil size={15} />
                        Edit
                      </button>

                      <label className="inline-flex cursor-pointer items-center justify-center gap-2 rounded-lg border border-blue-500/20 bg-blue-500/10 px-3 py-2.5 text-sm font-medium text-blue-200 hover:bg-blue-500/15">
                        <Upload size={15} />
                        Replace Photo

                        <input
                          type="file"
                          accept=".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp"
                          className="hidden"
                          onChange={(event) =>
                            uploadReferenceImage(
                              selectedPreview,
                              event
                            )
                          }
                        />
                      </label>

                    </div>

                    <button
                      type="button"
                      onClick={() =>
                        disablePerson(
                          selectedPreview
                        )
                      }
                      className="mt-2 inline-flex w-full items-center justify-center gap-2 rounded-lg border border-red-500/20 bg-red-500/10 px-3 py-2.5 text-sm font-medium text-red-200 hover:bg-red-500/15"
                    >
                      <Trash2 size={15} />
                      Disable Watchlist Record
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


export default PersonWatchlist;
