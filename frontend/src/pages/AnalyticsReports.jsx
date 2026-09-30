import { useCallback, useEffect, useState } from "react";
import { Download, FileBarChart, Loader2, RefreshCw, ShieldCheck } from "lucide-react";
import toast from "react-hot-toast";

import api from "../api/axios";
import { formatIndianDate } from "../utils/dateTime";

const isoDay = (date) => date.toISOString().slice(0, 10);
const readError = (error, fallback) => error?.response?.data?.detail || error?.message || fallback;
const REPORT_DEFAULT_END = isoDay(new Date());
const REPORT_DEFAULT_START = isoDay(new Date(new Date().getTime() - 7 * 86400000));

const AnalyticsReports = () => {
  const [filters, setFilters] = useState({ start: REPORT_DEFAULT_START, end: REPORT_DEFAULT_END, camera_id: "", watchlist_only: false });
  const [cameras, setCameras] = useState([]);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState("");

  const params = useCallback(() => ({
    start: `${filters.start}T00:00:00Z`,
    end: `${filters.end}T23:59:59Z`,
    camera_id: filters.camera_id || undefined,
    watchlist_only: filters.watchlist_only,
  }), [filters]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const [{ data: preview }, { data: inventory }] = await Promise.all([
        api.get("/api/reports/analytics/preview", { params: { ...params(), limit: 250 } }),
        api.get("/cameras"),
      ]);
      setReport(preview);
      setCameras(Array.isArray(inventory?.cameras) ? inventory.cameras : []);
    } catch (error) {
      toast.error(readError(error, "Unable to prepare analytics report"));
    } finally {
      setLoading(false);
    }
  }, [params]);

  useEffect(() => {
    // Initial data synchronization with the reports API.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  const download = async (format) => {
    setExporting(format);
    try {
      const response = await api.get("/api/reports/analytics/export", { params: { ...params(), format, limit: 10000 }, responseType: "blob", timeout: 120000 });
      const disposition = response.headers?.["content-disposition"] || "";
      const filename = disposition.match(/filename="?([^";]+)"?/i)?.[1] || `intel-i-analytics.${format}`;
      const url = URL.createObjectURL(response.data);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      toast.success(`${format.toUpperCase()} report downloaded`);
    } catch (error) {
      toast.error(readError(error, `Unable to export ${format.toUpperCase()}`));
    } finally {
      setExporting("");
    }
  };

  const summary = report?.summary || {};
  return (
    <div className="mx-auto w-full max-w-[1800px] space-y-5 text-white">
      <header className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[#1f4b7a] bg-[#0b1b2e] p-5">
        <div className="flex items-center gap-3"><div className="rounded-xl bg-blue-500/10 p-3 text-blue-300"><FileBarChart size={25} /></div><div><h1 className="text-2xl font-bold">Analytics Output Report</h1><p className="text-sm text-slate-400">Evaluator-ready vehicle, plate, watchlist, timing and evidence records.</p></div></div>
        <div className="flex gap-2"><button disabled={Boolean(exporting)} onClick={() => download("csv")} className="flex items-center gap-2 rounded-lg border border-blue-500/40 px-4 py-2.5 font-semibold text-blue-200 hover:bg-blue-500/10 disabled:opacity-50">{exporting === "csv" ? <Loader2 className="animate-spin" size={18} /> : <Download size={18} />} CSV</button><button disabled={Boolean(exporting)} onClick={() => download("pdf")} className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 font-semibold hover:bg-blue-500 disabled:opacity-50">{exporting === "pdf" ? <Loader2 className="animate-spin" size={18} /> : <Download size={18} />} PDF</button></div>
      </header>

      <section className="grid gap-3 rounded-xl border border-slate-700 bg-[#071626] p-4 md:grid-cols-5">
        <label className="space-y-1 text-sm text-slate-300"><span>From</span><input type="date" value={filters.start} onChange={(e) => setFilters((f) => ({ ...f, start: e.target.value }))} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2 text-white" /></label>
        <label className="space-y-1 text-sm text-slate-300"><span>To</span><input type="date" value={filters.end} onChange={(e) => setFilters((f) => ({ ...f, end: e.target.value }))} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2 text-white" /></label>
        <label className="space-y-1 text-sm text-slate-300 md:col-span-2"><span>Camera</span><select value={filters.camera_id} onChange={(e) => setFilters((f) => ({ ...f, camera_id: e.target.value }))} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2 text-white"><option value="">All cameras</option>{cameras.map((camera) => <option key={camera.cam_id} value={camera.cam_id}>{camera.camera_name || camera.name || camera.cam_id}</option>)}</select></label>
        <div className="flex items-end gap-3"><label className="flex min-h-10 flex-1 items-center gap-2 text-sm text-slate-300"><input type="checkbox" checked={filters.watchlist_only} onChange={(e) => setFilters((f) => ({ ...f, watchlist_only: e.target.checked }))} /> Watchlist only</label><button onClick={load} disabled={loading} className="rounded-lg bg-blue-600 p-2.5 hover:bg-blue-500 disabled:opacity-50" aria-label="Refresh report"><RefreshCw className={loading ? "animate-spin" : ""} size={18} /></button></div>
      </section>

      <section className="grid grid-cols-2 gap-3 lg:grid-cols-5">
        {[["Records", summary.record_count], ["Unique plates", summary.unique_plate_count], ["Watchlist hits", summary.watchlist_match_count], ["Cameras", summary.camera_count], ["Latest event", formatIndianDate(summary.last_timestamp)]].map(([label, value]) => <div key={label} className="rounded-xl border border-slate-700 bg-[#071626] p-4"><p className="text-xs uppercase tracking-wide text-slate-500">{label}</p><p className="mt-1 text-2xl font-bold text-blue-300">{value ?? 0}</p></div>)}
      </section>

      <section className="overflow-hidden rounded-xl border border-slate-700 bg-[#071626]">
        <div className="flex items-center gap-2 border-b border-slate-700 p-4 text-sm text-slate-300"><ShieldCheck size={17} className="text-emerald-400" /> Exports are permission-controlled, audit-logged and timestamp-provenanced.</div>
        <div className="overflow-auto">
          <table className="w-full min-w-[1100px] text-left text-xs">
            <thead className="bg-[#091728] text-slate-300"><tr>{["Timestamp", "Camera / location", "Type", "Vehicle / plate", "Confidence", "Watchlist", "Timing", "Evidence"].map((title) => <th key={title} className="p-3">{title}</th>)}</tr></thead>
            <tbody>
              {loading && <tr><td colSpan={8} className="p-8 text-center text-slate-400"><Loader2 className="mx-auto mb-2 animate-spin" /> Loading report…</td></tr>}
              {!loading && !(report?.rows || []).length && <tr><td colSpan={8} className="p-8 text-center text-slate-400">No records match these filters.</td></tr>}
              {!loading && (report?.rows || []).map((row, index) => <tr key={`${row.record_type}-${row.event_id}-${index}`} className="border-t border-slate-800"><td className="p-3">{row.timestamp || "—"}</td><td className="p-3"><p className="font-semibold text-white">{row.camera_name || row.camera_id || "—"}</p><p className="text-slate-500">{row.location || "—"}</p></td><td className="p-3 uppercase text-blue-300">{row.record_type}</td><td className="p-3"><p>{row.vehicle_id || "Vehicle"}</p><p className="font-mono font-bold text-cyan-300">{row.plate || "—"}</p></td><td className="p-3">{row.confidence == null ? "—" : `${(Number(row.confidence) * 100).toFixed(1)}%`}</td><td className="p-3">{row.watchlist_result || "NO_MATCH"}</td><td className="p-3"><p>{row.timestamp_source || "—"}</p><p className="text-slate-500">{row.timestamp_quality || "—"}{row.source_pts_seconds == null ? "" : ` · PTS ${Number(row.source_pts_seconds).toFixed(3)}`}</p></td><td className="max-w-64 truncate p-3 text-slate-400" title={row.evidence_path || ""}>{row.evidence_path || "—"}</td></tr>)}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
};

export default AnalyticsReports;
