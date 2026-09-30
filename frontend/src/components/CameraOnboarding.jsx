import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Camera, FileSpreadsheet, Link2, Network, Plus, RefreshCw, Search, Server, Upload } from "lucide-react";
import toast from "react-hot-toast";

import api, { WEBSOCKET_URL } from "../api/axios";
import CameraIntegrationPanel from "./CameraIntegrationPanel";
import LiveCam from "./LiveCam";
import { getStoredUser } from "../auth/rbac";

const METHODS = [
  { id: "discover", title: "Auto Discover", text: "Discover cameras from an authorized private network.", icon: Network },
  { id: "government", title: "Government / VMS", text: "Connect Sentinel, VMS, NVR, or a government catalogue.", icon: Server },
  { id: "inventory", title: "CSV / Excel", text: "Validate and import an existing camera inventory.", icon: FileSpreadsheet },
  { id: "rtsp", title: "RTSP URLs", text: "Paste multiple stream URLs for parallel validation.", icon: Link2 },
  { id: "single", title: "Single Camera", text: "Configure one camera manually using the existing form.", icon: Plus },
];

const errorText = (error, fallback) => {
  const detail = error?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (detail?.message) return detail.message;
  return fallback;
};

const StatusPill = ({ value }) => {
  const normalized = String(value || "unknown").toUpperCase();
  const good = normalized === "ONLINE" || normalized === "COMPLETED";
  const warn = normalized.includes("ATTENTION") || normalized === "DEGRADED" || normalized === "VALIDATING";
  return <span className={`rounded-full px-2 py-1 text-[11px] font-bold ${good ? "bg-emerald-500/15 text-emerald-300" : warn ? "bg-amber-500/15 text-amber-300" : "bg-red-500/15 text-red-300"}`}>{normalized.replaceAll("_", " ")}</span>;
};

const JobProgress = ({ job }) => {
  if (!job) return null;
  const percent = job.total ? Math.round((job.completed / job.total) * 100) : 0;
  return <div className="rounded-lg border border-blue-500/30 bg-blue-500/5 p-4">
    <div className="flex items-center justify-between"><div><p className="font-bold">Discovering and validating cameras</p><p className="text-xs text-slate-400">Job {job.job_id}</p></div><StatusPill value={job.status} /></div>
    <div className="mt-3 h-2 overflow-hidden rounded bg-slate-800"><div className="h-full bg-blue-500 transition-all" style={{ width: `${percent}%` }} /></div>
    <div className="mt-3 grid grid-cols-2 gap-2 text-sm sm:grid-cols-5"><span>{job.completed} / {job.total}</span><span className="text-emerald-300">Online: {job.successful || 0}</span><span className="text-red-300">Failed: {job.failed || 0}</span><span className="text-amber-300">Duplicate: {job.duplicate || 0}</span><span>Processing: {job.processing || 0}</span></div>
  </div>;
};

const CameraOnboarding = () => {
  const canManage = ["super_admin", "rto_admin", "crime_admin", "cyber_admin"].includes(getStoredUser()?.role);
  const [method, setMethod] = useState(null);
  const [network, setNetwork] = useState("192.168.1.0/24");
  const [rtspUrls, setRtspUrls] = useState("");
  const [file, setFile] = useState(null);
  const [busy, setBusy] = useState(false);
  const [currentJob, setCurrentJob] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [fleet, setFleet] = useState({ cameras: [], summary: {}, total: 0 });
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("");
  const [page, setPage] = useState(1);
  const socketRef = useRef(null);

  const loadJobs = useCallback(async () => {
    try { const { data } = await api.get("/api/camera-onboarding/jobs", { params: { limit: 10 } }); setJobs(data?.jobs || []); }
    catch { /* The page still supports manual setup if onboarding is disabled. */ }
  }, []);

  const loadFleet = useCallback(async () => {
    try {
      const { data } = await api.get("/api/camera-fleet", { params: { page, page_size: 25, search: search || undefined, status: statusFilter || undefined } });
      setFleet(data || { cameras: [], summary: {}, total: 0 });
    } catch (error) { toast.error(errorText(error, "Unable to load camera fleet")); }
  }, [page, search, statusFilter]);

  useEffect(() => {
    // Initial synchronization with the server-side fleet catalogue.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadJobs();
    loadFleet();
  }, [loadJobs, loadFleet]);
  useEffect(() => {
    const socket = new WebSocket(WEBSOCKET_URL);
    socketRef.current = socket;
    socket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === "camera_onboarding_progress") {
          setCurrentJob(message); loadJobs();
          if (["completed", "requires_attention"].includes(String(message.status).toLowerCase())) loadFleet();
        }
      } catch { /* Ignore non-JSON heartbeat messages. */ }
    };
    return () => { socket.close(); socketRef.current = null; };
  }, [loadFleet, loadJobs]);

  const launch = async (request) => {
    setBusy(true);
    try { const { data } = await request(); setCurrentJob(data); await loadJobs(); toast.success(`Onboarding job started for ${data.total} camera${data.total === 1 ? "" : "s"}`); }
    catch (error) { toast.error(errorText(error, "Unable to start camera onboarding")); }
    finally { setBusy(false); }
  };

  const submitDiscovery = () => launch(() => api.post("/api/camera-onboarding/discover", { network, ports: [554, 8554, 80, 8080] }));
  const submitRtsp = () => {
    const urls = rtspUrls.split(/\r?\n/).map((value) => value.trim()).filter(Boolean);
    if (!urls.length) return toast.error("Paste at least one RTSP URL");
    launch(() => api.post("/api/camera-onboarding/import/rtsp", { urls }));
  };
  const submitFile = () => {
    if (!file) return toast.error("Choose a CSV or XLSX file");
    const form = new FormData(); form.append("file", file);
    launch(() => api.post("/api/camera-onboarding/import/csv", form));
  };

  const pages = useMemo(() => Math.max(1, Math.ceil((fleet.total || 0) / 25)), [fleet.total]);
  const summary = fleet.summary || {};

  return <div className="space-y-6">
    <section>
      <div className="mb-4"><h1 className="text-2xl font-bold">CAMERA ONBOARDING</h1><p className="mt-1 text-sm text-slate-400">Choose how you want to add cameras. Registration does not automatically start AI processing.</p></div>
      {canManage ? <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-5">{METHODS.map(({ id, title, text, icon: Icon }) => <button key={id} type="button" onClick={() => setMethod(id)} className={`rounded-lg border p-4 text-left transition ${method === id ? "border-blue-400 bg-blue-500/15" : "border-[#1f4b7a] bg-[#0d2038] hover:border-blue-500"}`}><Icon className="mb-3 h-6 w-6 text-blue-400" /><div className="font-bold">{title}</div><div className="mt-1 text-xs leading-5 text-slate-400">{text}</div></button>)}</div> : <div className="rounded-lg border border-[#1f4b7a] bg-[#0d2038] p-4 text-sm text-slate-300">You can view camera fleet health. Camera onboarding and administrative actions require an administrator role.</div>}
    </section>

    {method === "discover" && <section className="rounded-lg border border-[#1f4b7a] bg-[#0d2038] p-4"><h2 className="font-bold">Authorized Network Discovery</h2><p className="my-2 text-xs text-amber-300">Only administrator-approved private/internal CIDR ranges are accepted.</p><div className="flex flex-col gap-3 sm:flex-row"><input value={network} onChange={(e) => setNetwork(e.target.value)} className="flex-1 rounded border border-[#21456d] bg-[#142b46] px-3 py-2" placeholder="192.168.10.0/24" /><button disabled={busy} onClick={submitDiscovery} className="rounded bg-blue-600 px-5 py-2 font-bold disabled:opacity-50"><Search className="mr-2 inline h-4 w-4" />Discover Cameras</button></div></section>}
    {method === "government" && <section className="rounded-lg border border-[#1f4b7a] bg-[#0d2038] p-4"><CameraIntegrationPanel /></section>}
    {method === "inventory" && <section className="rounded-lg border border-[#1f4b7a] bg-[#0d2038] p-4"><h2 className="font-bold">Import CSV / Excel Inventory</h2><p className="my-2 text-xs text-slate-400">Required: camera_id or camera_name, plus rtsp_url/source. Invalid rows are isolated from healthy rows.</p><div className="flex flex-col gap-3 sm:flex-row"><input type="file" accept=".csv,.xlsx" onChange={(e) => setFile(e.target.files?.[0] || null)} className="flex-1 rounded border border-[#21456d] bg-[#142b46] px-3 py-2" /><button disabled={busy} onClick={submitFile} className="rounded bg-blue-600 px-5 py-2 font-bold disabled:opacity-50"><Upload className="mr-2 inline h-4 w-4" />Validate & Import</button></div></section>}
    {method === "rtsp" && <section className="rounded-lg border border-[#1f4b7a] bg-[#0d2038] p-4"><h2 className="font-bold">Bulk RTSP Import</h2><p className="my-2 text-xs text-slate-400">One URL per line. Credentials are encrypted and are never returned to the browser.</p><textarea value={rtspUrls} onChange={(e) => setRtspUrls(e.target.value)} rows={7} className="w-full rounded border border-[#21456d] bg-[#142b46] p-3 font-mono text-sm" placeholder={"rtsp://camera-1:554/stream\nrtsp://camera-2:554/stream"} /><button disabled={busy} onClick={submitRtsp} className="mt-3 rounded bg-blue-600 px-5 py-2 font-bold disabled:opacity-50">Start Validation</button></section>}
    {method === "single" && <section className="rounded-lg border border-[#1f4b7a] bg-[#0d2038] p-4"><LiveCam /></section>}

    <JobProgress job={currentJob} />

    <section className="rounded-lg border border-[#1f4b7a] bg-[#0d2038] p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><h2 className="text-xl font-bold">CAMERA FLEET</h2><p className="text-xs text-slate-400">Server-side pagination keeps large government fleets responsive.</p></div><button onClick={() => { loadFleet(); loadJobs(); }} className="rounded border border-[#21456d] px-3 py-2"><RefreshCw className="h-4 w-4" /></button></div>
      <div className="mb-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-5">{[["Total", summary.total || 0], ["Online", summary.online || 0], ["Offline", summary.offline || 0], ["Degraded", summary.degraded || 0], ["Auth Error", summary.authentication_failed || 0]].map(([label, value]) => <div key={label} className="rounded border border-[#183b63] bg-[#102943] p-3"><div className="text-xs text-slate-400">{label}</div><div className="text-xl font-bold">{Number(value).toLocaleString()}</div></div>)}</div>
      <div className="mb-3 flex flex-col gap-2 sm:flex-row"><div className="relative flex-1"><Search className="absolute left-3 top-2.5 h-4 w-4 text-slate-500" /><input value={search} onChange={(e) => { setSearch(e.target.value); setPage(1); }} className="w-full rounded border border-[#21456d] bg-[#142b46] py-2 pl-9 pr-3" placeholder="Search camera ID, name, or location" /></div><select value={statusFilter} onChange={(e) => { setStatusFilter(e.target.value); setPage(1); }} className="rounded border border-[#21456d] bg-[#142b46] px-3 py-2"><option value="">All statuses</option><option>ONLINE</option><option>OFFLINE</option><option>DEGRADED</option><option>AUTHENTICATION_FAILED</option><option>DISABLED</option></select></div>
      <div className="overflow-x-auto"><table className="min-w-full text-left text-sm"><thead className="border-b border-[#21456d] text-xs uppercase text-slate-400"><tr>{["Camera ID", "Name", "Location", "District", "Source", "Resolution", "Codec", "AI", "Health", "Last Seen"].map((value) => <th key={value} className="px-3 py-3">{value}</th>)}</tr></thead><tbody>{fleet.cameras?.map((camera) => <tr key={camera.id} className="border-b border-[#183b63]"><td className="whitespace-nowrap px-3 py-3 font-medium">{camera.camera_id}</td><td className="px-3 py-3">{camera.camera_name}</td><td className="px-3 py-3">{camera.location || "—"}</td><td className="px-3 py-3">{camera.district || "—"}</td><td className="px-3 py-3">{camera.source}</td><td className="px-3 py-3">{camera.resolution || "—"}</td><td className="px-3 py-3">{camera.codec || "—"}</td><td className="px-3 py-3">{camera.processing_enabled ? "Active" : "Registered"}</td><td className="px-3 py-3"><StatusPill value={camera.health} /></td><td className="whitespace-nowrap px-3 py-3 text-xs">{camera.last_seen ? new Date(camera.last_seen).toLocaleString() : "—"}</td></tr>)}</tbody></table>{!fleet.cameras?.length && <div className="py-8 text-center text-sm text-slate-400"><Camera className="mx-auto mb-2 h-7 w-7" />No cameras match the current filters.</div>}</div>
      <div className="mt-4 flex items-center justify-between text-sm"><span>Page {page} of {pages} · {fleet.total || 0} cameras</span><div className="space-x-2"><button disabled={page <= 1} onClick={() => setPage((v) => v - 1)} className="rounded border border-[#21456d] px-3 py-1 disabled:opacity-40">Previous</button><button disabled={page >= pages} onClick={() => setPage((v) => v + 1)} className="rounded border border-[#21456d] px-3 py-1 disabled:opacity-40">Next</button></div></div>
    </section>

    <section className="rounded-lg border border-[#1f4b7a] bg-[#0d2038] p-4"><h2 className="mb-3 font-bold">Recent Onboarding Jobs</h2><div className="space-y-2">{jobs.map((job) => <button key={job.job_id} onClick={() => setCurrentJob(job)} className="flex w-full items-center justify-between rounded border border-[#183b63] p-3 text-left"><div><div className="font-medium">{job.total} Cameras · {job.successful} Success · {job.failed} Failed</div><div className="text-xs text-slate-400">{job.job_type?.replaceAll("_", " ")} · {job.created_at ? new Date(job.created_at).toLocaleString() : ""}</div></div><StatusPill value={job.status} /></button>)}</div></section>
  </div>;
};

export default CameraOnboarding;
