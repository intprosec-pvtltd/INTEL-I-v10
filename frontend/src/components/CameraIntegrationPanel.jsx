import { useCallback, useEffect, useMemo, useState } from "react";
import { CloudDownload, Database, Loader2, Plus, RefreshCw, Trash2 } from "lucide-react";
import toast from "react-hot-toast";

import api from "../api/axios";

const EMPTY_FORM = {
  name: "Government Sentinel",
  provider_type: "sentinel",
  base_url: "",
  catalogue_path: "/api/ingest",
  auth_type: "bearer",
  token: "",
  username: "",
  password: "",
  api_key: "",
  api_key_header: "X-API-Key",
  auto_sync: true,
  sync_interval_seconds: 300,
  verify_tls: true,
  mapping: "",
};

const readError = (error, fallback) => {
  const detail = error?.response?.data?.detail;
  return (typeof detail === "string" && detail) || error?.message || fallback;
};

const CameraIntegrationPanel = () => {
  const [integrations, setIntegrations] = useState([]);
  const [form, setForm] = useState(EMPTY_FORM);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [busyId, setBusyId] = useState(null);
  const [discovery, setDiscovery] = useState(null);

  const loadIntegrations = useCallback(async () => {
    try {
      const { data } = await api.get("/api/integrations");
      setIntegrations(Array.isArray(data?.integrations) ? data.integrations : []);
    } catch (error) {
      toast.error(readError(error, "Unable to load camera integrations"));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    // Initial data synchronization with the integration API.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    loadIntegrations();
  }, [loadIntegrations]);

  const isVms = form.provider_type !== "sentinel";
  const authNeedsSecret = useMemo(
    () => ["bearer", "api_key"].includes(form.auth_type),
    [form.auth_type],
  );

  const update = (name, value) => setForm((current) => ({ ...current, [name]: value }));

  const buildConfig = () => {
    const config = {
      base_url: form.base_url.trim(),
      auth_type: form.auth_type,
      verify_tls: Boolean(form.verify_tls),
    };
    config[isVms ? "catalogue_path" : "ingest_path"] = (form.catalogue_path || "/api/ingest").trim();
    if (form.auth_type === "bearer") config.token = form.token.trim();
    if (form.auth_type === "api_key") {
      config.api_key = form.api_key.trim();
      config.api_key_header = form.api_key_header.trim() || "X-API-Key";
    }
    if (form.auth_type === "basic") {
      config.username = form.username.trim();
      config.password = form.password;
    }
    if (isVms && form.mapping.trim()) {
      try {
        config.mapping = JSON.parse(form.mapping);
      } catch {
        throw new Error("VMS field mapping must be valid JSON");
      }
    }
    return config;
  };

  const createIntegration = async (event) => {
    event.preventDefault();
    setSaving(true);
    try {
      const payload = {
        name: form.name.trim(),
        provider_type: form.provider_type,
        config: buildConfig(),
        enabled: true,
        auto_sync: Boolean(form.auto_sync),
        sync_interval_seconds: Number(form.sync_interval_seconds),
      };
      const { data } = await api.post("/api/integrations", payload);
      setForm(EMPTY_FORM);
      setIntegrations((rows) => [data, ...rows]);
      toast.success("Secure camera integration saved");
    } catch (error) {
      toast.error(readError(error, "Unable to save camera integration"));
    } finally {
      setSaving(false);
    }
  };

  const discover = async (integration) => {
    setBusyId(`discover-${integration.id}`);
    setDiscovery(null);
    try {
      const { data } = await api.post(`/api/integrations/${integration.id}/discover`);
      setDiscovery({ name: integration.name, ...data });
      toast.success(`${data?.count || 0} catalogue cameras discovered`);
    } catch (error) {
      toast.error(readError(error, "Camera discovery failed"));
    } finally {
      setBusyId(null);
    }
  };

  const sync = async (integration) => {
    setBusyId(`sync-${integration.id}`);
    try {
      const { data } = await api.post(`/api/integrations/${integration.id}/sync`);
      toast.success(`Synchronized ${data?.discovered_count || 0} cameras`);
      await loadIntegrations();
    } catch (error) {
      toast.error(readError(error, "Camera synchronization failed"));
    } finally {
      setBusyId(null);
    }
  };

  const remove = async (integration) => {
    if (!window.confirm(`Remove integration “${integration.name}”? Synced cameras will be retained.`)) return;
    setBusyId(`delete-${integration.id}`);
    try {
      await api.delete(`/api/integrations/${integration.id}`);
      setIntegrations((rows) => rows.filter((row) => row.id !== integration.id));
      if (discovery?.name === integration.name) setDiscovery(null);
      toast.success("Integration removed; existing cameras retained");
    } catch (error) {
      toast.error(readError(error, "Unable to remove integration"));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-blue-500/25 bg-blue-500/5 p-4 text-sm text-slate-300">
        <div className="flex items-start gap-3">
          <CloudDownload className="mt-0.5 shrink-0 text-blue-400" size={20} />
          <p>
            Connect a Sentinel government catalogue or REST-based VMS/NVR. Credentials and stream URLs are
            encrypted by the backend and are never returned to this browser.
          </p>
        </div>
      </div>

      <form onSubmit={createIntegration} className="rounded-xl border border-slate-700 bg-[#071626] p-4 sm:p-5">
        <div className="mb-4 flex items-center gap-2">
          <Plus size={20} className="text-blue-400" />
          <h2 className="text-lg font-bold">Add catalogue integration</h2>
        </div>

        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <label className="space-y-1 text-sm text-slate-300">
            <span>Name</span>
            <input required maxLength={150} value={form.name} onChange={(e) => update("name", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white" />
          </label>
          <label className="space-y-1 text-sm text-slate-300">
            <span>Provider</span>
            <select value={form.provider_type} onChange={(e) => update("provider_type", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white">
              <option value="sentinel">Government Sentinel</option>
              <option value="generic_vms">Generic REST VMS</option>
              <option value="generic_nvr">Generic REST NVR</option>
            </select>
          </label>
          <label className="space-y-1 text-sm text-slate-300">
            <span>HTTPS base URL</span>
            <input required type="url" placeholder="https://sentinel.gov.example" value={form.base_url} onChange={(e) => update("base_url", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white" />
          </label>
          <label className="space-y-1 text-sm text-slate-300">
            <span>Catalogue path</span>
            <input required value={form.catalogue_path} onChange={(e) => update("catalogue_path", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white" />
          </label>
          <label className="space-y-1 text-sm text-slate-300">
            <span>Authentication</span>
            <select value={form.auth_type} onChange={(e) => update("auth_type", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white">
              <option value="bearer">Bearer token</option>
              <option value="api_key">API key</option>
              <option value="basic">Basic authentication</option>
              <option value="none">No authentication</option>
            </select>
          </label>
          {authNeedsSecret && (
            <label className="space-y-1 text-sm text-slate-300">
              <span>{form.auth_type === "bearer" ? "Bearer token" : "API key"}</span>
              <input required type="password" autoComplete="new-password" value={form.auth_type === "bearer" ? form.token : form.api_key} onChange={(e) => update(form.auth_type === "bearer" ? "token" : "api_key", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white" />
            </label>
          )}
          {form.auth_type === "api_key" && (
            <label className="space-y-1 text-sm text-slate-300">
              <span>API key header</span>
              <input value={form.api_key_header} onChange={(e) => update("api_key_header", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white" />
            </label>
          )}
          {form.auth_type === "basic" && (
            <>
              <label className="space-y-1 text-sm text-slate-300"><span>Username</span><input required autoComplete="username" value={form.username} onChange={(e) => update("username", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white" /></label>
              <label className="space-y-1 text-sm text-slate-300"><span>Password</span><input required type="password" autoComplete="new-password" value={form.password} onChange={(e) => update("password", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white" /></label>
            </>
          )}
          <label className="space-y-1 text-sm text-slate-300">
            <span>Sync interval (seconds)</span>
            <input required min="60" max="86400" type="number" value={form.sync_interval_seconds} onChange={(e) => update("sync_interval_seconds", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 text-white" />
          </label>
        </div>

        {isVms && (
          <label className="mt-4 block space-y-1 text-sm text-slate-300">
            <span>Vendor field mapping (optional JSON)</span>
            <textarea rows={4} placeholder={'{"items":"data.devices","id":"uuid","name":"label","stream_url":"play"}'} value={form.mapping} onChange={(e) => update("mapping", e.target.value)} className="w-full rounded-lg border border-slate-700 bg-[#020b14] px-3 py-2.5 font-mono text-xs text-white" />
          </label>
        )}

        <div className="mt-4 flex flex-wrap items-center gap-5">
          <label className="flex items-center gap-2 text-sm text-slate-300"><input type="checkbox" checked={form.auto_sync} onChange={(e) => update("auto_sync", e.target.checked)} /> Automatic synchronization</label>
          <label className="flex items-center gap-2 text-sm text-slate-300"><input type="checkbox" checked={form.verify_tls} onChange={(e) => update("verify_tls", e.target.checked)} /> Verify TLS certificates</label>
          <button disabled={saving} className="ml-auto flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 font-semibold hover:bg-blue-500 disabled:opacity-50">
            {saving ? <Loader2 className="animate-spin" size={18} /> : <Plus size={18} />} Save integration
          </button>
        </div>
      </form>

      <section className="space-y-3">
        <div className="flex items-center justify-between gap-3"><h2 className="text-lg font-bold">Configured integrations</h2><button onClick={loadIntegrations} className="rounded-lg border border-slate-700 p-2 text-slate-300 hover:text-white" aria-label="Refresh integrations"><RefreshCw size={18} /></button></div>
        {loading && <div className="flex items-center gap-2 text-slate-400"><Loader2 className="animate-spin" size={18} /> Loading integrations…</div>}
        {!loading && integrations.length === 0 && <p className="rounded-xl border border-dashed border-slate-700 p-7 text-center text-slate-400">No catalogue integrations configured.</p>}
        <div className="grid gap-3 lg:grid-cols-2">
          {integrations.map((integration) => (
            <article key={integration.id} className="rounded-xl border border-slate-700 bg-[#071626] p-4">
              <div className="flex items-start justify-between gap-3">
                <div><h3 className="font-bold text-white">{integration.name}</h3><p className="text-xs uppercase tracking-wide text-blue-300">{integration.provider_type.replaceAll("_", " ")} · {integration.endpoint_origin || "Configured endpoint"}</p></div>
                <span className={`rounded-full px-2.5 py-1 text-xs font-bold ${integration.enabled ? "bg-emerald-500/10 text-emerald-300" : "bg-slate-700 text-slate-300"}`}>{integration.enabled ? "Enabled" : "Disabled"}</span>
              </div>
              <div className="mt-4 grid grid-cols-2 gap-2 text-sm sm:grid-cols-4">
                {[["Cameras", integration.last_camera_count], ["Created", integration.last_created_count], ["Updated", integration.last_updated_count], ["Offline", integration.last_offline_count]].map(([label, value]) => <div key={label} className="rounded-lg bg-[#020b14] p-2"><p className="text-xs text-slate-500">{label}</p><p className="font-bold">{value || 0}</p></div>)}
              </div>
              <p className="mt-3 text-xs text-slate-400">Status: <span className="font-semibold text-slate-200">{integration.last_sync_status || "NOT_SYNCED"}</span> · Auto-sync every {integration.sync_interval_seconds}s</p>
              {integration.last_error_message && <p className="mt-2 rounded-lg bg-red-500/10 p-2 text-xs text-red-300">{integration.last_error_message}</p>}
              <div className="mt-4 flex flex-wrap gap-2">
                <button disabled={Boolean(busyId)} onClick={() => discover(integration)} className="flex items-center gap-2 rounded-lg border border-blue-500/40 px-3 py-2 text-sm font-semibold text-blue-300 hover:bg-blue-500/10 disabled:opacity-50">{busyId === `discover-${integration.id}` ? <Loader2 className="animate-spin" size={16} /> : <Database size={16} />} Discover</button>
                <button disabled={Boolean(busyId)} onClick={() => sync(integration)} className="flex items-center gap-2 rounded-lg bg-blue-600 px-3 py-2 text-sm font-semibold hover:bg-blue-500 disabled:opacity-50">{busyId === `sync-${integration.id}` ? <Loader2 className="animate-spin" size={16} /> : <RefreshCw size={16} />} Sync now</button>
                <button disabled={Boolean(busyId)} onClick={() => remove(integration)} className="ml-auto flex items-center gap-2 rounded-lg border border-red-500/30 px-3 py-2 text-sm font-semibold text-red-300 hover:bg-red-500/10 disabled:opacity-50"><Trash2 size={16} /> Remove</button>
              </div>
            </article>
          ))}
        </div>
      </section>

      {discovery && (
        <section className="rounded-xl border border-cyan-500/25 bg-cyan-500/5 p-4">
          <h2 className="font-bold text-cyan-200">Discovery preview · {discovery.name}</h2>
          <p className="mt-1 text-sm text-slate-400">{discovery.count || 0} cameras · revision {discovery.source_revision || "not supplied"}. Stream addresses remain hidden.</p>
          <div className="mt-3 max-h-72 overflow-auto rounded-lg border border-slate-700">
            <table className="w-full min-w-[680px] text-left text-xs">
              <thead className="sticky top-0 bg-[#091728] text-slate-300"><tr><th className="p-2">Camera ID</th><th className="p-2">Name</th><th className="p-2">Live</th><th className="p-2">Location</th><th className="p-2">Streams</th></tr></thead>
              <tbody>{(discovery.cameras || []).map((camera) => <tr key={camera.external_id} className="border-t border-slate-800"><td className="p-2 font-mono text-cyan-300">{camera.external_id}</td><td className="p-2">{camera.name}</td><td className="p-2">{camera.live_status}</td><td className="p-2">{camera.location_name || [camera.latitude, camera.longitude].filter((x) => x != null).join(", ") || "—"}</td><td className="p-2">{(camera.streams || []).map((stream) => `${stream.source_type} ${stream.codec || ""} ${stream.width || "?"}×${stream.height || "?"}`).join("; ") || "—"}</td></tr>)}</tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
};

export default CameraIntegrationPanel;
