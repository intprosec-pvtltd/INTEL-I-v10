import React, { useEffect, useMemo, useState } from "react";
import { Activity, Car, RefreshCw, UserRound } from "lucide-react";
import api from "../api/axios.js";
import WatchlistJourneyEvidence from "./WatchlistJourneyEvidence.jsx";

const WatchlistActiveJourneys = ({ identityType, entryIds = [] }) => {
  const [journeys, setJourneys] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [loading, setLoading] = useState(true);
  const normalizedType = String(identityType || "VEHICLE").toUpperCase();
  const entryIdSet = useMemo(() => new Set(entryIds.map(Number)), [entryIds]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await api.get("/api/watchlist/journeys", {
          params: { identity_type: normalizedType, active_only: true, limit: 100 },
        });
        if (cancelled) return;
        const all = Array.isArray(response.data?.journeys) ? response.data.journeys : [];
        const filtered = entryIdSet.size
          ? all.filter((item) => {
              const journey = item.journey || {};
              const entryId = normalizedType === "PERSON"
                ? journey.person_watchlist_entry_id
                : journey.watchlist_entry_id;
              return entryIdSet.has(Number(entryId));
            })
          : all;
        setJourneys(filtered);
        setSelectedId((current) =>
          filtered.some((item) => item.incident_id === current)
            ? current
            : filtered[0]?.incident_id || null
        );
      } catch (error) {
        console.error("Active watchlist journey load failed:", error);
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    const timer = window.setInterval(load, 3000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [normalizedType, entryIdSet]);

  if (!loading && journeys.length === 0) return null;
  const TypeIcon = normalizedType === "PERSON" ? UserRound : Car;

  return (
    <section className="mt-6 rounded-2xl border border-slate-800 bg-[#07101b] p-4 sm:p-5">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-2.5 text-emerald-300"><Activity size={20} /></div>
          <div><h2 className="font-bold text-white">Active {normalizedType.toLowerCase()} watchlist tracking</h2><p className="mt-1 text-xs text-slate-500">Evidence and route update when the identity reaches another camera</p></div>
        </div>
        {loading && <RefreshCw size={18} className="animate-spin text-cyan-300" />}
      </div>

      {journeys.length > 1 && (
        <div className="mb-4 flex gap-2 overflow-x-auto pb-2">
          {journeys.map((item) => (
            <button key={item.incident_id} type="button" onClick={() => setSelectedId(item.incident_id)} className={`min-w-[220px] rounded-lg border p-3 text-left ${selectedId === item.incident_id ? "border-cyan-500 bg-cyan-500/10" : "border-slate-800 bg-[#08111d]"}`}>
              <div className="flex items-center gap-2 text-xs font-bold text-white"><TypeIcon size={14} />{item.journey?.reference || item.journey?.identity_id}</div>
              <p className="mt-1 text-[11px] text-slate-500">{item.journey?.camera_count || 0} cameras · {item.journey?.tracking_status}</p>
            </button>
          ))}
        </div>
      )}

      {selectedId && <WatchlistJourneyEvidence incidentId={selectedId} />}
    </section>
  );
};

export default WatchlistActiveJourneys;
