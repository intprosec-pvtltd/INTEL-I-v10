import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, Car, MapPin, RefreshCw, Route } from "lucide-react";
import {
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import api from "../api/axios";
import { formatIndianDateTime as formatTime, formatIndianTime } from "../utils/dateTime";

const DEFAULT_CENTER = [23.0225, 72.5714];
const POLL_INTERVAL_MS = 3000;

const validCoordinate = (value, min, max) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= min && parsed <= max;
};

const FitMap = ({ points }) => {
  const map = useMap();

  useEffect(() => {
    if (!points.length) return;
    if (points.length === 1) {
      map.setView(points[0], 15, { animate: true });
      return;
    }
    map.fitBounds(points, { padding: [55, 55], maxZoom: 16 });
  }, [map, points]);

  return null;
};

const VehicleCorrelationTracker = () => {
  const [cameras, setCameras] = useState([]);
  const [paths, setPaths] = useState([]);
  const [selectedVehicleId, setSelectedVehicleId] = useState("");
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState("");
  const [lastUpdated, setLastUpdated] = useState(null);

  const loadCorrelation = useCallback(async (manual = false) => {
    if (manual) setRefreshing(true);
    try {
      const [cameraResponse, pathResponse] = await Promise.all([
        api.get("/api/gis/cameras"),
        api.get("/api/gis/vehicle-paths", { params: { limit: 100 } }),
      ]);

      const nextCameras = Array.isArray(cameraResponse.data?.cameras)
        ? cameraResponse.data.cameras
        : [];
      const nextPaths = Array.isArray(pathResponse.data?.paths)
        ? pathResponse.data.paths
        : [];

      setCameras(nextCameras);
      setPaths(nextPaths);
      setSelectedVehicleId((current) => {
        if (nextPaths.some((path) => path.global_vehicle_id === current)) return current;
        const correlated = nextPaths.find((path) => Number(path.transition_count) > 0);
        return correlated?.global_vehicle_id || nextPaths[0]?.global_vehicle_id || "";
      });
      setError("");
      setLastUpdated(new Date());
    } catch (requestError) {
      console.error("Vehicle correlation refresh failed:", requestError);
      setError(
        requestError?.response?.status === 401
          ? "Session expired. Sign in again to load correlation data."
          : requestError?.response?.data?.detail || "Unable to load vehicle correlation data."
      );
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, []);

  useEffect(() => {
    loadCorrelation();
    const timer = window.setInterval(() => loadCorrelation(), POLL_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [loadCorrelation]);

  const selectedPath = useMemo(
    () => paths.find((path) => path.global_vehicle_id === selectedVehicleId) || null,
    [paths, selectedVehicleId]
  );

  const route = useMemo(() => {
    const observations = Array.isArray(selectedPath?.route) ? selectedPath.route : [];
    return observations.filter(
      (point) =>
        validCoordinate(point.latitude, -90, 90) &&
        validCoordinate(point.longitude, -180, 180)
    );
  }, [selectedPath]);

  const routePositions = useMemo(
    () => route.map((point) => [Number(point.latitude), Number(point.longitude)]),
    [route]
  );

  const mappedCameras = useMemo(
    () =>
      cameras.filter(
        (camera) =>
          validCoordinate(camera.latitude, -90, 90) &&
          validCoordinate(camera.longitude, -180, 180)
      ),
    [cameras]
  );

  const mapPoints = routePositions.length
    ? routePositions
    : mappedCameras.map((camera) => [Number(camera.latitude), Number(camera.longitude)]);

  const correlatedCount = paths.filter((path) => Number(path.transition_count) > 0).length;

  return (
    <section className="overflow-hidden rounded-2xl border border-[#183b63] bg-[#07182b] text-white">
      <header className="flex flex-col gap-4 border-b border-[#183b63] p-5 xl:flex-row xl:items-center xl:justify-between">
        <div className="flex items-center gap-3">
          <div className="rounded-xl border border-cyan-700/70 bg-cyan-950/60 p-3 text-cyan-300">
            <Route size={25} />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-xl font-bold">Vehicle Correlation Tracker</h2>
              <span className="rounded-full border border-emerald-700 bg-emerald-950/60 px-2 py-1 text-[10px] font-bold text-emerald-300">
                LIVE
              </span>
            </div>
            <p className="mt-1 text-sm text-[#6ea8d8]">
              Cross-camera journeys from live streams and uploaded videos
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <select
            value={selectedVehicleId}
            onChange={(event) => setSelectedVehicleId(event.target.value)}
            className="min-w-[250px] rounded-lg border border-[#21456d] bg-[#091a2d] px-3 py-2 text-sm text-white outline-none focus:border-cyan-500"
          >
            {!paths.length && <option value="">No correlated vehicles</option>}
            {paths.map((path) => (
              <option key={path.global_vehicle_id} value={path.global_vehicle_id}>
                {path.reference ? `${path.reference} · ` : ""}{path.global_vehicle_id} · {path.transition_count || 0} transitions
              </option>
            ))}
          </select>

          <button
            type="button"
            onClick={() => loadCorrelation(true)}
            disabled={refreshing}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-bold hover:bg-blue-700 disabled:opacity-60"
          >
            <RefreshCw size={16} className={refreshing ? "animate-spin" : ""} />
            Refresh
          </button>
        </div>
      </header>

      <div className="grid grid-cols-2 gap-px bg-[#183b63] md:grid-cols-4">
        {[
          ["Mapped Cameras", mappedCameras.length, MapPin],
          ["Vehicle Identities", paths.length, Car],
          ["Correlated Journeys", correlatedCount, Route],
          ["Route Observations", route.length, Activity],
        ].map(([label, value, Icon]) => (
          <div key={label} className="bg-[#0b1d32] p-4">
            <div className="mb-2 flex items-center gap-2 text-[#6ea8d8]">
              <Icon size={15} />
              <span className="text-xs font-bold uppercase tracking-wider">{label}</span>
            </div>
            <span className="text-2xl font-bold text-blue-400">{value}</span>
          </div>
        ))}
      </div>

      {error && (
        <div className="m-4 rounded-lg border border-red-700 bg-red-950/50 px-4 py-3 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 p-4 2xl:grid-cols-[1fr_360px]">
        <div className="relative min-h-[620px] overflow-hidden rounded-xl border border-[#21456d] bg-[#091a2d]">
          <MapContainer
            center={mapPoints[0] || DEFAULT_CENTER}
            zoom={12}
            className="h-[620px] w-full"
          >
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <FitMap points={mapPoints} />

            {mappedCameras.map((camera) => (
              <CircleMarker
                key={camera.cam_id}
                center={[Number(camera.latitude), Number(camera.longitude)]}
                radius={9}
                pathOptions={{
                  color: "#ffffff",
                  weight: 2,
                  fillColor: camera.is_active ? "#22c55e" : "#64748b",
                  fillOpacity: 1,
                }}
              >
                <Popup>
                  <strong>{camera.camera_name || camera.cam_id}</strong><br />
                  {camera.cam_id}<br />
                  Source: {camera.source_type || "unknown"}<br />
                  Status: {camera.connection_state || (camera.is_active ? "ONLINE" : "OFFLINE")}
                </Popup>
              </CircleMarker>
            ))}

            {routePositions.length > 1 && (
              <Polyline positions={routePositions} pathOptions={{ color: "#06b6d4", weight: 5, opacity: 0.9 }} />
            )}

            {route.map((point, index) => (
              <CircleMarker
                key={`${point.camera_id}-${point.timestamp}-${index}`}
                center={[Number(point.latitude), Number(point.longitude)]}
                radius={index === route.length - 1 ? 12 : 8}
                pathOptions={{
                  color: "#ffffff",
                  weight: 3,
                  fillColor: index === route.length - 1 ? "#f59e0b" : "#2563eb",
                  fillOpacity: 1,
                }}
              >
                <Popup>
                  <strong>Observation {index + 1}</strong><br />
                  Camera: {point.camera_name || point.camera_id}<br />
                  Track: {point.track_id || "Unknown"}<br />
                  Time: {formatTime(point.timestamp)}<br />
                  Confidence: {Math.round(Number(point.correlation_score || 0) * 100)}%
                </Popup>
              </CircleMarker>
            ))}
          </MapContainer>

          {loading && (
            <div className="absolute inset-0 z-[1000] flex items-center justify-center bg-[#07182b]/80">
              <RefreshCw className="animate-spin text-cyan-300" size={35} />
            </div>
          )}
        </div>

        <aside className="space-y-4">
          <div className="rounded-xl border border-[#21456d] bg-[#091a2d] p-4">
            <p className="text-xs font-bold uppercase tracking-widest text-[#6ea8d8]">Selected identity</p>
            <p className="mt-2 break-all text-lg font-bold text-cyan-300">
              {selectedPath?.reference || selectedPath?.global_vehicle_id || "No correlation available"}
            </p>
            {selectedPath && (
              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                <div><span className="text-[#6ea8d8]">Type</span><p className="font-bold">{selectedPath.identity_source === "WATCHLIST" ? "Watchlist vehicle" : (selectedPath.vehicle_type || "vehicle")}</p></div>
                <div><span className="text-[#6ea8d8]">Strong match</span><p className={selectedPath.strong_match ? "font-bold text-emerald-400" : "font-bold text-amber-400"}>{selectedPath.strong_match ? "YES" : "PENDING"}</p></div>
                <div><span className="text-[#6ea8d8]">Transitions</span><p className="font-bold">{selectedPath.transition_count || 0}</p></div>
                <div><span className="text-[#6ea8d8]">Last seen</span><p className="font-bold">{formatTime(selectedPath.last_seen)}</p></div>
                {selectedPath.identity_source === "WATCHLIST" && (
                  <>
                    <div><span className="text-[#6ea8d8]">Category</span><p className="font-bold text-amber-300">{selectedPath.category || "WATCHLIST"}</p></div>
                    <div><span className="text-[#6ea8d8]">Tracking</span><p className="font-bold text-cyan-300">{selectedPath.tracking_status || "ACTIVE"}</p></div>
                  </>
                )}
              </div>
            )}
          </div>

          <div className="max-h-[440px] overflow-y-auto rounded-xl border border-[#21456d] bg-[#091a2d] p-4">
            <p className="mb-3 text-xs font-bold uppercase tracking-widest text-[#6ea8d8]">Journey timeline</p>
            {!route.length ? (
              <p className="text-sm text-[#6ea8d8]">
                A watchlist hit appears here from the first mapped camera. A second camera creates the A → B journey.
              </p>
            ) : (
              <div className="space-y-3">
                {route.map((point, index) => (
                  <div key={`${point.camera_id}-timeline-${index}`} className="relative rounded-lg border border-[#183b63] bg-[#071524] p-3 pl-11">
                    <span className="absolute left-3 top-3 flex h-6 w-6 items-center justify-center rounded-full bg-blue-600 text-xs font-bold">{index + 1}</span>
                    <p className="font-bold text-blue-300">{point.camera_name || point.camera_id}</p>
                    <p className="mt-1 break-all text-xs text-[#86bde8]">{point.camera_id}</p>
                    <p className="mt-1 text-xs text-[#6ea8d8]">{formatTime(point.timestamp)}</p>
                    <p className="mt-1 text-xs text-cyan-300">Confidence {Math.round(Number(point.correlation_score || 0) * 100)}%</p>
                  </div>
                ))}
              </div>
            )}
          </div>

          <p className="text-right text-xs text-[#567da4]">
            Auto-refresh: 3 seconds{lastUpdated ? ` · Updated ${formatIndianTime(lastUpdated)}` : ""}
          </p>
        </aside>
      </div>
    </section>
  );
};

export default VehicleCorrelationTracker;
