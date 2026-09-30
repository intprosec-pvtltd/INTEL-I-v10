import React, { useEffect, useMemo, useState } from "react";
import L from "leaflet";
import {
  Camera,
  Car,
  MapPin,
  Radio,
  RefreshCw,
  Route,
  Upload,
  UserRound,
} from "lucide-react";
import {
  CircleMarker,
  MapContainer,
  Marker,
  Polyline,
  Popup,
  TileLayer,
  useMap,
} from "react-leaflet";
import "leaflet/dist/leaflet.css";
import api from "../api/axios.js";
import { formatIndianDateTime as formatTime } from "../utils/dateTime";

const DEFAULT_CENTER = [23.0225, 72.5714];

const FitJourney = ({ points }) => {
  const map = useMap();

  useEffect(() => {
    const fit = () => {
      map.invalidateSize({ pan: false });

      if (points.length === 1) {
        map.setView(points[0], 15);
      } else if (points.length > 1) {
        map.fitBounds(points, {
          padding: [55, 55],
          maxZoom: 16,
        });
      }
    };

    const frame = window.requestAnimationFrame(fit);
    const timer = window.setTimeout(fit, 250);
    const container = map.getContainer();
    const observer =
      typeof ResizeObserver !== "undefined"
        ? new ResizeObserver(() => map.invalidateSize({ pan: false }))
        : null;

    observer?.observe(container);
    window.addEventListener("resize", fit);

    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(timer);
      window.removeEventListener("resize", fit);
      observer?.disconnect();
    };
  }, [map, points]);

  return null;
};

const firstText = (...values) => {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
};

const AuthenticatedEvidenceImage = ({ source, alt, emptyLabel }) => {
  const [imageUrl, setImageUrl] = useState("");
  const [loading, setLoading] = useState(Boolean(source));
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    let objectUrl = "";

    if (!source) {
      setImageUrl("");
      setError("");
      setLoading(false);
      return undefined;
    }

    setLoading(true);
    setError("");

    api
      .get(source, { responseType: "blob", timeout: 30000 })
      .then((response) => {
        const blob = response?.data;
        const contentType = String(
          response?.headers?.["content-type"] || blob?.type || "",
        ).toLowerCase();

        if (!(blob instanceof Blob) || blob.size <= 0) {
          throw new Error("Evidence image is empty");
        }
        if (contentType && !contentType.startsWith("image/")) {
          throw new Error("Evidence endpoint did not return an image");
        }

        objectUrl = URL.createObjectURL(blob);
        if (!cancelled) setImageUrl(objectUrl);
      })
      .catch((requestError) => {
        if (!cancelled) {
          setImageUrl("");
          setError(
            requestError?.response?.data?.detail ||
              requestError?.message ||
              "Unable to load evidence image",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [source]);

  if (loading) {
    return (
      <div className="flex h-56 items-center justify-center bg-slate-950/70 text-slate-500">
        <RefreshCw size={22} className="animate-spin" />
      </div>
    );
  }

  if (!imageUrl) {
    return (
      <div className="flex h-56 items-center justify-center bg-slate-950/70 px-6 text-center text-xs text-slate-500">
        {error || emptyLabel}
      </div>
    );
  }

  return (
    <img
      src={imageUrl}
      alt={alt}
      className="h-56 w-full bg-slate-950 object-contain"
    />
  );
};

const getObservationDisplay = (observation, index) => {
  const metadata =
    observation?.metadata &&
    typeof observation.metadata === "object" &&
    !Array.isArray(observation.metadata)
      ? observation.metadata
      : {};

  const camera =
    observation?.camera && typeof observation.camera === "object"
      ? observation.camera
      : {};

  const cameraId = firstText(
    observation?.camera_id,
    observation?.cam_id,
    metadata.camera_id,
    metadata.cam_id,
  );

  const cameraName = firstText(
    observation?.camera_name,
    observation?.name,
    camera.name,
    metadata.camera_name,
  );

  const sourceType = firstText(
    observation?.source_type,
    observation?.camera_type,
    camera.source_type,
    metadata.source_type,
    metadata.camera_type,
    metadata.source,
  ).toLowerCase();

  const location = firstText(
    observation?.location_name,
    observation?.camera_location,
    observation?.location,
    camera.location_name,
    camera.camera_location,
    camera.location,
    metadata.location_name,
    metadata.camera_location,
    metadata.location,
    metadata.address,
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
    const friendlyApiName = /^upload camera\s+\d+$/i.test(cameraName)
      ? cameraName.replace(/\s+/g, " ")
      : "";
    const numberFromId = cameraId.match(/^upload(?:ed)?[_\s-]*(\d+)/i)?.[1];

    return {
      label: friendlyApiName || `Upload Camera ${numberFromId || index + 1}`,
      helper: "Uploaded video source",
      icon: Upload,
      badgeClass: "border-violet-500/30 bg-violet-500/10 text-violet-200",
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
    badgeClass: "border-cyan-500/30 bg-cyan-500/10 text-cyan-200",
  };
};

const WatchlistJourneyEvidence = ({ incidentId, onOpenSnapshot }) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!incidentId) return undefined;
    let cancelled = false;
    const load = async () => {
      try {
        const response = await api.get(
          `/api/watchlist/incidents/${encodeURIComponent(incidentId)}/journey`,
        );
        if (!cancelled) {
          setData(response.data);
          setError("");
        }
      } catch (requestError) {
        if (!cancelled && requestError?.response?.status !== 404) {
          setError(
            requestError?.response?.data?.detail ||
              "Unable to load watchlist journey",
          );
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    load();
    const timer = window.setInterval(load, 3000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [incidentId]);

  const journey = data?.journey;
  const observations = useMemo(
    () => (Array.isArray(journey?.route) ? journey.route : []),
    [journey],
  );
  const latestObservation = observations[observations.length - 1] || null;
  const watchlistSubject = data?.watchlist_subject || {};
  const personName = firstText(
    watchlistSubject.person_name,
    journey?.watchlist_person_name,
    journey?.reference,
  );
  const referenceImageSource = firstText(
    watchlistSubject.reference_image_url,
    journey?.reference_snapshot_path,
    journey?.reference_image_url,
  );
  const detectedImageSource = latestObservation?.snapshot_id
    ? `/snapshot/${encodeURIComponent(latestObservation.snapshot_id)}`
    : "";
  const route = useMemo(
    () =>
      (Array.isArray(journey?.route) ? journey.route : []).filter((item) => {
        const latitude = Number(item.latitude);
        const longitude = Number(item.longitude);
        return (
          Number.isFinite(latitude) &&
          latitude >= -90 &&
          latitude <= 90 &&
          Number.isFinite(longitude) &&
          longitude >= -180 &&
          longitude <= 180
        );
      }),
    [journey],
  );
  const points = useMemo(
    () => route.map((item) => [Number(item.latitude), Number(item.longitude)]),
    [route],
  );
  const observationDisplays = useMemo(
    () =>
      route.map((observation, index) =>
        getObservationDisplay(observation, index),
      ),
    [route],
  );
  const isPerson = journey?.identity_type === "PERSON";
  const IdentityIcon = isPerson ? UserRound : Car;
  const latestIcon = useMemo(
    () =>
      L.divIcon({
        className: "watchlist-live-marker",
        html: `<div style="width:42px;height:42px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:${isPerson ? "#7c3aed" : "#ea580c"};border:3px solid white;color:white;font-size:19px;box-shadow:0 5px 18px rgba(0,0,0,.45)">${isPerson ? "●" : "◆"}</div>`,
        iconSize: [42, 42],
        iconAnchor: [21, 21],
      }),
    [isPerson],
  );

  if (!journey && !loading && !error) return null;

  return (
    <div className="overflow-hidden rounded-2xl border border-cyan-500/20 bg-gradient-to-br from-cyan-500/[0.055] via-[#07111d] to-[#050b14] p-4 shadow-[0_18px_55px_rgba(0,0,0,0.22)] sm:p-5">
      <div className="mb-5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2.5 text-cyan-300">
            <Route size={19} />
          </div>
          <div>
            <div className="text-xs font-bold uppercase tracking-[0.15em] text-cyan-300">
              Live Watchlist Journey
            </div>
            <div className="mt-1 text-xs text-slate-500">
              Confirmed camera observations · refreshes every 3 seconds
            </div>
          </div>
        </div>
        {journey && (
          <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-3 py-1 text-xs font-bold text-cyan-300">
            {String(
              journey.tracking_status || "WAITING_FOR_NEXT_CAMERA",
            ).replaceAll("_", " ")}
          </span>
        )}
      </div>

      {loading && !journey ? (
        <div className="flex h-48 items-center justify-center">
          <RefreshCw className="animate-spin text-cyan-300" />
        </div>
      ) : error ? (
        <div className="rounded-lg border border-red-500/20 bg-red-500/5 p-3 text-sm text-red-300">
          {error}
        </div>
      ) : journey ? (
        <>
          <div className="mb-5 grid grid-cols-2 gap-3 lg:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-[#08111d]/90 p-3.5">
              <span className="flex items-center gap-1.5 text-[10px] uppercase tracking-wide text-slate-600">
                <IdentityIcon size={12} />
                Identity
              </span>
              <p className="mt-1.5 break-all text-sm font-bold text-white">
                {journey.identity_id}
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-[#08111d]/90 p-3.5">
              <span className="text-[10px] uppercase tracking-wide text-slate-600">
                Reference
              </span>
              <p className="mt-1.5 text-sm font-bold text-white">
                {journey.reference || "—"}
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-[#08111d]/90 p-3.5">
              <span className="text-[10px] uppercase tracking-wide text-slate-600">
                Cameras
              </span>
              <p className="mt-1.5 text-sm font-bold text-blue-300">
                {journey.camera_count || 0}
              </p>
            </div>
            <div className="rounded-xl border border-slate-800 bg-[#08111d]/90 p-3.5">
              <span className="text-[10px] uppercase tracking-wide text-slate-600">
                Last detected
              </span>
              <p className="mt-1.5 text-xs font-bold text-white">
                {formatTime(journey.last_seen)}
              </p>
            </div>
          </div>

          {isPerson && (
            <div className="mb-5 overflow-hidden rounded-xl border border-purple-500/20 bg-purple-500/[0.035]">
              <div className="border-b border-purple-500/15 px-4 py-3">
                <p className="text-sm font-semibold text-purple-100">
                  Watchlist person {personName || "Unknown person"} was detected
                  in {getObservationDisplay(latestObservation || {}, 0).label}.
                </p>
                <p className="mt-1 text-xs text-slate-500">
                  Reference enrollment image compared with the latest confirmed
                  camera evidence.
                </p>
              </div>

              <div className="grid gap-px bg-slate-800 md:grid-cols-2">
                <div className="bg-[#07101b]">
                  <div className="flex items-center gap-2 px-4 py-3 text-xs font-semibold uppercase tracking-[0.12em] text-purple-300">
                    <UserRound size={14} />
                    Uploaded Watchlist Image
                  </div>
                  <AuthenticatedEvidenceImage
                    source={referenceImageSource}
                    alt={`Watchlist reference for ${personName || "person"}`}
                    emptyLabel="No uploaded watchlist image is available."
                  />
                  <div className="px-4 py-3">
                    <p className="text-sm font-bold text-white">
                      {personName || "Unknown person"}
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      Category: {watchlistSubject.category || journey.category || "—"}
                    </p>
                  </div>
                </div>

                <div className="bg-[#07101b]">
                  <div className="flex items-center gap-2 px-4 py-3 text-xs font-semibold uppercase tracking-[0.12em] text-cyan-300">
                    <Camera size={14} />
                    Camera Detection Image
                  </div>
                  <AuthenticatedEvidenceImage
                    source={detectedImageSource}
                    alt={`Camera detection of ${personName || "watchlist person"}`}
                    emptyLabel="No camera detection snapshot is available."
                  />
                  <div className="px-4 py-3">
                    <p className="text-sm font-bold text-white">
                      {getObservationDisplay(latestObservation || {}, 0).label}
                    </p>
                    <p className="mt-1 text-xs text-slate-500">
                      {latestObservation
                        ? `${formatTime(latestObservation.timestamp)} · ${(
                            Number(latestObservation.confidence || 0) * 100
                          ).toFixed(1)}% match`
                        : "Detection details unavailable"}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div className="grid min-w-0 gap-4 lg:grid-cols-[minmax(0,1fr)_340px]">
            <div className="relative z-0 h-[380px] min-w-0 overflow-hidden rounded-xl border border-slate-700/80 bg-[#08111d] shadow-inner sm:h-[460px] lg:h-[520px]">
              <MapContainer
                center={points[0] || DEFAULT_CENTER}
                zoom={12}
                className="h-full w-full"
              >
                <TileLayer
                  attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
                  url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                />
                <FitJourney points={points} />
                {points.length > 1 && (
                  <Polyline
                    positions={points}
                    pathOptions={{ color: "#06b6d4", weight: 5, opacity: 0.9 }}
                  />
                )}
                {route.map((observation, index) => (
                  <CircleMarker
                    key={`${observation.camera_id}-${index}`}
                    center={points[index]}
                    radius={8}
                    pathOptions={{
                      color: "white",
                      weight: 2,
                      fillColor: "#2563eb",
                      fillOpacity: 1,
                    }}
                  >
                    <Popup>
                      <strong>{observationDisplays[index]?.label}</strong>
                      <br />
                      Detection {index + 1}
                      <br />
                      {formatTime(observation.timestamp)}
                      <br />
                      Confidence{" "}
                      {(Number(observation.confidence || 0) * 100).toFixed(1)}%
                    </Popup>
                  </CircleMarker>
                ))}
                {points.length > 0 && (
                  <Marker
                    position={points[points.length - 1]}
                    icon={latestIcon}
                    zIndexOffset={1000}
                  >
                    <Popup>
                      <strong>Latest confirmed position</strong>
                      <br />
                      {
                        observationDisplays[observationDisplays.length - 1]
                          ?.label
                      }
                      <br />
                      Waiting for next camera detection
                    </Popup>
                  </Marker>
                )}
              </MapContainer>
            </div>

            <div className="min-w-0 rounded-xl border border-slate-800 bg-[#07101b]/80 p-3">
              <div className="mb-3 flex items-center justify-between gap-3 border-b border-slate-800 pb-3">
                <div>
                  <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-slate-500">
                    Camera Observations
                  </p>
                  <p className="mt-1 text-[11px] text-slate-600">
                    Ordered confirmed detections
                  </p>
                </div>
                <span className="rounded-full border border-blue-500/20 bg-blue-500/10 px-2.5 py-1 text-[10px] font-bold text-blue-300">
                  {route.length}
                </span>
              </div>

              <div className="max-h-[320px] space-y-3 overflow-y-auto pr-1 sm:max-h-[400px] lg:max-h-[442px]">
                {route.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-slate-800 p-6 text-center text-xs text-slate-500">
                    Camera coordinates are required to draw the route.
                  </div>
                ) : (
                  route.map((observation, index) => (
                    <div
                      key={`${observation.camera_id}-timeline-${index}`}
                      className="rounded-xl border border-slate-800 bg-[#08111d] p-3 transition hover:border-slate-700"
                    >
                      <div className="flex items-start gap-3">
                        <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-blue-600 text-xs font-bold text-white shadow-[0_0_18px_rgba(37,99,235,0.25)]">
                          {index + 1}
                        </span>
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <p className="min-w-0 truncate font-bold text-blue-300">
                              {observationDisplays[index]?.label}
                            </p>
                            <span
                              className={`inline-flex shrink-0 items-center gap-1 rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase ${observationDisplays[index]?.badgeClass}`}
                            >
                              {React.createElement(
                                observationDisplays[index]?.icon || Camera,
                                { size: 10 },
                              )}
                              {observationDisplays[index]?.helper}
                            </span>
                          </div>
                          <p className="mt-1.5 text-xs text-slate-500">
                            {formatTime(observation.timestamp)}
                          </p>
                          {observation.snapshot_id && onOpenSnapshot && (
                            <button
                              type="button"
                              onClick={() =>
                                onOpenSnapshot(observation.snapshot_id)
                              }
                              className="mt-2 inline-flex items-center gap-1.5 rounded-lg border border-cyan-500/20 bg-cyan-500/[0.05] px-2.5 py-1.5 text-[10px] font-medium text-cyan-300 transition hover:bg-cyan-500/10"
                            >
                              <Camera size={11} />
                              Evidence image
                            </button>
                          )}
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>

          <div className="mt-4 flex items-start gap-2 rounded-lg border border-amber-500/10 bg-amber-500/[0.035] px-3 py-2.5 text-xs leading-5 text-amber-300/80">
            <MapPin size={13} className="mt-0.5 shrink-0" />
            <span>
              The icon marks the latest confirmed camera. Lines show observed
              camera-to-camera travel, not continuous GPS.
            </span>
          </div>
        </>
      ) : null}
    </div>
  );
};

export default WatchlistJourneyEvidence;