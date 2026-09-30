import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
  CircleMarker,
  Tooltip,
  useMap,
  useMapEvents,
} from "react-leaflet";

import {
  getGISCameras,
  getGISVehiclePaths,
  getVehicleJourney,
  WEBSOCKET_URL,
} from "../api/axios.js";

import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { formatIndianDateTime } from "../utils/dateTime";

/* ============================================================
   MAP CONFIGURATION
   ============================================================ */

const DEFAULT_CENTER = [23.0225, 72.5714];
const DEFAULT_ZOOM = 12;
const CAMERA_ZOOM = 16;
const DEFAULT_MAP_VIEW = "street";
const LOCATION_SEARCH_ZOOM = 16;
const LOCATION_SEARCH_ENDPOINT =
  "https://nominatim.openstreetmap.org/search";
const LOCATION_SEARCH_LIMIT = 5;

const MAP_VIEWS = Object.freeze({
  STREET: "street",
  SATELLITE: "satellite",
  DARK: "dark",
});

const TILE_CONFIG = Object.freeze({
  [MAP_VIEWS.STREET]: Object.freeze({
    name: "Street",
    icon: "🗺️",
    baseUrl: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    baseAttribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors',
    baseSubdomains: ["a", "b", "c"],
    maxZoom: 19,
    labelUrl: null,
    labelAttribution: "",
    labelSubdomains: [],
    transportationUrl: null,
  }),

  [MAP_VIEWS.SATELLITE]: Object.freeze({
    name: "Satellite",
    icon: "🛰️",
    baseUrl:
      "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    baseAttribution:
      "Tiles &copy; Esri",
    baseSubdomains: [""],
    maxZoom: 19,

    // Transparent reference labels above satellite imagery.
    labelUrl:
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    labelAttribution:
      "Labels &copy; Esri",
    labelSubdomains: [""],
    transportationUrl:
      "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Transportation/MapServer/tile/{z}/{y}/{x}",
  }),

  [MAP_VIEWS.DARK]: Object.freeze({
    name: "Dark",
    icon: "🌑",

    // Positron/dark tiles are more reliable when base and labels are separated.
    baseUrl:
      "https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png",
    baseAttribution:
      '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener noreferrer">CARTO</a>',
    baseSubdomains: ["a", "b", "c", "d"],
    maxZoom: 20,

    // Transparent labels so city / road / place names are always visible.
    labelUrl:
      "https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png",
    labelAttribution:
      '&copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener noreferrer">CARTO</a>',
    labelSubdomains: ["a", "b", "c", "d"],
    transportationUrl: null,
  }),
});

/* ============================================================
   CAMERA STATUS
   ============================================================ */

const STATUS_COLORS = Object.freeze({
  ONLINE: "#22c55e",
  ALERT: "#ef4444",
  DEGRADED: "#f59e0b",
  OFFLINE: "#64748b",
});

const getCameraStatus = (camera) => {
  if (camera?.is_active === true) {
    return "ONLINE";
  }

  const status = String(camera?.status || "")
    .trim()
    .toUpperCase();

  switch (status) {
    case "ALERT":
      return "ALERT";

    case "DEGRADED":
      return "DEGRADED";

    case "ONLINE":
      return "ONLINE";

    default:
      return "OFFLINE";
  }
};

const getStatusColor = (status) =>
  STATUS_COLORS[status] || STATUS_COLORS.OFFLINE;

/* ============================================================
   COORDINATE HELPERS
   ============================================================ */

const isValidLatitude = (value) => {
  const latitude = Number(value);

  return (
    Number.isFinite(latitude) &&
    latitude >= -90 &&
    latitude <= 90
  );
};

const isValidLongitude = (value) => {
  const longitude = Number(value);

  return (
    Number.isFinite(longitude) &&
    longitude >= -180 &&
    longitude <= 180
  );
};

const hasValidCoordinates = (camera) =>
  isValidLatitude(camera?.latitude) &&
  isValidLongitude(camera?.longitude);

const normalizeCoordinates = (latitude, longitude) => {
  const lat = Number(latitude);
  const lng = Number(longitude);

  if (
    !isValidLatitude(lat) ||
    !isValidLongitude(lng)
  ) {
    return null;
  }

  return {
    latitude: Number(lat.toFixed(6)),
    longitude: Number(lng.toFixed(6)),
  };
};

/* ============================================================
   DISPLAY HELPERS
   ============================================================ */

const safeDisplayValue = (
  value,
  fallback = "N/A"
) => {
  if (
    value === null ||
    value === undefined
  ) {
    return fallback;
  }

  if (typeof value === "string") {
    const trimmed = value.trim();

    return trimmed
      ? trimmed
      : fallback;
  }

  if (
    typeof value === "number" &&
    Number.isFinite(value)
  ) {
    return String(value);
  }

  return fallback;
};

const firstDefined = (...values) => {
  for (const value of values) {
    if (
      value !== null &&
      value !== undefined
    ) {
      return value;
    }
  }

  return null;
};

const numericValue = (...values) => {
  const value = firstDefined(...values);

  if (
    value === null ||
    value === undefined ||
    value === ""
  ) {
    return null;
  }

  const number = Number(value);

  return Number.isFinite(number)
    ? number
    : null;
};

const formatNumber = (
  value,
  decimals = 2,
  fallback = "N/A"
) => {
  const number = numericValue(value);

  if (number === null) {
    return fallback;
  }

  return number.toFixed(decimals);
};

const formatPercent = (
  value,
  decimals = 1
) => {
  const number = numericValue(value);

  if (number === null) {
    return "N/A";
  }

  const normalized =
    number <= 1
      ? number * 100
      : number;

  return `${normalized.toFixed(decimals)}%`;
};

const formatVehicleTimestamp = (
  timestamp
) => {
  return formatIndianDateTime(timestamp, "N/A");
};

const formatElapsedTime = (seconds) => {
  const value = numericValue(seconds);

  if (value === null || value < 0) {
    return "N/A";
  }

  if (value < 60) {
    return `${value.toFixed(1)} sec`;
  }

  const minutes = Math.floor(value / 60);
  const remainingSeconds = Math.round(
    value % 60
  );

  if (minutes < 60) {
    return `${minutes}m ${remainingSeconds}s`;
  }

  const hours = Math.floor(
    minutes / 60
  );

  const remainingMinutes =
    minutes % 60;

  return `${hours}h ${remainingMinutes}m`;
};

/* ============================================================
   CAMERA ICON
   ============================================================ */

const createCameraIcon = (
  status,
  selected
) => {
  const safeStatus =
    Object.prototype.hasOwnProperty.call(
      STATUS_COLORS,
      status
    )
      ? status
      : "OFFLINE";

  const color =
    getStatusColor(safeStatus);

  const border = selected
    ? "4px solid white"
    : "3px solid rgba(255,255,255,0.9)";

  const shadow = selected
    ? "0 0 18px"
    : "0 0 8px";

  return L.divIcon({
    className: "",

    html: `
      <div
        aria-hidden="true"
        style="
          width:38px;
          height:38px;
          border-radius:50%;
          background:${color};
          border:${border};
          box-shadow:${shadow} ${color};
          display:flex;
          align-items:center;
          justify-content:center;
          color:white;
          font-size:17px;
          font-weight:700;
          cursor:pointer;
          user-select:none;
        "
      >
        📹
      </div>
    `,

    iconSize: [38, 38],
    iconAnchor: [19, 19],
    popupAnchor: [0, -19],
  });
};

/* ============================================================
   SELECTED LOCATION ICON
   ============================================================ */

const createSelectedLocationIcon =
  () =>
    L.divIcon({
      className: "",

      html: `
        <div
          aria-hidden="true"
          style="
            width:30px;
            height:30px;
            border-radius:50%;
            background:#2563eb;
            border:4px solid white;
            box-shadow:0 0 14px #2563eb;
            display:flex;
            align-items:center;
            justify-content:center;
            color:white;
            font-size:18px;
            font-weight:700;
            cursor:pointer;
            user-select:none;
          "
        >
          +
        </div>
      `,

      iconSize: [30, 30],
      iconAnchor: [15, 15],
      popupAnchor: [0, -15],
    });

/* ============================================================
   VEHICLE CONFIGURATION
   ============================================================ */

const VEHICLE_PATH_COLORS =
  Object.freeze([
    "#00e5ff",
    "#a78bfa",
    "#f97316",
    "#22c55e",
    "#f43f5e",
    "#eab308",
    "#14b8a6",
    "#ec4899",
  ]);

const MAX_VEHICLE_PATHS = 100;
const MAX_ROUTE_POINTS = 1000;

const normalizeVehiclePathPoint =
  (point) => {
    if (
      !point ||
      typeof point !== "object"
    ) {
      return null;
    }

    const latitude =
      Number(point.latitude);

    const longitude =
      Number(point.longitude);

    if (
      !Number.isFinite(latitude) ||
      !Number.isFinite(longitude) ||
      latitude < -90 ||
      latitude > 90 ||
      longitude < -180 ||
      longitude > 180
    ) {
      return null;
    }

    const cameraId = String(
      point.camera_id ??
        point.cam_id ??
        ""
    ).trim();

    if (
      !cameraId ||
      cameraId.length > 128 ||
      !/^[A-Za-z0-9_.:-]+$/.test(
        cameraId
      )
    ) {
      return null;
    }

    return {
      ...point,

      camera_id: cameraId,

      latitude,

      longitude,

      timestamp:
        numericValue(
          point.timestamp
        ) ?? 0,

      similarity:
        Math.min(
          1,
          Math.max(
            0,
            numericValue(
              point.similarity
            ) ?? 0
          )
        ),

      correlation_score:
        Math.min(
          1,
          Math.max(
            0,
            numericValue(
              point.correlation_score
            ) ?? 0
          )
        ),

      strong_match:
        Boolean(
          point.strong_match
        ),
    };
  };

const normalizeVehiclePath =
  (path) => {
    if (
      !path ||
      typeof path !== "object"
    ) {
      return null;
    }

    const globalVehicleId =
      String(
        path.global_vehicle_id ??
          ""
      ).trim();

    if (
      !globalVehicleId ||
      globalVehicleId.length > 128 ||
      !/^[A-Za-z0-9_.:-]+$/.test(
        globalVehicleId
      )
    ) {
      return null;
    }

    const route =
      Array.isArray(path.route)
        ? path.route
            .slice(
              0,
              MAX_ROUTE_POINTS
            )
            .map(
              normalizeVehiclePathPoint
            )
            .filter(Boolean)
        : [];

    if (!route.length) {
      return null;
    }

    return {
      ...path,

      global_vehicle_id:
        globalVehicleId,

      vehicle_type:
        typeof path.vehicle_type ===
        "string"
          ? path.vehicle_type.slice(
              0,
              64
            )
          : "vehicle",

      route,

      segments:
        Array.isArray(
          path.segments
        )
          ? path.segments.slice(
              0,
              MAX_ROUTE_POINTS
            )
          : [],

      transitions:
        Array.isArray(
          path.transitions
        )
          ? path.transitions.slice(
              0,
              MAX_ROUTE_POINTS
            )
          : [],

      transition_count:
        Math.max(
          0,
          Number.isFinite(
            Number(
              path.transition_count
            )
          )
            ? Math.trunc(
                Number(
                  path.transition_count
                )
              )
            : 0
        ),

      strong_match:
        Boolean(
          path.strong_match
        ),

      last_seen:
        numericValue(
          path.last_seen
        ) ?? 0,
    };
  };

const getVehiclePathColor =
  (index) =>
    VEHICLE_PATH_COLORS[
      index %
        VEHICLE_PATH_COLORS.length
    ];

const createVehicleIcon = (
  color,
  selected
) => {
  const safeColor =
    typeof color === "string" &&
    /^#[0-9a-fA-F]{6}$/.test(
      color
    )
      ? color
      : "#00e5ff";

  return L.divIcon({
    className: "",

    html: `
      <div
        aria-hidden="true"
        style="
          width:34px;
          height:34px;
          border-radius:50%;
          background:${safeColor};
          border:${
            selected
              ? "4px solid white"
              : "3px solid rgba(255,255,255,.95)"
          };
          box-shadow:0 ${
            selected ? "0 18px" : "0 10px"
          } ${safeColor};
          display:flex;
          align-items:center;
          justify-content:center;
          color:#071526;
          font-size:17px;
          font-weight:800;
          cursor:pointer;
          user-select:none;
        "
      >
        🚗
      </div>
    `,

    iconSize: [34, 34],
    iconAnchor: [17, 17],
    popupAnchor: [0, -17],
  });
};

const routeCoordinates = (
  route
) =>
  route
    .map((point) => [
      Number(point.latitude),
      Number(point.longitude),
    ])
    .filter(
      ([latitude, longitude]) =>
        Number.isFinite(latitude) &&
        Number.isFinite(longitude) &&
        latitude >= -90 &&
        latitude <= 90 &&
        longitude >= -180 &&
        longitude <= 180
    );

/* ============================================================
   TRANSITION NORMALIZATION
   ============================================================ */

const normalizeTransition =
  (
    transition,
    fallbackSource,
    fallbackDestination
  ) => {
    if (
      !transition ||
      typeof transition !==
        "object"
    ) {
      return null;
    }

    const sourceCamera =
      firstDefined(
        transition.source_camera,
        transition.source_camera_id,
        transition.from_camera,
        transition.from_camera_id,
        transition.source,
        fallbackSource
      );

    const destinationCamera =
      firstDefined(
        transition.destination_camera,
        transition.destination_camera_id,
        transition.to_camera,
        transition.to_camera_id,
        transition.destination,
        fallbackDestination
      );

    return {
      ...transition,

      source_camera:
        sourceCamera,

      destination_camera:
        destinationCamera,

      distance_meters:
        numericValue(
          transition.distance_meters,
          transition.distance,
          transition.distance_m
        ),

      distance_km:
        numericValue(
          transition.distance_km
        ),

      elapsed_seconds:
        numericValue(
          transition.elapsed_seconds,
          transition.elapsed,
          transition.travel_time_seconds
        ),

      implied_speed_kmh:
        numericValue(
          transition.implied_speed_kmh,
          transition.speed_kmh,
          transition.implied_speed
        ),

      bearing_degrees:
        numericValue(
          transition.bearing_degrees,
          transition.bearing
        ),

      direction_score:
        numericValue(
          transition.direction_score
        ),

      travel_time_score:
        numericValue(
          transition.travel_time_score
        ),

      distance_score:
        numericValue(
          transition.distance_score
        ),

      gis_score:
        numericValue(
          transition.gis_score,
          transition.score
        ),

      feasible:
        typeof transition.feasible ===
        "boolean"
          ? transition.feasible
          : null,

      reason:
        firstDefined(
          transition.reason,
          transition.status_reason
        ),

      correlation:
        firstDefined(
          transition.correlation,
          transition.match,
          transition.decision
        ),
    };
  };

/* ============================================================
   BUILD TRANSITIONS FROM ROUTE
   ============================================================ */

const buildRouteTransitions =
  (route) => {
    if (
      !Array.isArray(route) ||
      route.length < 2
    ) {
      return [];
    }

    const transitions = [];

    for (
      let index = 1;
      index < route.length;
      index += 1
    ) {
      const previous =
        route[index - 1];

      const current =
        route[index];

      transitions.push(
        normalizeTransition(
          current?.transition ||
            current?.gis_transition ||
            current?.transition_data ||
            null,
          previous.camera_id,
          current.camera_id
        )
      );
    }

    return transitions.filter(
      Boolean
    );
  };

/* ============================================================
   MAP CONTROLLER
   ============================================================ */

const MapController = ({
  selectedCamera,
}) => {
  const map = useMap();

  useEffect(() => {
    if (
      !selectedCamera ||
      !hasValidCoordinates(
        selectedCamera
      )
    ) {
      return;
    }

    const latitude =
      Number(
        selectedCamera.latitude
      );

    const longitude =
      Number(
        selectedCamera.longitude
      );

    if (
      !isValidLatitude(latitude) ||
      !isValidLongitude(longitude)
    ) {
      return;
    }

    map.flyTo(
      [
        latitude,
        longitude,
      ],
      CAMERA_ZOOM,
      {
        duration: 1.2,
      }
    );
  }, [
    selectedCamera,
    map,
  ]);

  return null;
};

/* ============================================================
   MAP CLICK HANDLER
   ============================================================ */

const SearchLocationController = ({
  location,
}) => {
  const map = useMap();

  useEffect(() => {
    if (
      !location ||
      !isValidLatitude(location.latitude) ||
      !isValidLongitude(location.longitude)
    ) {
      return;
    }

    map.flyTo(
      [
        Number(location.latitude),
        Number(location.longitude),
      ],
      LOCATION_SEARCH_ZOOM,
      {
        duration: 1.25,
      }
    );
  }, [location, map]);

  return null;
};

/* ============================================================
   MAP CLICK HANDLER
   ============================================================ */

const MapClickHandler = ({
  onMapClick,
}) => {
  useMapEvents({
    click(event) {
      if (
        !event?.latlng ||
        typeof onMapClick !==
          "function"
      ) {
        return;
      }

      const coordinates =
        normalizeCoordinates(
          event.latlng.lat,
          event.latlng.lng
        );

      if (!coordinates) {
        return;
      }

      onMapClick(
        coordinates
      );
    },
  });

  return null;
};

/* ============================================================
   MAP TILES
   ============================================================ */

const MapTiles = ({
  selectedView,
}) => {
  const safeView =
    Object.prototype.hasOwnProperty.call(
      TILE_CONFIG,
      selectedView
    )
      ? selectedView
      : DEFAULT_MAP_VIEW;

  const config =
    TILE_CONFIG[
      safeView
    ];

  return (
    <>
      <TileLayer
        key={`${safeView}-base`}
        url={config.baseUrl}
        attribution={config.baseAttribution}
        subdomains={config.baseSubdomains}
        maxZoom={config.maxZoom}
        keepBuffer={2}
        updateWhenZooming
        updateWhenIdle
        zIndex={1}
      />

      {config.labelUrl && (
        <TileLayer
          key={`${safeView}-labels`}
          url={config.labelUrl}
          attribution={config.labelAttribution}
          subdomains={config.labelSubdomains}
          maxZoom={config.maxZoom}
          keepBuffer={2}
          updateWhenZooming
          updateWhenIdle
          zIndex={2}
          opacity={1}
        />
      )}

      {config.transportationUrl && (
        <TileLayer
          key={`${safeView}-transportation`}
          url={config.transportationUrl}
          attribution="Roads &copy; Esri"
          maxZoom={config.maxZoom}
          keepBuffer={2}
          updateWhenIdle
          zIndex={2}
          opacity={1}
        />
      )}
    </>
  );
};

/* ============================================================
   MAP VIEW SELECTOR
   ============================================================ */

const MapViewSelector = ({
  selectedView,
  onViewChange,
}) => {
  const [
    isOpen,
    setIsOpen,
  ] = useState(false);

  const safeSelectedView =
    Object.prototype.hasOwnProperty.call(
      TILE_CONFIG,
      selectedView
    )
      ? selectedView
      : DEFAULT_MAP_VIEW;

  const activeConfig =
    TILE_CONFIG[
      safeSelectedView
    ];

  const handleViewChange =
    useCallback(
      (view) => {
        if (
          !Object.prototype.hasOwnProperty.call(
            TILE_CONFIG,
            view
          )
        ) {
          return;
        }

        if (
          typeof onViewChange !==
          "function"
        ) {
          return;
        }

        onViewChange(view);
        setIsOpen(false);
      },
      [onViewChange]
    );

  return (
    <div
      className="absolute top-4 right-4 z-[60]"
    >
      <button
        type="button"
        onClick={() =>
          setIsOpen(
            (previous) =>
              !previous
          )
        }
        aria-label="Select map view"
        aria-haspopup="menu"
        aria-expanded={isOpen}
        className="
          flex items-center gap-2
          px-4 py-3 rounded-lg
          bg-[#071526]
          border border-blue-500
          text-white shadow-2xl
          hover:bg-[#102b49]
          transition
          focus:outline-none
          focus:ring-2
          focus:ring-blue-400
        "
      >
        <span
          aria-hidden="true"
          className="text-lg"
        >
          {activeConfig.icon}
        </span>

        <span className="text-sm font-bold">
          {activeConfig.name}
        </span>

        <span
          aria-hidden="true"
          className={
            isOpen
              ? "rotate-180 text-xs transition-transform"
              : "text-xs transition-transform"
          }
        >
          ▼
        </span>
      </button>

      {isOpen && (
        <div
          role="menu"
          aria-label="Map views"
          className="
            absolute right-0 mt-2
            w-52 overflow-hidden
            rounded-lg bg-[#071526]
            border border-blue-500
            shadow-2xl
          "
        >
          <div className="px-4 py-2 border-b border-[#1f4b7a]">
            <p className="text-[10px] uppercase tracking-wider text-gray-400">
              Map View
            </p>
          </div>

          {Object.entries(
            TILE_CONFIG
          ).map(
            ([
              viewKey,
              config,
            ]) => {
              const isActive =
                safeSelectedView ===
                viewKey;

              return (
                <button
                  key={viewKey}
                  type="button"
                  role="menuitem"
                  aria-current={
                    isActive
                      ? "true"
                      : undefined
                  }
                  onClick={() =>
                    handleViewChange(
                      viewKey
                    )
                  }
                  className={`
                    w-full flex items-center
                    gap-3 px-4 py-3
                    text-left text-sm
                    transition
                    focus:outline-none
                    focus:ring-2
                    focus:ring-inset
                    focus:ring-blue-400
                    ${
                      isActive
                        ? "bg-blue-600 text-white"
                        : "text-gray-200 hover:bg-[#102b49]"
                    }
                  `}
                >
                  <span
                    aria-hidden="true"
                    className="text-lg"
                  >
                    {config.icon}
                  </span>

                  <span className="font-semibold">
                    {config.name}
                  </span>

                  {isActive && (
                    <span className="ml-auto">
                      ✓
                    </span>
                  )}
                </button>
              );
            }
          )}
        </div>
      )}
    </div>
  );
};

/* ============================================================
   INTELLIGENCE VALUE ROW
   ============================================================ */

const IntelligenceRow = ({
  label,
  value,
  valueClassName = "",
}) => (
  <div className="flex items-center justify-between gap-3 py-1.5 border-b border-white/5 last:border-b-0">
    <span className="text-[10px] text-gray-400">
      {label}
    </span>

    <span
      className={`text-[11px] font-semibold text-right ${valueClassName}`}
    >
      {safeDisplayValue(value)}
    </span>
  </div>
);

/* ============================================================
   GIS DECISION BADGE
   ============================================================ */

const GISDecisionBadge = ({
  transition,
}) => {
  if (!transition) {
    return null;
  }

  const feasible =
    transition.feasible;

  const correlation =
    String(
      transition.correlation ||
        ""
    )
      .trim()
      .toUpperCase();

  let label = "GIS ANALYSIS";
  let className =
    "border-slate-500/60 bg-slate-900/60 text-slate-300";

  if (feasible === false) {
    label = "REJECTED";
    className =
      "border-red-500/60 bg-red-950/60 text-red-300";
  } else if (
    correlation.includes(
      "MATCH"
    )
  ) {
    label = "MATCH";
    className =
      "border-green-500/60 bg-green-950/60 text-green-300";
  } else if (
    feasible === true
  ) {
    label = "FEASIBLE";
    className =
      "border-cyan-500/60 bg-cyan-950/60 text-cyan-300";
  } else if (
    correlation.includes(
      "UNCERTAIN"
    )
  ) {
    label = "UNCERTAIN";
    className =
      "border-yellow-500/60 bg-yellow-950/60 text-yellow-300";
  }

  return (
    <span
      className={`
        inline-flex items-center
        px-2 py-1 rounded
        border text-[9px]
        font-bold tracking-wider
        ${className}
      `}
    >
      {label}
    </span>
  );
};

/* ============================================================
   GIS MAP
   ============================================================ */

const GISMap = ({
  cameras = [],
  selectedCamera = null,
  onCameraClick,
  onMapClick,
}) => {
  const [
    selectedMapView,
    setSelectedMapView,
  ] = useState(
    DEFAULT_MAP_VIEW
  );

  const [
    selectedLocation,
    setSelectedLocation,
  ] = useState(null);

  const [
    locationSearchQuery,
    setLocationSearchQuery,
  ] = useState("");

  const [
    locationSearchResults,
    setLocationSearchResults,
  ] = useState([]);

  const [
    locationSearchLoading,
    setLocationSearchLoading,
  ] = useState(false);

  const [
    locationSearchError,
    setLocationSearchError,
  ] = useState("");

  const [
    searchedLocation,
    setSearchedLocation,
  ] = useState(null);

  const [
    gisCameras,
    setGisCameras,
  ] = useState([]);

  const [
    vehiclePaths,
    setVehiclePaths,
  ] = useState([]);

  const [
    selectedVehicleId,
    setSelectedVehicleId,
  ] = useState(null);

  const [
    selectedVehicleJourney,
    setSelectedVehicleJourney,
  ] = useState(null);

  const [
    liveVehiclePositions,
    setLiveVehiclePositions,
  ] = useState({});

  const [livePersonPositions, setLivePersonPositions] = useState({});

  const [
    gisLoading,
    setGisLoading,
  ] = useState(false);

  const [
    gisError,
    setGisError,
  ] = useState("");

  /* ==========================================================
     REAL-TIME VEHICLE / PERSON POSITION STREAM
     ========================================================== */

  useEffect(() => {
    let socket = null;
    let reconnectTimer = null;
    let heartbeatTimer = null;
    let staleTimer = null;
    let initialConnectTimer = null;
    let cancelled = false;
    let reconnectAttempt = 0;
    let lastMessageAt = Date.now();

    const clearReconnectTimer = () => {
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const clearHeartbeatTimer = () => {
      if (heartbeatTimer !== null) {
        window.clearInterval(heartbeatTimer);
        heartbeatTimer = null;
      }
    };

    const clearInitialConnectTimer = () => {
      if (initialConnectTimer !== null) {
        window.clearTimeout(initialConnectTimer);
        initialConnectTimer = null;
      }
    };

    /*
     * Never call WebSocket.close() while the socket is CONNECTING.
     *
     * Chromium reports:
     * "WebSocket is closed before the connection is established"
     * when a CONNECTING socket is closed during React StrictMode cleanup.
     *
     * For CONNECTING sockets we detach the old handlers and install one
     * onopen cleanup handler. The connection is then closed only after
     * the handshake has completed.
     */
    const closeSocketSafely = (targetSocket) => {
      if (!targetSocket) {
        return;
      }

      if (socket === targetSocket) {
        socket = null;
      }

      if (targetSocket.readyState === WebSocket.CONNECTING) {
        targetSocket.onmessage = null;
        targetSocket.onerror = null;
        targetSocket.onclose = null;

        targetSocket.onopen = () => {
          targetSocket.onopen = null;

          try {
            if (targetSocket.readyState === WebSocket.OPEN) {
              targetSocket.close(1000, "client_cleanup");
            }
          } catch {
            // Best-effort cleanup only.
          }
        };

        return;
      }

      targetSocket.onopen = null;
      targetSocket.onmessage = null;
      targetSocket.onerror = null;
      targetSocket.onclose = null;

      if (targetSocket.readyState === WebSocket.OPEN) {
        try {
          targetSocket.close(1000, "client_cleanup");
        } catch {
          // Best-effort cleanup only.
        }
      }
    };

    const scheduleReconnect = () => {
      if (cancelled || reconnectTimer !== null) {
        return;
      }

      const existingState = socket?.readyState;

      if (
        existingState === WebSocket.OPEN ||
        existingState === WebSocket.CONNECTING
      ) {
        return;
      }

      const baseDelay = Math.min(
        15000,
        1000 * 2 ** Math.min(reconnectAttempt, 4)
      );

      /*
       * A small random jitter prevents Alert.jsx and GISMap.jsx from
       * reconnecting at exactly the same millisecond after an outage.
       */
      const delay = Math.round(
        baseDelay * (0.85 + Math.random() * 0.3)
      );

      reconnectAttempt += 1;

      reconnectTimer = window.setTimeout(() => {
        reconnectTimer = null;

        if (!cancelled) {
          connect();
        }
      }, delay);
    };

    const startHeartbeat = (connectedSocket) => {
      clearHeartbeatTimer();
      lastMessageAt = Date.now();

      heartbeatTimer = window.setInterval(() => {
        if (
          cancelled ||
          socket !== connectedSocket ||
          connectedSocket.readyState !== WebSocket.OPEN
        ) {
          return;
        }

        /*
         * If the server has been silent for too long, close an already-open
         * socket and let onclose schedule a clean reconnect.
         */
        if (Date.now() - lastMessageAt > 75000) {
          try {
            connectedSocket.close(4000, "heartbeat_timeout");
          } catch {
            // onclose/reconnect path remains authoritative.
          }

          return;
        }

        try {
          connectedSocket.send("ping");
        } catch {
          // Do not force-close from here; onclose will reconnect if needed.
        }
      }, 20000);
    };

    const connect = () => {
      if (cancelled) {
        return;
      }

      clearReconnectTimer();

      const existingState = socket?.readyState;

      if (
        existingState === WebSocket.OPEN ||
        existingState === WebSocket.CONNECTING
      ) {
        return;
      }

      const wsURL = String(WEBSOCKET_URL || "").trim();

      if (!wsURL) {
        console.warn(
          "[INTEL-I][GIS][WS] WebSocket URL is not configured"
        );
        return;
      }

      let parsedURL;

      try {
        parsedURL = new URL(
          wsURL,
          typeof window !== "undefined"
            ? window.location.origin
            : undefined
        );
      } catch (error) {
        console.error(
          "[INTEL-I][GIS][WS] Invalid WebSocket URL",
          error
        );
        scheduleReconnect();
        return;
      }

      if (
        parsedURL.protocol !== "ws:" &&
        parsedURL.protocol !== "wss:"
      ) {
        console.error(
          "[INTEL-I][GIS][WS] WebSocket URL must use ws:// or wss://"
        );
        return;
      }

      let newSocket;

      try {
        newSocket = new WebSocket(parsedURL.toString());
      } catch (error) {
        console.warn(
          "[INTEL-I][GIS][WS] Failed to create WebSocket",
          error
        );
        scheduleReconnect();
        return;
      }

      socket = newSocket;

      newSocket.onopen = () => {
        /*
         * A stale effect instance can finish connecting after React has
         * already cleaned it up. Close that obsolete socket safely.
         */
        if (cancelled || socket !== newSocket) {
          closeSocketSafely(newSocket);
          return;
        }

        reconnectAttempt = 0;
        lastMessageAt = Date.now();

        console.info(
          "[INTEL-I][GIS][WS] Connected"
        );

        startHeartbeat(newSocket);
      };

      newSocket.onmessage = (event) => {
        if (
          cancelled ||
          socket !== newSocket
        ) {
          return;
        }

        lastMessageAt = Date.now();

        if (typeof event.data !== "string") {
          return;
        }

        let payload;

        try {
          payload = JSON.parse(event.data || "{}");
        } catch {
          /*
           * Support a backend that responds with a plain-text ping/pong.
           */
          if (
            event.data === "ping" &&
            newSocket.readyState === WebSocket.OPEN
          ) {
            try {
              newSocket.send(
                JSON.stringify({
                  type: "pong",
                })
              );
            } catch {
              // Ignore control-message send failures.
            }
          }

          return;
        }

        /*
         * Connection/control messages are not GIS telemetry.
         */
        if (
          [
            "connected",
            "heartbeat",
            "ping",
            "pong",
          ].includes(payload?.type)
        ) {
          if (
            (
              payload?.type === "heartbeat" ||
              payload?.type === "ping"
            ) &&
            newSocket.readyState === WebSocket.OPEN
          ) {
            try {
              newSocket.send(
                JSON.stringify({
                  type: "pong",
                })
              );
            } catch {
              // Ignore control-message send failures.
            }
          }

          return;
        }

        const isVehicle =
          payload?.type === "vehicle_position" &&
          payload?.event === "VEHICLE_POSITION_UPDATED";

        const isPerson =
          payload?.type === "person_position" &&
          payload?.event === "PERSON_POSITION_UPDATED";

        if (!isVehicle && !isPerson) {
          return;
        }

        const id = String(
          payload.global_vehicle_id ||
            payload.global_person_id ||
            payload.vehicle_id ||
            payload.person_id ||
            ""
        ).trim();

        const latitude = Number(payload.latitude);
        const longitude = Number(payload.longitude);
        const timestamp = Number(payload.timestamp);

        if (
          !id ||
          !/^[A-Za-z0-9_.:-]{1,128}$/.test(id) ||
          !Number.isFinite(latitude) ||
          !Number.isFinite(longitude) ||
          latitude < -90 ||
          latitude > 90 ||
          longitude < -180 ||
          longitude > 180 ||
          !Number.isFinite(timestamp)
        ) {
          return;
        }

        const setter = isPerson
          ? setLivePersonPositions
          : setLiveVehiclePositions;

        setter((previous) => ({
          ...previous,

          [id]: {
            global_vehicle_id: isVehicle
              ? id
              : null,

            global_person_id: isPerson
              ? id
              : null,

            entity_type: isPerson
              ? "person"
              : "vehicle",

            camera_id: String(
              payload.camera_id || ""
            ),

            track_id: String(
              payload.track_id || ""
            ),

            latitude,

            longitude,

            position_type: String(
              payload.position_type || "UNKNOWN"
            ),

            vehicle_type: String(
              payload.vehicle_type || "vehicle"
            ).slice(0, 64),

            speed_kmh: Number.isFinite(
              Number(payload.speed_kmh)
            )
              ? Number(payload.speed_kmh)
              : null,

            heading: Number.isFinite(
              Number(payload.heading)
            )
              ? Number(payload.heading)
              : null,

            confidence: Math.max(
              0,
              Math.min(
                1,
                Number(payload.confidence) || 0
              )
            ),

            timestamp,

            received_at: Date.now() / 1000,
          },
        }));
      };

      newSocket.onerror = () => {
        /*
         * Do NOT call close() here.
         *
         * onerror can fire while readyState === CONNECTING. Closing from
         * this callback is one direct cause of Chromium's
         * "closed before the connection is established" warning.
         *
         * The browser will subsequently emit onclose and that is the only
         * place that schedules reconnection.
         */
      };

      newSocket.onclose = (event) => {
        if (socket === newSocket) {
          socket = null;
        }

        clearHeartbeatTimer();

        if (cancelled) {
          return;
        }

        console.debug(
          "[INTEL-I][GIS][WS] Closed",
          {
            code: event.code,
            reason: event.reason || "",
          }
        );

        /*
         * A 1008 close normally means authentication/policy rejection.
         * Reconnecting forever would only create noise and load.
         */
        if (event.code === 1008) {
          console.warn(
            "[INTEL-I][GIS][WS] Connection rejected by server policy/authentication"
          );
          return;
        }

        scheduleReconnect();
      };
    };

    /*
     * Remove only stale transient live overlays. Persisted GIS route/camera
     * data is loaded independently through REST below.
     */
    staleTimer = window.setInterval(() => {
      if (cancelled) {
        return;
      }

      const now = Date.now() / 1000;

      setLiveVehiclePositions((previous) => {
        let changed = false;
        const next = {
          ...previous,
        };

        for (
          const [id, position] of Object.entries(next)
        ) {
          if (
            now -
              Number(
                position?.received_at || 0
              ) >
            10
          ) {
            delete next[id];
            changed = true;
          }
        }

        return changed
          ? next
          : previous;
      });

      setLivePersonPositions((previous) => {
        let changed = false;
        const next = {
          ...previous,
        };

        for (
          const [id, position] of Object.entries(next)
        ) {
          if (
            now -
              Number(
                position?.received_at || 0
              ) >
            10
          ) {
            delete next[id];
            changed = true;
          }
        }

        return changed
          ? next
          : previous;
      });
    }, 2000);

    const ensureConnected = () => {
      if (
        cancelled ||
        document.hidden
      ) {
        return;
      }

      const state = socket?.readyState;

      if (
        state !== WebSocket.OPEN &&
        state !== WebSocket.CONNECTING
      ) {
        reconnectAttempt = 0;
        connect();
      }
    };

    document.addEventListener(
      "visibilitychange",
      ensureConnected
    );

    window.addEventListener(
      "online",
      ensureConnected
    );

    /*
     * React StrictMode performs a development-only
     * mount -> cleanup -> mount cycle. A short delay prevents the discarded
     * first effect from creating a socket that is immediately torn down.
     */
    initialConnectTimer = window.setTimeout(() => {
      initialConnectTimer = null;

      if (!cancelled) {
        connect();
      }
    }, 150);

    return () => {
      cancelled = true;

      document.removeEventListener(
        "visibilitychange",
        ensureConnected
      );

      window.removeEventListener(
        "online",
        ensureConnected
      );

      clearInitialConnectTimer();
      clearReconnectTimer();
      clearHeartbeatTimer();

      if (staleTimer !== null) {
        window.clearInterval(staleTimer);
        staleTimer = null;
      }

      const currentSocket = socket;
      socket = null;

      closeSocketSafely(
        currentSocket
      );
    };
  }, []);

  /* ==========================================================
     GIS DATA LOADING
     ========================================================== */

  useEffect(() => {
    let cancelled = false;

    const loadGISData =
      async () => {
        setGisLoading(true);
        setGisError("");

        try {
          const [
            cameraResponse,
            pathResponse,
          ] = await Promise.all([
            getGISCameras(),

            getGISVehiclePaths({
              limit:
                MAX_VEHICLE_PATHS,
            }),
          ]);

          if (cancelled) {
            return;
          }

          setGisCameras(
            Array.isArray(
              cameraResponse?.cameras
            )
              ? cameraResponse.cameras
              : []
          );

          setVehiclePaths(
            Array.isArray(
              pathResponse?.paths
            )
              ? pathResponse.paths
                  .map(
                    normalizeVehiclePath
                  )
                  .filter(Boolean)
              : []
          );
        } catch (error) {
          if (!cancelled) {
            setGisError(
              error?.response?.data
                ?.detail ||
                error?.message ||
                "Unable to load GIS data"
            );
          }
        } finally {
          if (!cancelled) {
            setGisLoading(false);
          }
        }
      };

    loadGISData();

    const intervalId =
      window.setInterval(
        loadGISData,
        15000
      );

    return () => {
      cancelled = true;
      window.clearInterval(
        intervalId
      );
    };
  }, []);

  /* ==========================================================
     MERGE CAMERAS
     ========================================================== */

  const mergedCameras =
    useMemo(() => {
      const map = new Map();

      for (const camera of [
        ...(Array.isArray(
          cameras
        )
          ? cameras
          : []),

        ...(Array.isArray(
          gisCameras
        )
          ? gisCameras
          : []),
      ]) {
        const id = String(
          camera?.cam_id ??
            camera?.camera_id ??
            camera?.id ??
            ""
        ).trim();

        if (!id) {
          continue;
        }

        map.set(id, {
          ...(map.get(id) || {}),
          ...camera,
        });
      }

      return Array.from(
        map.values()
      );
    }, [
      cameras,
      gisCameras,
    ]);

  /* ==========================================================
     VALID CAMERAS
     ========================================================== */

  const validCameras =
    useMemo(
      () =>
        Array.isArray(
          mergedCameras
        )
          ? mergedCameras.filter(
              hasValidCoordinates
            )
          : [],
      [mergedCameras]
    );

  /* ==========================================================
     CAMERA COUNTS
     ========================================================== */

  const cameraCounts =
    useMemo(() => {
      const counts = {
        ONLINE: 0,
        ALERT: 0,
        DEGRADED: 0,
        OFFLINE: 0,
      };

      validCameras.forEach(
        (camera) => {
          const status =
            getCameraStatus(
              camera
            );

          if (
            Object.prototype.hasOwnProperty.call(
              counts,
              status
            )
          ) {
            counts[status] += 1;
          }
        }
      );

      return counts;
    }, [
      validCameras,
    ]);

  /* ==========================================================
     LOCATION SEARCH
     ========================================================== */

  const searchLocation =
    useCallback(
      async (event) => {
        event?.preventDefault?.();

        const query =
          String(locationSearchQuery || "")
            .trim()
            .replace(/\s+/g, " ");

        if (query.length < 2) {
          setLocationSearchResults([]);
          setLocationSearchError(
            "Enter at least 2 characters."
          );
          return;
        }

        setLocationSearchLoading(true);
        setLocationSearchError("");

        try {
          const params = new URLSearchParams({
            q: query,
            format: "jsonv2",
            addressdetails: "1",
            limit: String(LOCATION_SEARCH_LIMIT),
          });

          const response = await fetch(
            `${LOCATION_SEARCH_ENDPOINT}?${params.toString()}`,
            {
              method: "GET",
              headers: {
                Accept: "application/json",
              },
            }
          );

          if (!response.ok) {
            throw new Error(
              `Location search failed (${response.status})`
            );
          }

          const payload = await response.json();

          const results = (
            Array.isArray(payload)
              ? payload
              : []
          )
            .map((item) => {
              const coordinates =
                normalizeCoordinates(
                  item?.lat,
                  item?.lon
                );

              if (!coordinates) {
                return null;
              }

              return {
                id: String(
                  item?.place_id ??
                    `${coordinates.latitude}-${coordinates.longitude}`
                ),
                label: String(
                  item?.display_name ||
                    query
                ).trim(),
                type: String(
                  item?.type ||
                    item?.category ||
                    "location"
                ).trim(),
                ...coordinates,
              };
            })
            .filter(Boolean);

          setLocationSearchResults(
            results
          );

          if (results.length === 0) {
            setLocationSearchError(
              "No matching location found."
            );
          }
        } catch (error) {
          setLocationSearchResults([]);
          setLocationSearchError(
            error?.message ||
              "Unable to search this location."
          );
        } finally {
          setLocationSearchLoading(false);
        }
      },
      [locationSearchQuery]
    );

  const selectSearchResult =
    useCallback(
      (result) => {
        const coordinates =
          normalizeCoordinates(
            result?.latitude,
            result?.longitude
          );

        if (!coordinates) {
          return;
        }

        const location = {
          ...coordinates,
          label: String(
            result?.label ||
              locationSearchQuery ||
              "Searched location"
          ).trim(),
        };

        setSearchedLocation(location);
        setSelectedLocation(location);
        setLocationSearchQuery(
          location.label
        );
        setLocationSearchResults([]);
        setLocationSearchError("");

        if (
          typeof onMapClick ===
          "function"
        ) {
          onMapClick(coordinates);
        }
      },
      [
        locationSearchQuery,
        onMapClick,
      ]
    );

  const clearLocationSearch =
    useCallback(() => {
      setLocationSearchQuery("");
      setLocationSearchResults([]);
      setLocationSearchError("");
      setSearchedLocation(null);
    }, []);

  /* ==========================================================
     CAMERA CLICK
     ========================================================== */

  const handleCameraClick =
    useCallback(
      (camera) => {
        if (
          !camera ||
          typeof onCameraClick !==
            "function"
        ) {
          return;
        }

        onCameraClick(
          camera
        );
      },
      [onCameraClick]
    );

  /* ==========================================================
     MAP CLICK
     ========================================================== */

  const handleMapClick =
    useCallback(
      (coordinates) => {
        const normalized =
          normalizeCoordinates(
            coordinates?.latitude,
            coordinates?.longitude
          );

        if (!normalized) {
          return;
        }

        setSelectedLocation(
          normalized
        );

        if (
          typeof onMapClick ===
          "function"
        ) {
          onMapClick(
            normalized
          );
        }
      },
      [onMapClick]
    );

  /* ==========================================================
     CLEAR LOCATION
     ========================================================== */

  const clearSelectedLocation =
    useCallback(() => {
      setSelectedLocation(
        null
      );
    }, []);

  /* ==========================================================
     MAP VIEW
     ========================================================== */

  const handleMapViewChange =
    useCallback(
      (view) => {
        if (
          !Object.prototype.hasOwnProperty.call(
            TILE_CONFIG,
            view
          )
        ) {
          return;
        }

        setSelectedMapView(
          view
        );
      },
      []
    );

  /* ==========================================================
     VEHICLE JOURNEY
     ========================================================== */

  useEffect(() => {
    let cancelled = false;

    if (!selectedVehicleId) {
      setSelectedVehicleJourney(
        null
      );

      return undefined;
    }

    setGisError("");

    getVehicleJourney(
      selectedVehicleId
    )
      .then((journey) => {
        if (!cancelled) {
          setSelectedVehicleJourney(
            journey
          );
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setSelectedVehicleJourney(
            null
          );

          setGisError(
            error?.response?.data
              ?.detail ||
              error?.message ||
              "Unable to load vehicle journey"
          );
        }
      });

    return () => {
      cancelled = true;
    };
  }, [
    selectedVehicleId,
  ]);

  /* ==========================================================
     VEHICLE CLICK
     ========================================================== */

  const handleVehicleClick =
    useCallback(
      (vehicleId) => {
        const safeId = String(
          vehicleId ?? ""
        ).trim();

        if (
          !safeId ||
          safeId.length > 128 ||
          !/^[A-Za-z0-9_.:-]+$/.test(
            safeId
          )
        ) {
          return;
        }

        setSelectedVehicleId(
          safeId
        );
      },
      []
    );

  /* ==========================================================
     CLEAR VEHICLE
     ========================================================== */

  const clearSelectedVehicle =
    useCallback(() => {
      setSelectedVehicleId(
        null
      );

      setSelectedVehicleJourney(
        null
      );
    }, []);

  /* ==========================================================
     SELECTED PATH
     ========================================================== */

  const selectedPath =
    useMemo(() => {
      if (!selectedVehicleId) {
        return null;
      }

      return (
        vehiclePaths.find(
          (path) =>
            path.global_vehicle_id ===
            selectedVehicleId
        ) || null
      );
    }, [
      selectedVehicleId,
      vehiclePaths,
    ]);

  /* ==========================================================
     SELECTED ROUTE
     ========================================================== */

  const selectedRoute =
    useMemo(() => {
      const route =
        selectedVehicleJourney?.route ||
        selectedPath?.route ||
        [];

      return Array.isArray(
        route
      )
        ? route
            .slice(
              0,
              MAX_ROUTE_POINTS
            )
            .map(
              normalizeVehiclePathPoint
            )
            .filter(Boolean)
        : [];
    }, [
      selectedVehicleJourney,
      selectedPath,
    ]);

  /* ==========================================================
     SELECTED TRANSITIONS
     ========================================================== */

  const selectedTransitions =
    useMemo(() => {
      const journeyTransitions =
        Array.isArray(
          selectedVehicleJourney?.transitions
        )
          ? selectedVehicleJourney.transitions
          : [];

      const journeySegments =
        Array.isArray(
          selectedVehicleJourney?.segments
        )
          ? selectedVehicleJourney.segments
          : [];

      const pathTransitions =
        Array.isArray(
          selectedPath?.transitions
        )
          ? selectedPath.transitions
          : [];

      const pathSegments =
        Array.isArray(
          selectedPath?.segments
        )
          ? selectedPath.segments
          : [];

      const source =
        journeyTransitions.length
          ? journeyTransitions
          : pathTransitions.length
          ? pathTransitions
          : journeySegments.length
          ? journeySegments
          : pathSegments;

      if (source.length) {
        return source
          .slice(
            0,
            MAX_ROUTE_POINTS
          )
          .map(
            (transition, index) => {
              const fallbackSource =
                selectedRoute[
                  index
                ]?.camera_id;

              const fallbackDestination =
                selectedRoute[
                  index + 1
                ]?.camera_id;

              return normalizeTransition(
                transition,
                fallbackSource,
                fallbackDestination
              );
            }
          )
          .filter(Boolean);
      }

      return buildRouteTransitions(
        selectedRoute
      );
    }, [
      selectedVehicleJourney,
      selectedPath,
      selectedRoute,
    ]);

  /* ==========================================================
     SELECTED JOURNEY METRICS
     ========================================================== */

  const selectedJourneyMetrics =
    useMemo(() => {
      const journey =
        selectedVehicleJourney ||
        selectedPath ||
        {};

      return {
        distanceKm:
          numericValue(
            journey.total_distance_km,
            journey.distance_km
          ),

        transitionCount:
          numericValue(
            journey.transition_count
          ),

        lastSimilarity:
          numericValue(
            journey.last_similarity,
            selectedPath?.route?.[
              selectedPath.route.length -
                1
            ]?.similarity
          ),

        lastCorrelation:
          numericValue(
            journey.last_correlation_score,
            selectedPath?.route?.[
              selectedPath.route.length -
                1
            ]?.correlation_score
          ),

        totalObservations:
          selectedRoute.length,

        feasibleTransitions:
          selectedTransitions.filter(
            (transition) =>
              transition.feasible ===
              true
          ).length,

        rejectedTransitions:
          selectedTransitions.filter(
            (transition) =>
              transition.feasible ===
              false
          ).length,
      };
    }, [
      selectedVehicleJourney,
      selectedPath,
      selectedRoute,
      selectedTransitions,
    ]);

  /* ==========================================================
     RENDER
     ========================================================== */

  return (
    <div
      className="
        relative
        isolate
        z-0
        w-full
        min-w-0
        max-w-full
        h-[520px]
        rounded-xl
        overflow-hidden
        border border-[#1f4b7a]
        bg-[#071526]
        [contain:layout_paint]
        [&_.leaflet-container]:!relative
        [&_.leaflet-container]:!z-0
        [&_.leaflet-container]:!h-full
        [&_.leaflet-container]:!w-full
        [&_.leaflet-container]:!min-w-0
        [&_.leaflet-container]:!max-w-full
        [&_.leaflet-top]:!z-[30]
        [&_.leaflet-bottom]:!z-[30]
      "
    >
      {/* ======================================================
          LOCATION SEARCH
          ====================================================== */}

      <div
        className="
          absolute
          top-4
          left-1/2
          -translate-x-1/2
          z-[70]
          w-[min(560px,calc(100%-32px))]
        "
      >
        <form
          onSubmit={searchLocation}
          className="
            flex items-center
            overflow-hidden
            rounded-xl
            border border-[#2c5f91]
            bg-[#071526]/98
            shadow-2xl
            backdrop-blur-md
          "
        >
          <span
            aria-hidden="true"
            className="pl-4 text-lg text-blue-300"
          >
            ⌕
          </span>

          <input
            type="search"
            value={locationSearchQuery}
            onChange={(event) => {
              setLocationSearchQuery(
                event.target.value
              );
              setLocationSearchError("");
              if (!event.target.value.trim()) {
                setLocationSearchResults([]);
              }
            }}
            placeholder="Search location, city or address"
            aria-label="Search map location"
            autoComplete="off"
            className="
              min-w-0 flex-1
              bg-transparent
              px-3 py-3
              text-sm text-white
              placeholder:text-slate-400
              outline-none
            "
          />

          {(locationSearchQuery ||
            searchedLocation) && (
            <button
              type="button"
              onClick={clearLocationSearch}
              aria-label="Clear location search"
              className="
                px-3 py-3
                text-lg leading-none
                text-slate-400
                hover:text-white
              "
            >
              ×
            </button>
          )}

          <button
            type="submit"
            disabled={locationSearchLoading}
            className="
              m-1.5
              rounded-lg
              bg-blue-600
              px-4 py-2
              text-xs font-bold
              text-white
              transition
              hover:bg-blue-500
              disabled:cursor-wait
              disabled:opacity-60
            "
          >
            {locationSearchLoading
              ? "Searching..."
              : "Search"}
          </button>
        </form>

        {(locationSearchResults.length > 0 ||
          locationSearchError) && (
          <div
            className="
              mt-2
              overflow-hidden
              rounded-xl
              border border-[#2c5f91]
              bg-[#071526]/98
              shadow-2xl
              backdrop-blur-md
            "
          >
            {locationSearchError ? (
              <div className="px-4 py-3 text-xs text-red-300">
                {locationSearchError}
              </div>
            ) : (
              locationSearchResults.map(
                (result) => (
                  <button
                    key={result.id}
                    type="button"
                    onClick={() =>
                      selectSearchResult(
                        result
                      )
                    }
                    className="
                      flex w-full
                      items-start gap-3
                      border-b border-white/5
                      px-4 py-3
                      text-left
                      transition
                      last:border-b-0
                      hover:bg-[#102b49]
                      focus:bg-[#102b49]
                      focus:outline-none
                    "
                  >
                    <span
                      aria-hidden="true"
                      className="mt-0.5 text-blue-300"
                    >
                      ●
                    </span>

                    <span className="min-w-0">
                      <span className="block truncate text-sm font-semibold text-white">
                        {result.label}
                      </span>

                      <span className="mt-0.5 block text-[10px] uppercase tracking-wider text-slate-400">
                        {result.type}
                      </span>
                    </span>
                  </button>
                )
              )
            )}

            <div className="border-t border-white/5 px-4 py-2 text-[9px] text-slate-500">
              Search data © OpenStreetMap contributors
            </div>
          </div>
        )}
      </div>

      {/* ======================================================
          MAP VIEW
          ====================================================== */}

      <MapViewSelector
        selectedView={
          selectedMapView
        }
        onViewChange={
          handleMapViewChange
        }
      />

      {/* ======================================================
          CAMERA STATUS
          ====================================================== */}

      <div
        className="
          absolute
          bottom-4
          left-4
          bg-[#071526]/95
          border border-[#1f4b7a]
          rounded-lg
          p-4
          text-white
          shadow-xl
          backdrop-blur-sm
          z-[50]
        "
      >
        <p className="text-sm font-bold text-blue-400 mb-3">
          CAMERA STATUS
        </p>

        <div className="space-y-2 text-xs">
          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-green-500" />
            <span>Online</span>
            <span className="ml-auto text-gray-400">
              {cameraCounts.ONLINE}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-red-500" />
            <span>Alert</span>
            <span className="ml-auto text-gray-400">
              {cameraCounts.ALERT}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-orange-500" />
            <span>Degraded</span>
            <span className="ml-auto text-gray-400">
              {cameraCounts.DEGRADED}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="w-3 h-3 rounded-full bg-slate-500" />
            <span>Offline</span>
            <span className="ml-auto text-gray-400">
              {cameraCounts.OFFLINE}
            </span>
          </div>
        </div>
      </div>

      {/* ======================================================
          GIS CAMERA COUNT
          ====================================================== */}

      <div
        className="
          absolute
          top-[82px]
          right-4
          bg-[#071526]/95
          border border-[#1f4b7a]
          rounded-lg
          px-4 py-3
          text-white
          shadow-xl
          backdrop-blur-sm
          z-[50]
        "
      >
        <p className="text-xs text-gray-400">
          GIS CAMERAS
        </p>

        <p className="text-2xl font-bold text-blue-400">
          {validCameras.length}
        </p>
      </div>

      {/* ======================================================
          SELECTED LOCATION
          ====================================================== */}

      {selectedLocation && (
        <div
          className="
            absolute
            bottom-4
            left-[170px]
            bg-[#071526]/95
            border border-blue-500
            rounded-lg
            px-4 py-3
            text-white
            shadow-xl
            backdrop-blur-sm
          z-[50]
        "
        >
          <div className="flex items-start gap-4">
            <div>
              <p className="text-xs font-bold text-blue-400 uppercase">
                Selected Location
              </p>

              <p className="text-xs mt-1">
                Latitude:{" "}
                <span className="font-bold">
                  {selectedLocation.latitude.toFixed(
                    6
                  )}
                </span>
              </p>

              <p className="text-xs">
                Longitude:{" "}
                <span className="font-bold">
                  {selectedLocation.longitude.toFixed(
                    6
                  )}
                </span>
              </p>
            </div>

            <button
              type="button"
              onClick={
                clearSelectedLocation
              }
              className="
                text-gray-400
                hover:text-white
                text-xl
                leading-none
              "
            >
              ×
            </button>
          </div>
        </div>
      )}

      {/* ======================================================
          VEHICLE CORRELATION LIST
          ====================================================== */}

      <div
        className="
          absolute
          top-[145px]
          right-4
          w-[290px]
          max-h-[250px]
          overflow-auto
          bg-[#071526]/95
          border border-cyan-500/60
          rounded-lg
          p-4
          text-white
          shadow-xl
          backdrop-blur-sm
          z-[50]
        "
      >
        <div className="flex items-center justify-between mb-3">
          <p className="text-xs text-cyan-400 font-bold uppercase tracking-wider">
            Vehicle Correlation
          </p>

          {gisLoading && (
            <span className="text-[10px] text-gray-400">
              Updating...
            </span>
          )}
        </div>

        {gisError && (
          <p className="text-xs text-red-400 mb-2">
            {safeDisplayValue(
              gisError,
              "GIS unavailable"
            )}
          </p>
        )}

        {vehiclePaths.length ===
        0 ? (
          <p className="text-xs text-gray-400">
            No correlated vehicles available.
          </p>
        ) : (
          <div className="space-y-2">
            {vehiclePaths
              .slice(0, 20)
              .map(
                (
                  path,
                  index
                ) => {
                  const selected =
                    selectedVehicleId ===
                    path.global_vehicle_id;

                  return (
                    <button
                      key={
                        path.global_vehicle_id
                      }
                      type="button"
                      onClick={() =>
                        handleVehicleClick(
                          path.global_vehicle_id
                        )
                      }
                      className={`
                        w-full text-left
                        rounded-md px-3 py-2
                        border transition
                        focus:outline-none
                        focus:ring-2
                        focus:ring-cyan-400
                        ${
                          selected
                            ? "border-cyan-400 bg-cyan-950/70"
                            : "border-[#1f4b7a] bg-[#0b2038] hover:bg-[#102b49]"
                        }
                      `}
                    >
                      <div className="flex items-center gap-2">
                        <span
                          className="w-2.5 h-2.5 rounded-full"
                          style={{
                            backgroundColor:
                              getVehiclePathColor(
                                index
                              ),
                          }}
                        />

                        <span className="font-bold text-xs truncate">
                          {
                            path.global_vehicle_id
                          }
                        </span>

                        {path.strong_match && (
                          <span className="ml-auto text-[9px] text-green-400">
                            STRONG
                          </span>
                        )}
                      </div>

                      <div className="grid grid-cols-2 gap-1 mt-2 text-[10px] text-gray-400">
                        <span>
                          Type:{" "}
                          {
                            path.vehicle_type
                          }
                        </span>

                        <span>
                          Transitions:{" "}
                          {
                            path.transition_count
                          }
                        </span>

                        <span>
                          Points:{" "}
                          {
                            path.route.length
                          }
                        </span>

                        <span>
                          Last:{" "}
                          {formatVehicleTimestamp(
                            path.last_seen
                          )}
                        </span>
                      </div>
                    </button>
                  );
                }
              )}
          </div>
        )}
      </div>

      {/* ======================================================
          SELECTED VEHICLE INTELLIGENCE
          ====================================================== */}

      {selectedVehicleId && (
        <div
          className="
            absolute
            bottom-4
            right-4
            w-[330px]
            max-h-[430px]
            overflow-auto
            bg-[#071526]/97
            border border-cyan-500
            rounded-lg
            px-4 py-3
            text-white
            shadow-2xl
            backdrop-blur-sm
          z-[50]
        "
        >
          {/* HEADER */}

          <div className="flex items-start justify-between">
            <div>
              <p className="text-[10px] text-cyan-400 font-bold uppercase tracking-wider">
                GIS Intelligence
              </p>

              <p className="font-bold text-sm mt-1">
                {selectedVehicleId}
              </p>
            </div>

            <button
              type="button"
              onClick={
                clearSelectedVehicle
              }
              className="
                text-gray-400
                hover:text-white
                text-xl
                leading-none
              "
            >
              ×
            </button>
          </div>

          {/* SUMMARY */}

          <div className="grid grid-cols-2 gap-2 mt-4">
            <div className="rounded-md bg-[#0b2038] border border-[#1f4b7a] p-2">
              <span className="text-[9px] text-gray-400">
                VEHICLE TYPE
              </span>

              <p className="text-xs font-bold mt-1">
                {safeDisplayValue(
                  selectedVehicleJourney?.vehicle_type ||
                    selectedPath?.vehicle_type,
                  "vehicle"
                )}
              </p>
            </div>

            <div className="rounded-md bg-[#0b2038] border border-[#1f4b7a] p-2">
              <span className="text-[9px] text-gray-400">
                OBSERVATIONS
              </span>

              <p className="text-xs font-bold mt-1">
                {
                  selectedJourneyMetrics.totalObservations
                }
              </p>
            </div>

            <div className="rounded-md bg-[#0b2038] border border-[#1f4b7a] p-2">
              <span className="text-[9px] text-gray-400">
                SIMILARITY
              </span>

              <p className="text-xs font-bold text-cyan-300 mt-1">
                {formatPercent(
                  selectedJourneyMetrics.lastSimilarity
                )}
              </p>
            </div>

            <div className="rounded-md bg-[#0b2038] border border-[#1f4b7a] p-2">
              <span className="text-[9px] text-gray-400">
                CORRELATION
              </span>

              <p className="text-xs font-bold text-cyan-300 mt-1">
                {formatPercent(
                  selectedJourneyMetrics.lastCorrelation
                )}
              </p>
            </div>
          </div>

          {/* JOURNEY METRICS */}

          <div className="mt-4">
            <p className="text-[10px] font-bold text-blue-400 uppercase tracking-wider mb-2">
              Journey Metrics
            </p>

            <IntelligenceRow
              label="Total Distance"
              value={
                selectedJourneyMetrics.distanceKm ===
                null
                  ? "N/A"
                  : `${formatNumber(
                      selectedJourneyMetrics.distanceKm
                    )} km`
              }
            />

            <IntelligenceRow
              label="Transitions"
              value={
                selectedJourneyMetrics.transitionCount ??
                "N/A"
              }
            />

            <IntelligenceRow
              label="Feasible Transitions"
              value={
                selectedJourneyMetrics.feasibleTransitions
              }
              valueClassName="text-green-400"
            />

            <IntelligenceRow
              label="Rejected Transitions"
              value={
                selectedJourneyMetrics.rejectedTransitions
              }
              valueClassName={
                selectedJourneyMetrics.rejectedTransitions >
                0
                  ? "text-red-400"
                  : "text-gray-200"
              }
            />
          </div>

          {/* TRANSITION INTELLIGENCE */}

          <div className="mt-4">
            <div className="flex items-center justify-between mb-2">
              <p className="text-[10px] font-bold text-cyan-400 uppercase tracking-wider">
                Transition Intelligence
              </p>

              <span className="text-[9px] text-gray-500">
                {
                  selectedTransitions.length
                } evaluated
              </span>
            </div>

            {selectedTransitions.length ===
            0 ? (
              <div className="rounded-md bg-[#0b2038] border border-[#1f4b7a] p-3">
                <p className="text-[10px] text-gray-400">
                  No transition-level GIS metrics were returned by the current API response.
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {selectedTransitions
                  .slice(0, 12)
                  .map(
                    (
                      transition,
                      index
                    ) => (
                      <div
                        key={`${transition.source_camera}-${transition.destination_camera}-${index}`}
                        className="
                          rounded-md
                          bg-[#0b2038]
                          border
                          border-[#1f4b7a]
                          p-3
                        "
                      >
                        <div className="flex items-center justify-between gap-2 mb-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="text-[9px] text-gray-500">
                              {index + 1}
                            </span>

                            <span className="text-[10px] font-bold truncate">
                              {safeDisplayValue(
                                transition.source_camera
                              )}
                            </span>

                            <span className="text-cyan-400">
                              →
                            </span>

                            <span className="text-[10px] font-bold truncate">
                              {safeDisplayValue(
                                transition.destination_camera
                              )}
                            </span>
                          </div>

                          <GISDecisionBadge
                            transition={
                              transition
                            }
                          />
                        </div>

                        <IntelligenceRow
                          label="Distance"
                          value={
                            transition.distance_km !==
                              null &&
                            transition.distance_km !==
                              undefined
                              ? `${formatNumber(
                                  transition.distance_km
                                )} km`
                              : transition.distance_meters !==
                                  null &&
                                transition.distance_meters !==
                                  undefined
                              ? `${formatNumber(
                                  transition.distance_meters
                                )} m`
                              : "N/A"
                          }
                        />

                        <IntelligenceRow
                          label="Elapsed Time"
                          value={formatElapsedTime(
                            transition.elapsed_seconds
                          )}
                        />

                        <IntelligenceRow
                          label="Implied Speed"
                          value={
                            transition.implied_speed_kmh ===
                            null
                              ? "N/A"
                              : `${formatNumber(
                                  transition.implied_speed_kmh
                                )} km/h`
                          }
                        />

                        <IntelligenceRow
                          label="Bearing"
                          value={
                            transition.bearing_degrees ===
                            null
                              ? "N/A"
                              : `${formatNumber(
                                  transition.bearing_degrees,
                                  1
                                )}°`
                          }
                        />

                        <IntelligenceRow
                          label="Direction Score"
                          value={formatPercent(
                            transition.direction_score
                          )}
                        />

                        <IntelligenceRow
                          label="Travel-Time Score"
                          value={formatPercent(
                            transition.travel_time_score
                          )}
                        />

                        <IntelligenceRow
                          label="Distance Score"
                          value={formatPercent(
                            transition.distance_score
                          )}
                        />

                        <IntelligenceRow
                          label="GIS Score"
                          value={formatPercent(
                            transition.gis_score
                          )}
                          valueClassName="text-cyan-300"
                        />

                        {transition.reason && (
                          <div className="mt-2 pt-2 border-t border-white/5">
                            <span className="text-[9px] text-gray-500">
                              Decision Reason
                            </span>

                            <p className="text-[10px] text-gray-300 mt-1 break-words">
                              {
                                transition.reason
                              }
                            </p>
                          </div>
                        )}
                      </div>
                    )
                  )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ======================================================
          LEAFLET MAP
          ====================================================== */}

      <MapContainer
        center={DEFAULT_CENTER}
        zoom={DEFAULT_ZOOM}
        scrollWheelZoom
        doubleClickZoom
        dragging
        touchZoom
        zoomControl
        className="!relative !z-0 !h-full !w-full !min-w-0 !max-w-full"
      >
        <MapTiles
          selectedView={
            selectedMapView
          }
        />

        <MapController
          selectedCamera={
            selectedCamera
          }
        />

        <SearchLocationController
          location={searchedLocation}
        />

        <MapClickHandler
          onMapClick={
            handleMapClick
          }
        />

        {/* ====================================================
            VEHICLE ROUTES
            ==================================================== */}

        {vehiclePaths.map(
          (
            path,
            pathIndex
          ) => {
            const route =
              path.route || [];

            const coordinates =
              routeCoordinates(
                route
              );

            if (
              !coordinates.length
            ) {
              return null;
            }

            const color =
              getVehiclePathColor(
                pathIndex
              );

            const selected =
              selectedVehicleId ===
              path.global_vehicle_id;

            return (
              <React.Fragment
                key={
                  path.global_vehicle_id
                }
              >
                {/* MAIN ROUTE */}

                {coordinates.length >=
                  2 && (
                  <Polyline
                    positions={
                      coordinates
                    }
                    pathOptions={{
                      color,
                      weight:
                        selected
                          ? 7
                          : 3,
                      opacity:
                        selected
                          ? 0.98
                          : 0.55,
                      dashArray:
                        selected
                          ? undefined
                          : "8 8",
                    }}
                    eventHandlers={{
                      click: () =>
                        handleVehicleClick(
                          path.global_vehicle_id
                        ),
                    }}
                  />
                )}

                {/* ROUTE POINTS */}

                {route.map(
                  (
                    point,
                    pointIndex
                  ) => {
                    const isLast =
                      pointIndex ===
                      route.length -
                        1;

                    return (
                      <CircleMarker
                        key={`${path.global_vehicle_id}-${point.camera_id}-${point.timestamp}-${pointIndex}`}
                        center={[
                          point.latitude,
                          point.longitude,
                        ]}
                        radius={
                          isLast
                            ? 7
                            : selected
                            ? 5
                            : 4
                        }
                        pathOptions={{
                          color,
                          fillColor:
                            color,
                          fillOpacity:
                            selected
                              ? 0.95
                              : 0.7,
                          weight: 2,
                        }}
                        eventHandlers={{
                          click: () =>
                            handleVehicleClick(
                              path.global_vehicle_id
                            ),
                        }}
                      >
                        <Tooltip>
                          <div className="text-xs">
                            <strong>
                              {
                                path.global_vehicle_id
                              }
                            </strong>

                            <br />

                            Camera:{" "}
                            {
                              point.camera_id
                            }

                            <br />

                            Similarity:{" "}
                            {formatPercent(
                              point.similarity,
                              0
                            )}

                            <br />

                            Correlation:{" "}
                            {formatPercent(
                              point.correlation_score,
                              0
                            )}

                            <br />

                            Position:{" "}
                            {point.position_type === "OBSERVED_CALIBRATED"
                              ? "Observed"
                              : "Camera anchor"}

                            <br />

                            Seen:{" "}
                            {formatVehicleTimestamp(
                              point.timestamp
                            )}
                          </div>
                        </Tooltip>

                        <Popup>
                          <div className="min-w-[240px] text-sm">
                            <h3 className="font-bold text-base mb-2">
                              Vehicle Correlation
                            </h3>

                            <div className="space-y-1">
                              <p>
                                <strong>
                                  Global ID:
                                </strong>{" "}
                                {
                                  path.global_vehicle_id
                                }
                              </p>

                              <p>
                                <strong>
                                  Camera:
                                </strong>{" "}
                                {
                                  point.camera_id
                                }
                              </p>

                              <p>
                                <strong>
                                  Similarity:
                                </strong>{" "}
                                {formatPercent(
                                  point.similarity
                                )}
                              </p>

                              <p>
                                <strong>
                                  Correlation:
                                </strong>{" "}
                                {formatPercent(
                                  point.correlation_score
                                )}
                              </p>

                              <div className="mt-2 rounded border border-slate-200 bg-slate-50 p-2">
                                <div className="font-semibold mb-1">Evidence fusion</div>
                                <div>ANPR/LPR: {formatPercent(point.anpr_score, 0)}</div>
                                <div>Vehicle Re-ID: {formatPercent(point.reid_score, 0)}</div>
                                <div>Metadata: {formatPercent(point.metadata_score, 0)}</div>
                              </div>

                              <p>
                                <strong>
                                  Match:
                                </strong>{" "}
                                {point.strong_match
                                  ? "Strong"
                                  : "Supporting"}
                              </p>

                              <p>
                                <strong>
                                  Seen:
                                </strong>{" "}
                                {formatVehicleTimestamp(
                                  point.timestamp
                                )}
                              </p>
                            </div>

                            <button
                              type="button"
                              onClick={() =>
                                handleVehicleClick(
                                  path.global_vehicle_id
                                )
                              }
                              className="
                                w-full mt-3
                                px-3 py-2
                                rounded-md
                                bg-blue-600
                                hover:bg-blue-700
                                text-white
                                font-bold
                              "
                            >
                              View Journey
                            </button>
                          </div>
                        </Popup>
                      </CircleMarker>
                    );
                  }
                )}

                {/* LAST SEEN VEHICLE */}

                <Marker
                  position={(() => {
                    const live =
                      liveVehiclePositions[
                        path.global_vehicle_id
                      ];
                    const last =
                      route[
                        route.length - 1
                      ];

                    return [
                      Number(
                        live?.latitude ??
                          last.latitude
                      ),
                      Number(
                        live?.longitude ??
                          last.longitude
                      ),
                    ];
                  })()}
                  icon={createVehicleIcon(
                    color,
                    selected
                  )}
                  eventHandlers={{
                    click: () =>
                      handleVehicleClick(
                        path.global_vehicle_id
                      ),
                  }}
                >
                  <Popup>
                    <div className="min-w-[240px] text-sm">
                      <h3 className="font-bold text-base mb-2">
                        Correlated Vehicle
                      </h3>

                      <div className="space-y-1">
                        <p>
                          <strong>
                            Global ID:
                          </strong>{" "}
                          {
                            path.global_vehicle_id
                          }
                        </p>

                        <p>
                          <strong>
                            Type:
                          </strong>{" "}
                          {
                            path.vehicle_type
                          }
                        </p>

                        <p>
                          <strong>
                            Last Camera:
                          </strong>{" "}
                          {
                            route[
                              route.length -
                                1
                            ].camera_id
                          }
                        </p>

                        <p>
                          <strong>
                            Match:
                          </strong>{" "}
                          {path.strong_match
                            ? "Strong"
                            : "Supporting"}
                        </p>

                        <p>
                          <strong>
                            Transitions:
                          </strong>{" "}
                          {
                            path.transition_count
                          }
                        </p>

                        <p>
                          <strong>
                            Last Seen:
                          </strong>{" "}
                          {formatVehicleTimestamp(
                            path.last_seen
                          )}
                        </p>
                        {liveVehiclePositions[path.global_vehicle_id] && (
                          <>
                            <p>
                              <strong>Live:</strong>{" "}
                              {liveVehiclePositions[path.global_vehicle_id].position_type === "OBSERVED_CALIBRATED"
                                ? "Observed"
                                : "Camera anchor"}
                            </p>
                            {liveVehiclePositions[path.global_vehicle_id].speed_kmh !== null && (
                              <p>
                                <strong>Speed:</strong>{" "}
                                {formatNumber(
                                  liveVehiclePositions[path.global_vehicle_id].speed_kmh,
                                  1
                                )} km/h
                              </p>
                            )}
                          </>
                        )}
                      </div>
                    </div>
                  </Popup>
                </Marker>
              </React.Fragment>
            );
          }
        )}

        {/* ====================================================
            SELECTED ROUTE HIGHLIGHT
            ==================================================== */}

        {selectedRoute.length >=
          2 && (
          <Polyline
            positions={routeCoordinates(
              selectedRoute
            )}
            pathOptions={{
              color: "#ffffff",
              weight: 8,
              opacity: 0.95,
            }}
          />
        )}

        {selectedVehicleId &&
          selectedRoute.map(
            (
              point,
              index
            ) => {
              if (
                index ===
                selectedRoute.length -
                  1
              ) {
                return null;
              }

              const transition =
                selectedTransitions[
                  index
                ];

              if (
                !transition
              ) {
                return null;
              }

              return (
                <CircleMarker
                  key={`transition-${selectedVehicleId}-${index}`}
                  center={[
                    point.latitude,
                    point.longitude,
                  ]}
                  radius={8}
                  pathOptions={{
                    color:
                      transition.feasible ===
                      false
                        ? "#ef4444"
                        : "#22d3ee",
                    fillColor:
                      transition.feasible ===
                      false
                        ? "#ef4444"
                        : "#22d3ee",
                    fillOpacity: 0.9,
                    weight: 2,
                  }}
                >
                  <Tooltip>
                    <div className="text-xs">
                      <strong>
                        GIS Transition
                      </strong>

                      <br />

                      {
                        transition.source_camera
                      }

                      {" → "}

                      {
                        transition.destination_camera
                      }

                      <br />

                      GIS Score:{" "}
                      {formatPercent(
                        transition.gis_score
                      )}

                      <br />

                      Travel:{" "}
                      {transition.feasible ===
                      false
                        ? "REJECTED"
                        : transition.feasible ===
                          true
                        ? "FEASIBLE"
                        : "N/A"}
                    </div>
                  </Tooltip>
                </CircleMarker>
              );
            }
          )}

        {/* ====================================================
            SELECTED LOCATION
            ==================================================== */}

        {selectedLocation && (
          <Marker
            position={[
              selectedLocation.latitude,
              selectedLocation.longitude,
            ]}
            icon={
              createSelectedLocationIcon()
            }
          >
            <Popup>
              <div className="min-w-[210px] text-sm">
                <h3 className="font-bold text-base mb-2">
                  Selected Location
                </h3>

                <div className="space-y-1">
                  <p>
                    <strong>
                      Latitude:
                    </strong>{" "}
                    {selectedLocation.latitude.toFixed(
                      6
                    )}
                  </p>

                  <p>
                    <strong>
                      Longitude:
                    </strong>{" "}
                    {selectedLocation.longitude.toFixed(
                      6
                    )}
                  </p>
                </div>
              </div>
            </Popup>
          </Marker>
        )}

        {/* ====================================================
            LIVE VEHICLE POSITIONS
            ==================================================== */}

        {Object.values(liveVehiclePositions)
          .filter(
            (live) =>
              live &&
              !vehiclePaths.some(
                (path) =>
                  path.global_vehicle_id ===
                  live.global_vehicle_id
              )
          )
          .slice(0, 200)
          .map((live) => (
            <Marker
              key={`live-${live.global_vehicle_id}`}
              position={[
                live.latitude,
                live.longitude,
              ]}
              icon={createVehicleIcon(
                "#00e5ff",
                selectedVehicleId === live.global_vehicle_id
              )}
              eventHandlers={{
                click: () =>
                  handleVehicleClick(
                    live.global_vehicle_id
                  ),
              }}
            >
              <Tooltip>
                <div className="text-xs">
                  <strong>
                    {live.global_vehicle_id}
                  </strong>
                  <br />
                  {live.position_type ===
                  "OBSERVED_CALIBRATED"
                    ? "LIVE OBSERVED"
                    : "CAMERA ANCHOR"}
                  <br />
                  Camera: {live.camera_id || "—"}
                  {live.speed_kmh !== null && (
                    <>
                      <br />
                      Speed:{" "}
                      {formatNumber(
                        live.speed_kmh,
                        1
                      )} km/h
                    </>
                  )}
                </div>
              </Tooltip>
            </Marker>
          ))}

        {Object.values(livePersonPositions).slice(0, 200).map((live) => (
          <CircleMarker
            key={`live-person-${live.global_person_id}`}
            center={[live.latitude, live.longitude]}
            radius={8}
            pathOptions={{ color: "#f97316", fillColor: "#fb923c", fillOpacity: 0.85, weight: 2 }}
          >
            <Tooltip>
              <div className="text-xs">
                <strong>{live.global_person_id}</strong><br />
                {live.position_type === "OBSERVED_CALIBRATED" ? "LIVE PERSON - OBSERVED" : "PERSON CAMERA ANCHOR"}<br />
                Camera: {live.camera_id || "—"}
              </div>
            </Tooltip>
          </CircleMarker>
        ))}

        {/* ====================================================
            CAMERA MARKERS
            ==================================================== */}

        {validCameras.map(
          (camera) => {
            const cameraId =
              safeDisplayValue(
                camera?.cam_id ||
                  camera?.camera_id ||
                  camera?.id,
                ""
              );

            if (!cameraId) {
              return null;
            }

            const status =
              getCameraStatus(
                camera
              );

            const latitude =
              Number(
                camera.latitude
              );

            const longitude =
              Number(
                camera.longitude
              );

            if (
              !isValidLatitude(
                latitude
              ) ||
              !isValidLongitude(
                longitude
              )
            ) {
              return null;
            }

            const selected =
              String(
                selectedCamera?.cam_id ||
                  selectedCamera?.camera_id ||
                  selectedCamera?.id ||
                  ""
              ) ===
              String(
                camera?.cam_id ||
                  camera?.camera_id ||
                  camera?.id ||
                  ""
              );

            const cameraName =
              safeDisplayValue(
                camera?.camera_name,
                cameraId
              );

            const location =
              safeDisplayValue(
                camera?.location_name,
                "Location unavailable"
              );

            return (
              <Marker
                key={cameraId}
                position={[
                  latitude,
                  longitude,
                ]}
                icon={createCameraIcon(
                  status,
                  selected
                )}
                eventHandlers={{
                  click: () =>
                    handleCameraClick(
                      camera
                    ),
                }}
              >
                <Popup>
                  <div className="min-w-[230px] text-sm">
                    <h3 className="font-bold text-base mb-3">
                      {cameraName}
                    </h3>

                    <div className="space-y-1">
                      <p>
                        <strong>
                          Camera ID:
                        </strong>{" "}
                        {cameraId}
                      </p>

                      <p>
                        <strong>
                          Location:
                        </strong>{" "}
                        {location}
                      </p>

                      <p>
                        <strong>
                          Zone:
                        </strong>{" "}
                        {safeDisplayValue(
                          camera?.zone,
                          "Not assigned"
                        )}
                      </p>

                      <p>
                        <strong>
                          Status:
                        </strong>{" "}
                        <span
                          style={{
                            color:
                              getStatusColor(
                                status
                              ),
                            fontWeight: 700,
                          }}
                        >
                          {status}
                        </span>
                      </p>

                      <p>
                        <strong>
                          Latitude:
                        </strong>{" "}
                        {latitude.toFixed(
                          6
                        )}
                      </p>

                      <p>
                        <strong>
                          Longitude:
                        </strong>{" "}
                        {longitude.toFixed(
                          6
                        )}
                      </p>
                    </div>

                    {typeof onCameraClick ===
                      "function" && (
                      <button
                        type="button"
                        onClick={() =>
                          handleCameraClick(
                            camera
                          )
                        }
                        className="
                          w-full
                          mt-3
                          px-3 py-2
                          rounded-md
                          bg-blue-600
                          hover:bg-blue-700
                          text-white
                          font-bold
                        "
                      >
                        Focus Camera
                      </button>
                    )}
                  </div>
                </Popup>
              </Marker>
            );
          }
        )}
      </MapContainer>
    </div>
  );
};

export default GISMap;