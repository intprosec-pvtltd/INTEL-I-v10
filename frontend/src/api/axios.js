import axios from "axios";


const IS_DEVELOPMENT =
  import.meta.env.DEV;


const CONFIGURED_API_BASE_URL =
  String(
    import.meta.env.VITE_BASE_URL || ""
  )
    .trim()
    .replace(/\/+$/, "");

const NORMALIZED_API_BASE_URL =
  CONFIGURED_API_BASE_URL;


export const API_URL =
  NORMALIZED_API_BASE_URL;



const CONFIGURED_WEBSOCKET_URL =
  String(
    import.meta.env.VITE_WEBSOCKET_URL || ""
  )
    .trim()
    .replace(/\/+$/, "");

const getBrowserWebSocketBase = () => {
  if (
    typeof window === "undefined" ||
    !window.location
  ) {
    return "";
  }


  const protocol =
    window.location.protocol === "https:"
      ? "wss:"
      : "ws:";


  return `${protocol}//${window.location.host}`;
};


const getApiWebSocketBase = () => {
  if (!NORMALIZED_API_BASE_URL) {
    return "";
  }


  return NORMALIZED_API_BASE_URL
    .replace(
      /^https:/i,
      "wss:"
    )
    .replace(
      /^http:/i,
      "ws:"
    );
};


const normalizeWebSocketUrl = (
  configuredUrl
) => {
  const value =
    String(
      configuredUrl || ""
    ).trim();


  /*
   * Already absolute:
   *
   * ws://...
   * wss://...
   */
  if (
    /^wss?:\/\//i.test(
      value
    )
  ) {
    return value;
  }


  /*
   * Relative WebSocket route.
   *
   * Example:
   *
   * /ws
   *
   * becomes:
   *
   * ws://localhost:5173/ws
   *
   * Vite then proxies /ws to AWS.
   */
  if (value) {
    const browserBase =
      getBrowserWebSocketBase();


    const path =
      `/${value.replace(
        /^\/+/,
        ""
      )}`;


    return browserBase
      ? `${browserBase}${path}`
      : path;
  }


  /*
   * If API URL is explicitly configured,
   * derive WebSocket from API URL.
   */
  const apiWebSocketBase =
    getApiWebSocketBase();


  if (apiWebSocketBase) {
    return `${apiWebSocketBase}/ws`;
  }


  /*
   * Default development behavior:
   *
   * ws://localhost:5173/ws
   */
  const browserBase =
    getBrowserWebSocketBase();


  return browserBase
    ? `${browserBase}/ws`
    : "/ws";
};


export const WEBSOCKET_URL =
  normalizeWebSocketUrl(
    CONFIGURED_WEBSOCKET_URL
  );


/* ============================================================
 * API URL BUILDER
 * ============================================================ */


export const createApiUrl = (
  path = ""
) => {
  const normalizedPath =
    String(path)
      .trim()
      .replace(
        /^\/+/,
        ""
      );


  if (!normalizedPath) {
    return (
      NORMALIZED_API_BASE_URL ||
      ""
    );
  }


  /*
   * Explicit backend configured.
   */
  if (
    NORMALIZED_API_BASE_URL
  ) {
    return `${NORMALIZED_API_BASE_URL}/${normalizedPath}`;
  }


  /*
   * Development through Vite proxy.
   */
  return `/${normalizedPath}`;
};


/* ============================================================
 * CAMERA STREAM URL
 * ============================================================ */


export const createCameraStreamUrl = (
  cameraId,
  cacheBuster = Date.now()
) => {
  const normalizedCameraId =
    String(
      cameraId ?? ""
    ).trim();


  if (!normalizedCameraId) {
    throw new Error(
      "Camera ID is required"
    );
  }


  return createApiUrl(
    `stream/${encodeURIComponent(
      normalizedCameraId
    )}?t=${encodeURIComponent(
      cacheBuster
    )}`
  );
};


/* ============================================================
 * DEVELOPMENT INFORMATION
 * ============================================================ */


if (IS_DEVELOPMENT) {
  console.info(
    "[INTEL-I][API] Backend:",
    NORMALIZED_API_BASE_URL ||
      "Vite proxy -> AWS"
  );


  console.info(
    "[INTEL-I][WS] WebSocket:",
    WEBSOCKET_URL
  );
}

/* ============================================================
 * STORAGE
 * ============================================================ */

const CSRF_STORAGE_KEY = "csrf_token";
const USER_STORAGE_KEY = "user";


/* ============================================================
 * UNSAFE HTTP METHODS
 * ============================================================ */

const unsafeMethods = [
  "post",
  "put",
  "patch",
  "delete",
];


/* ============================================================
 * AXIOS INSTANCE
 * ============================================================ */

const api = axios.create({
  baseURL: NORMALIZED_API_BASE_URL,
  withCredentials: true,
  timeout: 30000,
});


/* ============================================================
 * COOKIE HELPERS
 * ============================================================ */

const getCookie = (name) => {
  if (typeof document === "undefined") {
    return null;
  }

  const encodedName = `${name}=`;

  const row = document.cookie
    .split("; ")
    .find((item) =>
      item.startsWith(encodedName)
    );

  if (!row) {
    return null;
  }

  return row.substring(
    encodedName.length
  );
};


/* ============================================================
 * CSRF
 * ============================================================ */

export const getCsrfToken = () => {
  const cookieToken =
    getCookie("csrf_token");

  if (cookieToken) {
    return cookieToken;
  }

  if (
    typeof window !== "undefined"
  ) {
    return localStorage.getItem(
      CSRF_STORAGE_KEY
    );
  }

  return null;
};


export const saveCsrfToken = (
  token
) => {
  if (
    typeof token !== "string" ||
    !token.trim()
  ) {
    return;
  }

  if (
    typeof window === "undefined"
  ) {
    return;
  }

  localStorage.setItem(
    CSRF_STORAGE_KEY,
    token.trim()
  );
};


export const clearAuthStorage = () => {
  if (
    typeof window === "undefined"
  ) {
    return;
  }

  localStorage.removeItem(
    USER_STORAGE_KEY
  );

  localStorage.removeItem(
    CSRF_STORAGE_KEY
  );
};


/* ============================================================
 * FETCH CSRF TOKEN
 * ============================================================ */

export const fetchCsrfToken =
  async () => {
    const response =
      await api.get(
        "/auth/csrf"
      );

    if (
      response.data?.csrf_token
    ) {
      saveCsrfToken(
        response.data.csrf_token
      );
    }

    return response.data
      ?.csrf_token;
  };


/* ============================================================
 * REQUEST INTERCEPTOR
 * ============================================================ */

api.interceptors.request.use(
  (config) => {
    const method =
      String(
        config.method || "get"
      ).toLowerCase();


    /* --------------------------------------------------------
     * Development URL protection
     * -------------------------------------------------------- */

    if (
      import.meta.env.DEV &&
      config.url
    ) {
      const base =
        config.baseURL ||
        NORMALIZED_API_BASE_URL;

      const normalizedBase =
        String(base)
          .replace(/\/+$/, "");

      const normalizedPath =
        String(config.url)
          .replace(/^\/+/, "");

      const finalUrl =
        `${normalizedBase}/${normalizedPath}`;


      if (
        finalUrl.startsWith(
          "http://localhost:5173"
        ) ||
        finalUrl.startsWith(
          "http://127.0.0.1:5173"
        )
      ) {
        console.error(
          "[INTEL-I][API][ERROR] Request incorrectly targeting Vite:",
          finalUrl
        );
      }
    }


    /* --------------------------------------------------------
     * CSRF
     * -------------------------------------------------------- */

    if (
      unsafeMethods.includes(
        method
      )
    ) {
      const csrfToken =
        getCsrfToken();

      if (csrfToken) {
        config.headers =
          config.headers || {};

        config.headers[
          "X-CSRF-Token"
        ] = csrfToken;
      }
    }

    return config;
  },

  (error) =>
    Promise.reject(error)
);


/* ============================================================
 * RESPONSE INTERCEPTOR
 * ============================================================ */

api.interceptors.response.use(
  (response) => {
    if (
      response.data?.csrf_token
    ) {
      saveCsrfToken(
        response.data.csrf_token
      );
    }

    return response;
  },

  async (error) => {
    const originalRequest =
      error.config;

    if (!originalRequest) {
      return Promise.reject(
        error
      );
    }

    const url =
      String(
        originalRequest.url || ""
      );

    const isAuthRoute =
      url.includes(
        "/auth/login"
      ) ||
      url.includes(
        "/auth/refresh"
      ) ||
      url.includes(
        "/auth/logout"
      ) ||
      url.includes(
        "/auth/csrf"
      );


    /* --------------------------------------------------------
     * Authentication refresh
     * -------------------------------------------------------- */

    if (
      error.response?.status ===
        401 &&
      !originalRequest._retry &&
      !isAuthRoute
    ) {
      originalRequest._retry =
        true;

      try {
        let csrfToken =
          getCsrfToken();

        if (!csrfToken) {
          csrfToken =
            await fetchCsrfToken();
        }

        const refreshResponse =
          await api.post(
            "/auth/refresh"
          );

        if (
          refreshResponse.data
            ?.csrf_token
        ) {
          saveCsrfToken(
            refreshResponse.data
              .csrf_token
          );
        }

        return api(
          originalRequest
        );
      } catch (
        refreshError
      ) {
        console.error(
          "Authentication refresh failed:",
          refreshError
        );

        clearAuthStorage();

        if (
          typeof window !==
            "undefined" &&
          window.location
            .pathname !== "/"
        ) {
          window.location.href =
            "/";
        }

        return Promise.reject(
          refreshError
        );
      }
    }

    return Promise.reject(
      error
    );
  }
);


/* ============================================================
 * SAFE GIS / VEHICLE IDENTIFIERS
 * ============================================================ */

const MAX_CAMERA_ID_LENGTH =
  128;

const MAX_GLOBAL_VEHICLE_ID_LENGTH =
  128;

const CAMERA_ID_PATTERN =
  /^[A-Za-z0-9_.:-]+$/;

const GLOBAL_VEHICLE_ID_PATTERN =
  /^[A-Za-z0-9_.:-]+$/;


const normalizeIdentifier = (
  value,
  pattern,
  maxLength,
  fieldName
) => {
  const normalized =
    String(value ?? "").trim();

  if (!normalized) {
    throw new Error(
      `${fieldName} is required`
    );
  }

  if (
    normalized.length >
    maxLength
  ) {
    throw new Error(
      `${fieldName} is invalid`
    );
  }

  if (
    !pattern.test(normalized)
  ) {
    throw new Error(
      `${fieldName} is invalid`
    );
  }

  return normalized;
};


const normalizeCameraId = (
  cameraId
) =>
  normalizeIdentifier(
    cameraId,
    CAMERA_ID_PATTERN,
    MAX_CAMERA_ID_LENGTH,
    "Camera ID"
  );


const normalizeGlobalVehicleId = (
  globalVehicleId
) =>
  normalizeIdentifier(
    globalVehicleId,
    GLOBAL_VEHICLE_ID_PATTERN,
    MAX_GLOBAL_VEHICLE_ID_LENGTH,
    "Global vehicle ID"
  );


const toFiniteNumber = (
  value,
  fallback = null,
  minimum = null,
  maximum = null
) => {
  const number =
    Number(value);

  if (
    !Number.isFinite(number)
  ) {
    return fallback;
  }

  if (
    minimum !== null &&
    number < minimum
  ) {
    return fallback;
  }

  if (
    maximum !== null &&
    number > maximum
  ) {
    return fallback;
  }

  return number;
};


/* ============================================================
 * GIS CAMERA NORMALIZATION
 * ============================================================ */

const normalizeCameraGIS = (
  camera
) => {
  if (
    !camera ||
    typeof camera !== "object"
  ) {
    return null;
  }

  const latitude =
    toFiniteNumber(
      camera.latitude,
      null,
      -90,
      90
    );

  const longitude =
    toFiniteNumber(
      camera.longitude,
      null,
      -180,
      180
    );

  if (
    latitude === null ||
    longitude === null
  ) {
    return null;
  }

  const cameraId =
    camera.cam_id ??
    camera.camera_id ??
    camera.id;

  if (
    cameraId === null ||
    cameraId === undefined
  ) {
    return null;
  }

  let safeCameraId;

  try {
    safeCameraId =
      normalizeCameraId(
        cameraId
      );
  } catch {
    return null;
  }

  return {
    ...camera,

    cam_id:
      safeCameraId,

    latitude:
      Number(
        latitude.toFixed(6)
      ),

    longitude:
      Number(
        longitude.toFixed(6)
      ),
  };
};


/* ============================================================
 * GIS ROUTE POINT
 * ============================================================ */

const normalizeGISRoutePoint = (
  point
) => {
  if (
    !point ||
    typeof point !== "object"
  ) {
    return null;
  }

  const latitude =
    toFiniteNumber(
      point.latitude,
      null,
      -90,
      90
    );

  const longitude =
    toFiniteNumber(
      point.longitude,
      null,
      -180,
      180
    );

  if (
    latitude === null ||
    longitude === null
  ) {
    return null;
  }

  let cameraId;

  try {
    cameraId =
      normalizeCameraId(
        point.camera_id ??
          point.cam_id
      );
  } catch {
    return null;
  }

  return {
    ...point,

    camera_id:
      cameraId,

    latitude:
      Number(
        latitude.toFixed(6)
      ),

    longitude:
      Number(
        longitude.toFixed(6)
      ),

    timestamp:
      toFiniteNumber(
        point.timestamp,
        0,
        0
      ),

    similarity:
      toFiniteNumber(
        point.similarity,
        0,
        0,
        1
      ),

    correlation_score:
      toFiniteNumber(
        point.correlation_score,
        0,
        0,
        1
      ),

    supporting_frames:
      Math.max(
        0,
        Math.min(
          1000,
          Number.isFinite(
            Number(
              point.supporting_frames
            )
          )
            ? Math.trunc(
                Number(
                  point.supporting_frames
                )
              )
            : 0
        )
      ),

    consistency:
      toFiniteNumber(
        point.consistency,
        0,
        0,
        1
      ),

    strong_match:
      Boolean(
        point.strong_match
      ),

    match_type:
      typeof point.match_type ===
      "string"
        ? point.match_type.slice(
            0,
            64
          )
        : "unknown",
  };
};


/* ============================================================
 * GIS SEGMENT
 * ============================================================ */

const normalizeGISSegment = (
  segment
) => {
  if (
    !segment ||
    typeof segment !== "object"
  ) {
    return null;
  }

  let fromCameraId;
  let toCameraId;

  try {
    fromCameraId =
      normalizeCameraId(
        segment.from_camera_id
      );

    toCameraId =
      normalizeCameraId(
        segment.to_camera_id
      );
  } catch {
    return null;
  }


  const fromLatitude =
    toFiniteNumber(
      segment.from_latitude,
      null,
      -90,
      90
    );

  const fromLongitude =
    toFiniteNumber(
      segment.from_longitude,
      null,
      -180,
      180
    );

  const toLatitude =
    toFiniteNumber(
      segment.to_latitude,
      null,
      -90,
      90
    );

  const toLongitude =
    toFiniteNumber(
      segment.to_longitude,
      null,
      -180,
      180
    );


  if (
    fromLatitude === null ||
    fromLongitude === null ||
    toLatitude === null ||
    toLongitude === null
  ) {
    return null;
  }


  return {
    ...segment,

    from_camera_id:
      fromCameraId,

    to_camera_id:
      toCameraId,

    from_latitude:
      Number(
        fromLatitude.toFixed(6)
      ),

    from_longitude:
      Number(
        fromLongitude.toFixed(6)
      ),

    to_latitude:
      Number(
        toLatitude.toFixed(6)
      ),

    to_longitude:
      Number(
        toLongitude.toFixed(6)
      ),

    elapsed_seconds:
      toFiniteNumber(
        segment.elapsed_seconds,
        0,
        0
      ),

    distance_km:
      toFiniteNumber(
        segment.distance_km,
        null,
        0
      ),

    similarity:
      toFiniteNumber(
        segment.similarity,
        0,
        0,
        1
      ),

    correlation_score:
      toFiniteNumber(
        segment.correlation_score,
        0,
        0,
        1
      ),

    strong_match:
      Boolean(
        segment.strong_match
      ),

    match_type:
      typeof segment.match_type ===
      "string"
        ? segment.match_type.slice(
            0,
            64
          )
        : "unknown",
  };
};


/* ============================================================
 * VEHICLE PATH NORMALIZATION
 * ============================================================ */

const normalizeVehiclePath = (
  path
) => {
  if (
    !path ||
    typeof path !== "object"
  ) {
    return null;
  }

  let globalVehicleId;

  try {
    globalVehicleId =
      normalizeGlobalVehicleId(
        path.global_vehicle_id
      );
  } catch {
    return null;
  }


  const route =
    Array.isArray(
      path.route
    )
      ? path.route
          .map(
            normalizeGISRoutePoint
          )
          .filter(Boolean)
          .slice(
            0,
            1000
          )
      : [];


  const segments =
    Array.isArray(
      path.segments
    )
      ? path.segments
          .map(
            normalizeGISSegment
          )
          .filter(Boolean)
          .slice(
            0,
            1000
          )
      : [];


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

    last_seen:
      toFiniteNumber(
        path.last_seen,
        0,
        0
      ),

    strong_match:
      Boolean(
        path.strong_match
      ),

    route,

    segments,

    transition_count:
      Math.max(
        0,
        Math.min(
          1000,
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
            : segments.length
        )
      ),
  };
};


/* ============================================================
 * VEHICLE JOURNEY NORMALIZATION
 * ============================================================ */

const normalizeVehicleJourney = (
  journey
) => {
  if (
    !journey ||
    typeof journey !== "object"
  ) {
    return null;
  }

  let globalVehicleId;

  try {
    globalVehicleId =
      normalizeGlobalVehicleId(
        journey.global_vehicle_id
      );
  } catch {
    return null;
  }


  const route =
    Array.isArray(
      journey.route
    )
      ? journey.route
          .map(
            normalizeGISRoutePoint
          )
          .filter(Boolean)
          .slice(
            0,
            2000
          )
      : [];


  const segments =
    Array.isArray(
      journey.segments
    )
      ? journey.segments
          .map(
            normalizeGISSegment
          )
          .filter(Boolean)
          .slice(
            0,
            2000
          )
      : [];


  const cameraVisits =
    Array.isArray(
      journey.camera_visits
    )
      ? journey.camera_visits
          .map(
            (visit) => {
              if (
                !visit ||
                typeof visit !==
                  "object"
              ) {
                return null;
              }


              let cameraId;

              try {
                cameraId =
                  normalizeCameraId(
                    visit.camera_id
                  );
              } catch {
                return null;
              }


              const latitude =
                toFiniteNumber(
                  visit.latitude,
                  null,
                  -90,
                  90
                );

              const longitude =
                toFiniteNumber(
                  visit.longitude,
                  null,
                  -180,
                  180
                );


              return {
                ...visit,

                camera_id:
                  cameraId,

                latitude,

                longitude,

                first_seen:
                  toFiniteNumber(
                    visit.first_seen,
                    0,
                    0
                  ),

                last_seen:
                  toFiniteNumber(
                    visit.last_seen,
                    0,
                    0
                  ),

                observation_count:
                  Math.max(
                    0,
                    Math.min(
                      100000,
                      Number.isFinite(
                        Number(
                          visit.observation_count
                        )
                      )
                        ? Math.trunc(
                            Number(
                              visit.observation_count
                            )
                          )
                        : 0
                    )
                  ),

                track_ids:
                  Array.isArray(
                    visit.track_ids
                  )
                    ? visit.track_ids
                        .filter(
                          (id) =>
                            typeof id ===
                            "string"
                        )
                        .slice(
                          0,
                          100
                        )
                    : [],
              };
            }
          )
          .filter(Boolean)
          .slice(
            0,
            1000
          )
      : [];


  return {
    ...journey,

    global_vehicle_id:
      globalVehicleId,

    vehicle_type:
      typeof journey.vehicle_type ===
      "string"
        ? journey.vehicle_type.slice(
            0,
            64
          )
        : "vehicle",

    first_seen:
      toFiniteNumber(
        journey.first_seen,
        0,
        0
      ),

    last_seen:
      toFiniteNumber(
        journey.last_seen,
        0,
        0
      ),

    last_similarity:
      toFiniteNumber(
        journey.last_similarity,
        0,
        0,
        1
      ),

    last_correlation_score:
      toFiniteNumber(
        journey.last_correlation_score,
        0,
        0,
        1
      ),

    strong_match:
      Boolean(
        journey.strong_match
      ),

    camera_visits:
      cameraVisits,

    route,

    segments,

    transition_count:
      segments.length,

    total_distance_km:
      toFiniteNumber(
        journey.total_distance_km,
        0,
        0
      ),
  };
};


/* ============================================================
 * GIS API
 * ============================================================ */

/*
 * GET:
 * /api/gis/cameras
 *
 * Required by GisMap.jsx.
 */

export const getGISCameras =
  async (
    config = {}
  ) => {
    const response =
      await api.get(
        "/api/gis/cameras",
        config
      );

    const cameras =
      Array.isArray(
        response.data?.cameras
      )
        ? response.data.cameras
            .map(
              normalizeCameraGIS
            )
            .filter(Boolean)
        : [];


    return {
      ...response.data,

      count:
        cameras.length,

      cameras,
    };
  };


/*
 * GET:
 * /api/gis/vehicle-paths
 *
 * Required by GisMap.jsx.
 */

export const getGISVehiclePaths =
  async (
    params = {},
    config = {}
  ) => {
    const safeLimit =
      Math.max(
        1,
        Math.min(
          500,
          Number.isFinite(
            Number(
              params.limit
            )
          )
            ? Math.trunc(
                Number(
                  params.limit
                )
              )
            : 100
        )
      );


    const response =
      await api.get(
        "/api/gis/vehicle-paths",
        {
          ...config,

          params: {
            ...params,

            limit:
              safeLimit,
          },
        }
      );


    const paths =
      Array.isArray(
        response.data?.paths
      )
        ? response.data.paths
            .map(
              normalizeVehiclePath
            )
            .filter(Boolean)
        : [];


    return {
      ...response.data,

      count:
        paths.length,

      paths,
    };
  };


/*
 * GET:
 * /api/vehicles/{globalVehicleId}/journey
 *
 * Required by GisMap.jsx.
 */

export const getVehicleJourney =
  async (
    globalVehicleId,
    config = {}
  ) => {
    const safeId =
      normalizeGlobalVehicleId(
        globalVehicleId
      );


    const response =
      await api.get(
        `/api/vehicles/${encodeURIComponent(
          safeId
        )}/journey`,
        config
      );


    return normalizeVehicleJourney(
      response.data
    );
  };


/*
 * GET:
 * /api/vehicles/correlated
 */

export const getCorrelatedVehicles =
  async (
    params = {},
    config = {}
  ) => {
    const response =
      await api.get(
        "/api/vehicles/correlated",
        {
          ...config,
          params,
        }
      );


    const vehicles =
      Array.isArray(
        response.data?.vehicles
      )
        ? response.data.vehicles
            .map(
              (vehicle) => {
                if (
                  !vehicle ||
                  typeof vehicle !==
                    "object"
                ) {
                  return null;
                }


                let globalVehicleId;

                try {
                  globalVehicleId =
                    normalizeGlobalVehicleId(
                      vehicle.global_vehicle_id
                    );
                } catch {
                  return null;
                }


                return {
                  ...vehicle,

                  global_vehicle_id:
                    globalVehicleId,

                  vehicle_type:
                    typeof vehicle.vehicle_type ===
                    "string"
                      ? vehicle.vehicle_type.slice(
                          0,
                          64
                        )
                      : "vehicle",

                  last_similarity:
                    toFiniteNumber(
                      vehicle.last_similarity,
                      0,
                      0,
                      1
                    ),

                  last_correlation_score:
                    toFiniteNumber(
                      vehicle.last_correlation_score,
                      0,
                      0,
                      1
                    ),

                  strong_match:
                    Boolean(
                      vehicle.strong_match
                    ),

                  observations:
                    Array.isArray(
                      vehicle.observations
                    )
                      ? vehicle.observations
                          .slice(
                            0,
                            1000
                          )
                          .map(
                            normalizeGISRoutePoint
                          )
                          .filter(
                            Boolean
                          )
                      : [],
                };
              }
            )
            .filter(Boolean)
        : [];


    return {
      ...response.data,

      count:
        vehicles.length,

      vehicles,
    };
  };

export const getCorrelatedPersons = async (params = {}, config = {}) => {
  const response = await api.get("/api/persons/correlated", { ...config, params });
  return response.data;
};

export const createInvestigationSummary = async (subjectType, subjectId, incidentId = null, config = {}) => {
  const response = await api.post("/api/persons/investigation-summaries", {
    subject_type: String(subjectType || "").toUpperCase(),
    subject_id: String(subjectId || "").trim().toUpperCase(),
    incident_id: incidentId,
  }, config);
  return response.data;
};

export const getInvestigationSummary = async (summaryId, config = {}) => {
  const id = Number(summaryId);
  if (!Number.isInteger(id) || id < 1) throw new Error("Invalid summary ID");
  const response = await api.get(`/api/persons/investigation-summaries/${id}`, config);
  return response.data;
};


/* ============================================================
 * CAMERA API
 * ============================================================ */

export const getCameras =
  async (
    config = {}
  ) => {
    const response =
      await api.get(
        "/cameras",
        config
      );

    return response.data;
  };


export const getCamera =
  async (
    camId,
    config = {}
  ) => {
    const safeId =
      normalizeCameraId(
        camId
      );


    const response =
      await api.get(
        `/camera/${encodeURIComponent(
          safeId
        )}`,
        config
      );


    return response.data;
  };


/* ============================================================
 * GENERAL WATCHLIST API
 * ============================================================ */

export const getWatchlists =
  async (
    config = {}
  ) => {
    const response =
      await api.get(
        "/api/watchlist",
        config
      );

    return response.data;
  };


export const createWatchlist =
  async (
    payload,
    config = {}
  ) => {
    const response =
      await api.post(
        "/api/watchlist",
        payload,
        config
      );

    return response.data;
  };


export const updateWatchlist =
  async (
    watchlistId,
    payload,
    config = {}
  ) => {
    const response =
      await api.patch(
        `/api/watchlist/${encodeURIComponent(
          watchlistId
        )}`,
        payload,
        config
      );

    return response.data;
  };


export const deleteWatchlist =
  async (
    watchlistId,
    config = {}
  ) => {
    const response =
      await api.delete(
        `/api/watchlist/${encodeURIComponent(
          watchlistId
        )}`,
        config
      );

    return response.data;
  };


export const getWatchlistEntries =
  async (
    watchlistId,
    params = {},
    config = {}
  ) => {
    const response =
      await api.get(
        `/api/watchlist/${encodeURIComponent(
          watchlistId
        )}/entries`,
        {
          ...config,
          params,
        }
      );

    return response.data;
  };


export const createWatchlistEntry =
  async (
    watchlistId,
    payload,
    config = {}
  ) => {
    const response =
      await api.post(
        `/api/watchlist/${encodeURIComponent(
          watchlistId
        )}/entries`,
        payload,
        config
      );

    return response.data;
  };


export const updateWatchlistEntry =
  async (
    entryId,
    payload,
    config = {}
  ) => {
    const response =
      await api.patch(
        `/api/watchlist/entries/${encodeURIComponent(
          entryId
        )}`,
        payload,
        config
      );

    return response.data;
  };


export const deleteWatchlistEntry =
  async (
    entryId,
    config = {}
  ) => {
    const response =
      await api.delete(
        `/api/watchlist/entries/${encodeURIComponent(
          entryId
        )}`,
        config
      );

    return response.data;
  };


export const matchWatchlistPlate =
  async (
    plate,
    includePossible = true,
    config = {}
  ) => {
    const response =
      await api.post(
        "/api/watchlist/match",
        {
          plate,
          include_possible:
            includePossible,
        },
        config
      );

    return response.data;
  };


/* ============================================================
 * PERSON WATCHLIST API
 * ============================================================ */

export const getPersonWatchlist =
  async (
    params = {},
    config = {}
  ) => {
    const response =
      await api.get(
        "/api/intelligence/person-watchlist",
        {
          ...config,
          params,
        }
      );

    return response.data;
  };


export const enrollPersonWatchlist =
  async (
    payload,
    config = {}
  ) => {
    if (
      !payload ||
      typeof payload !== "object"
    ) {
      throw new Error(
        "Person enrollment payload is required."
      );
    }


    const fullName =
      String(
        payload.full_name ??
          ""
      ).trim();


    if (!fullName) {
      throw new Error(
        "Person full name is required."
      );
    }


    const image =
      payload.image;


    if (!image) {
      throw new Error(
        "Reference image is required."
      );
    }


    if (
      typeof File !==
        "undefined" &&
      !(image instanceof File)
    ) {
      throw new Error(
        "Invalid reference image."
      );
    }


    const allowedImageTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ];


    if (
      image.type &&
      !allowedImageTypes.includes(
        image.type
      )
    ) {
      throw new Error(
        "Only JPG, PNG and WEBP images are allowed."
      );
    }


    const maxImageSize =
      10 * 1024 * 1024;


    if (
      Number.isFinite(
        image.size
      ) &&
      image.size >
        maxImageSize
    ) {
      throw new Error(
        "Reference image must be smaller than 10 MB."
      );
    }


    const formData =
      new FormData();


    formData.append(
      "full_name",
      fullName
    );


    formData.append(
      "category",
      String(
        payload.category ||
          "OTHER"
      )
        .trim()
        .toUpperCase()
    );


    formData.append(
      "status",
      String(
        payload.status ||
          "ACTIVE"
      )
        .trim()
        .toUpperCase()
    );


    formData.append(
      "description",
      String(
        payload.description ||
          ""
      ).trim()
    );


    formData.append(
      "source",
      String(
        payload.source ||
          ""
      ).trim()
    );


    formData.append(
      "external_reference",
      String(
        payload.external_reference ||
          ""
      ).trim()
    );


    formData.append(
      "image",
      image,
      image.name ||
        "reference.jpg"
    );


    const headers = {
      ...(config.headers || {}),
    };


    delete headers[
      "Content-Type"
    ];

    delete headers[
      "content-type"
    ];


    const response =
      await api.post(
        "/api/intelligence/person-watchlist/enroll",
        formData,
        {
          ...config,
          headers,
        }
      );


    return response.data;
  };


export const getPersonWatchlistImage =
  async (
    id,
    config = {}
  ) => {
    if (
      id === null ||
      id === undefined ||
      id === ""
    ) {
      throw new Error(
        "Person watchlist ID is required."
      );
    }


    const response =
      await api.get(
        `/api/intelligence/person-watchlist/${encodeURIComponent(
          id
        )}/image`,
        {
          ...config,
          responseType:
            "blob",
        }
      );


    return response.data;
  };


export const uploadPersonWatchlistImage =
  async (
    id,
    image,
    config = {}
  ) => {
    if (
      id === null ||
      id === undefined ||
      id === ""
    ) {
      throw new Error(
        "Person watchlist ID is required."
      );
    }


    if (
      typeof File !==
        "undefined" &&
      !(image instanceof File)
    ) {
      throw new Error(
        "Reference image is required."
      );
    }


    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/webp",
    ];


    if (
      image.type &&
      !allowedTypes.includes(
        image.type
      )
    ) {
      throw new Error(
        "Only JPG, PNG and WEBP images are allowed."
      );
    }


    if (
      Number.isFinite(
        image.size
      ) &&
      image.size >
        10 * 1024 * 1024
    ) {
      throw new Error(
        "Reference image must be smaller than 10 MB."
      );
    }


    const formData =
      new FormData();


    formData.append(
      "image",
      image,
      image.name ||
        "reference.jpg"
    );


    const headers = {
      ...(config.headers || {}),
    };


    delete headers[
      "Content-Type"
    ];

    delete headers[
      "content-type"
    ];


    const response =
      await api.post(
        `/api/intelligence/person-watchlist/${encodeURIComponent(
          id
        )}/image`,
        formData,
        {
          ...config,
          headers,
        }
      );


    return response.data;
  };


export const updatePersonWatchlist =
  async (
    id,
    payload,
    config = {}
  ) => {
    if (
      id === null ||
      id === undefined ||
      id === ""
    ) {
      throw new Error(
        "Person watchlist ID is required."
      );
    }


    const formData =
      new FormData();


    if (
      payload?.full_name !==
      undefined
    ) {
      formData.append(
        "full_name",
        String(
          payload.full_name
        ).trim()
      );
    }


    if (
      payload?.category !==
      undefined
    ) {
      formData.append(
        "category",
        String(
          payload.category
        )
          .trim()
          .toUpperCase()
      );
    }


    if (
      payload?.status !==
      undefined
    ) {
      formData.append(
        "status",
        String(
          payload.status
        )
          .trim()
          .toUpperCase()
      );
    }


    if (
      payload?.description !==
      undefined
    ) {
      formData.append(
        "description",
        String(
          payload.description
        )
      );
    }


    if (
      payload?.source !==
      undefined
    ) {
      formData.append(
        "source",
        String(
          payload.source
        )
      );
    }


    if (
      payload?.external_reference !==
      undefined
    ) {
      formData.append(
        "external_reference",
        String(
          payload.external_reference
        )
      );
    }


    const headers = {
      ...(config.headers || {}),
    };


    delete headers[
      "Content-Type"
    ];

    delete headers[
      "content-type"
    ];


    const response =
      await api.patch(
        `/api/intelligence/person-watchlist/${encodeURIComponent(
          id
        )}`,
        formData,
        {
          ...config,
          headers,
        }
      );


    return response.data;
  };


export const deletePersonWatchlist =
  async (
    id,
    config = {}
  ) => {
    if (
      id === null ||
      id === undefined ||
      id === ""
    ) {
      throw new Error(
        "Person watchlist ID is required."
      );
    }


    const response =
      await api.delete(
        `/api/intelligence/person-watchlist/${encodeURIComponent(
          id
        )}`,
        config
      );


    return response.data;
  };


/* ============================================================
 * GIS CAMERA / STREAM SESSION
 * ============================================================ */

export const createCameraStreamSession =
  async (
    cameraId
  ) => {
    const safeId =
      normalizeCameraId(
        cameraId
      );


    const response =
      await api.post(
        `/stream-session/camera/${encodeURIComponent(
          safeId
        )}`
      );


    return response.data;
  };


export const getGISCameraCalibration =
  async (
    cameraId
  ) => {
    const safeId =
      normalizeCameraId(
        cameraId
      );


    const response =
      await api.get(
        `/api/gis/cameras/${encodeURIComponent(
          safeId
        )}/calibration`
      );


    return response.data;
  };


export const setGISCameraCalibration =
  async (
    cameraId,
    imagePoints,
    geoPoints
  ) => {
    const safeId =
      normalizeCameraId(
        cameraId
      );


    if (
      !Array.isArray(
        imagePoints
      ) ||
      !Array.isArray(
        geoPoints
      ) ||
      imagePoints.length !==
        geoPoints.length ||
      imagePoints.length < 4
    ) {
      throw new Error(
        "At least four corresponding image and GIS points are required."
      );
    }


    const response =
      await api.post(
        `/api/gis/cameras/${encodeURIComponent(
          safeId
        )}/calibration`,
        {
          image_points:
            imagePoints,

          geo_points:
            geoPoints,
        }
      );


    return response.data;
  };


/* ============================================================
 * INCIDENT API
 * ============================================================ */

export const getIncidents =
  async (
    params = {},
    config = {}
  ) => {
    const response =
      await api.get(
        "/api/intelligence/incidents",
        {
          ...config,
          params,
        }
      );


    return response.data;
  };


export const createIncident =
  async (
    payload,
    config = {}
  ) => {
    const response =
      await api.post(
        "/api/intelligence/incidents",
        payload,
        config
      );


    return response.data;
  };


export const updateIncident =
  async (
    id,
    payload,
    config = {}
  ) => {
    if (
      id === null ||
      id === undefined ||
      id === ""
    ) {
      throw new Error(
        "Incident ID is required."
      );
    }


    const response =
      await api.patch(
        `/api/intelligence/incidents/${encodeURIComponent(
          id
        )}`,
        payload,
        config
      );


    return response.data;
  };


export const addIncidentEvidence =
  async (
    id,
    payload,
    config = {}
  ) => {
    if (
      id === null ||
      id === undefined ||
      id === ""
    ) {
      throw new Error(
        "Incident ID is required."
      );
    }


    const response =
      await api.post(
        `/api/intelligence/incidents/${encodeURIComponent(
          id
        )}/evidence`,
        payload,
        config
      );


    return response.data;
  };




export const getIncidentEvidenceReportPreview =
  async (
    id,
    classification = "RESTRICTED",
    config = {}
  ) => {
    if (id === null || id === undefined || id === "") {
      throw new Error("Incident ID is required.");
    }

    const response = await api.get(
      `/api/reports/incidents/${encodeURIComponent(id)}/evidence/preview`,
      {
        ...config,
        params: {
          ...(config.params || {}),
          classification,
        },
      }
    );

    return response.data;
  };


export const downloadIncidentEvidenceReport =
  async (
    id,
    classification = "RESTRICTED",
    config = {}
  ) => {
    if (id === null || id === undefined || id === "") {
      throw new Error("Incident ID is required.");
    }

    return api.get(
      `/api/reports/incidents/${encodeURIComponent(id)}/evidence/export`,
      {
        ...config,
        params: {
          ...(config.params || {}),
          classification,
        },
        responseType: "blob",
        timeout: config.timeout ?? 120000,
      }
    );
  };


/* ============================================================
 * DEFAULT EXPORT
 * ============================================================ */

export default api;



