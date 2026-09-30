import React, {
  useCallback,
  useEffect,
  useRef,
  useState,
} from "react";

import toast from "react-hot-toast";

import {
  CameraOff,
  MapPin,
  Navigation,
  Loader2,
  AlertCircle,
  RefreshCw,
} from "lucide-react";

import Loading from "./Loading";
import api, { createCameraStreamUrl } from "../api/axios";
import { formatIndianDateTime } from "../utils/dateTime";

// ============================================================
// CONSTANTS
// ============================================================

const MAX_STREAM_RETRIES = 3;
const STREAM_RETRY_DELAY_MS = 2000;

// ============================================================
// CAMERA LIST
// ============================================================

const CameraList = ({
  onCameraMapFocus = null,
  selectedCamera = null,
}) => {

  // ==========================================================
  // STATE
  // ==========================================================

  const [cameras, setCameras] = useState([]);

  const [loading, setLoading] = useState(true);

  /*
   * Stream state:
   *
   * idle
   * starting
   * connecting
   * online
   * error
   * stopping
   */
  const [streamStates, setStreamStates] =
    useState({});

  /*
   * Secure stream URLs.
   */
  const [streamUrls, setStreamUrls] =
    useState({});

  /*
   * Human-readable stream errors.
   */
  const [streamErrors, setStreamErrors] =
    useState({});

  const [deletingId, setDeletingId] =
    useState(null);

  const [startingId, setStartingId] =
    useState(null);

  const [stoppingId, setStoppingId] =
    useState(null);

  const [systemCapacity, setSystemCapacity] =
    useState(null);

  const [retryingId, setRetryingId] =
    useState(null);


  // ==========================================================
  // REFS
  // ==========================================================

  const mountedRef =
    useRef(false);

  const cameraCardRefs =
    useRef({});

  const retryTimersRef =
    useRef({});

  const streamRetryCountsRef =
    useRef({});

  const startRequestsRef =
    useRef({});

  const streamSessionRequestsRef =
    useRef({});


  // ==========================================================
  // SAFE ERROR MESSAGE
  // ==========================================================

  const getErrorMessage = useCallback(
    (error, fallback) => {

      const detail =
        error?.response?.data?.detail;

      if (
        typeof detail === "string" &&
        detail.trim()
      ) {
        return detail.trim();
      }

      if (
        detail &&
        typeof detail === "object" &&
        typeof detail.message === "string" &&
        detail.message.trim()
      ) {
        return detail.message.trim();
      }

      const message =
        error?.response?.data?.message;

      if (
        typeof message === "string" &&
        message.trim()
      ) {
        return message.trim();
      }

      if (
        typeof error?.message === "string" &&
        error.message.trim()
      ) {
        return error.message.trim();
      }

      return fallback;
    },
    []
  );


  // ==========================================================
  // CAMERA ID
  // ==========================================================

  const getCameraKey = useCallback(
    (camera) => {

      const camId = String(
        camera?.cam_id ||
        camera?.camera_id ||
        camera?.id ||
        ""
      ).trim();

      return camId;
    },
    []
  );

  /*
   * Normalize the backend camera schema once at the UI boundary.
   *
   * The backend may return:
   *   camera_id / name
   * while older frontend code expects:
   *   cam_id / camera_name
   *
   * Keeping both aliases here prevents rendering, start/stop, GIS,
   * and stream logic from breaking when the API uses the canonical
   * camera_id/name fields.
   */
  const normalizeCamera = useCallback(
    (camera) => {
      if (!camera || typeof camera !== "object") {
        return null;
      }

      const camId = String(
        camera?.cam_id ||
        camera?.camera_id ||
        camera?.id ||
        ""
      ).trim();

      if (!camId) {
        return null;
      }

      const cameraName = String(
        camera?.camera_name ||
        camera?.name ||
        camId
      ).trim();

      return {
        ...camera,
        cam_id: camId,
        camera_id: camera?.camera_id || camId,
        camera_name: cameraName,
        name: camera?.name || cameraName,
        latitude:
          camera?.latitude ??
          camera?.lat ??
          null,
        longitude:
          camera?.longitude ??
          camera?.lng ??
          camera?.lon ??
          null,
      };
    },
    []
  );


  // ==========================================================
  // VALIDATE GIS COORDINATES
  // ==========================================================

  const hasValidCoordinates =
    useCallback((camera) => {

      const latitude =
        Number(camera?.latitude);

      const longitude =
        Number(camera?.longitude);

      return (
        Number.isFinite(latitude) &&
        Number.isFinite(longitude) &&
        latitude >= -90 &&
        latitude <= 90 &&
        longitude >= -180 &&
        longitude <= 180
      );
    }, []);


  // ==========================================================
  // GET LOCATION NAME
  // ==========================================================
  /*
   * Camera setup stores:
   *
   * location_name
   *
   * Example:
   *
   * "Main Gate, SG Highway"
   *
   * Keep this separate from latitude/longitude.
   */

  const getLocationName = useCallback(
    (camera) => {

      // ========================================================
      // 1. PRIMARY: Explicit human-readable camera location
      // ========================================================

      const locationName =
        camera?.location_name;

      if (
        typeof locationName === "string" &&
        locationName.trim()
      ) {
        return locationName.trim();
      }

      // ========================================================
      // 2. FALLBACK: Road / highway + city
      // ========================================================

      const roadName =
        camera?.road_name;

      if (
        typeof roadName === "string" &&
        roadName.trim()
      ) {
        const road =
          roadName.trim();

        const city =
          typeof camera?.city === "string"
            ? camera.city.trim()
            : "";

        if (city) {
          return `${road}, ${city}`;
        }

        return road;
      }

      // ========================================================
      // 3. FALLBACK: City + State + Country
      // ========================================================

      const locationParts = [
        camera?.city,
        camera?.state,
        camera?.country,
      ]
        .filter(
          (value) =>
            typeof value === "string" &&
            value.trim()
        )
        .map(
          (value) => value.trim()
        );

      if (locationParts.length > 0) {
        return locationParts.join(", ");
      }

      // ========================================================
      // 4. No human-readable location available
      // ========================================================

      return "";
    },
    []
  );


  // ==========================================================
  // GET CAMERA DISPLAY NAME
  // ==========================================================

  const getCameraName = useCallback(
    (camera) => {
      const value =
        camera?.camera_name ||
        camera?.name;

      if (
        typeof value === "string" &&
        value.trim()
      ) {
        return value.trim();
      }

      return getCameraKey(camera) || "Camera";
    },
    [getCameraKey]
  );


  // ==========================================================
  // ==========================================================
  // CREATE BACKEND STREAM URL
  // ==========================================================

  const createStreamURL =
    useCallback(
      (camId) => {

        const safeCamId =
          String(camId || "").trim();

        if (!safeCamId) {
          return "";
        }

        /*
         * Use the shared API URL builder instead of reading
         * environment-specific backend variables directly here.
         *
         * Local Vite development:
         *   /stream/{cameraId} -> Vite proxy -> FastAPI :3000
         *
         * Production:
         *   createCameraStreamUrl() uses the configured API origin.
         *
         * This keeps the camera preview consistent with the same
         * backend routing strategy used by the rest of INTEL-I.
         */

        try {
          return createCameraStreamUrl(
            safeCamId,
            Date.now()
          );
        } catch (error) {
          console.error(
            "[INTEL-I][CAMERA] Failed to create stream URL:",
            error
          );

          return "";
        }
      },
      []
    );


  // SET STREAM STATE
  // ==========================================================

  const setCameraStreamState =
    useCallback(
      (camId, state) => {

        if (!mountedRef.current) {
          return;
        }

        setStreamStates(
          (previous) => ({
            ...previous,
            [camId]: state,
          })
        );
      },
      []
    );


  // ==========================================================
  // CLEAR STREAM
  // ==========================================================

  const clearStream =
    useCallback((camId) => {

      const safeCamId =
        String(camId || "").trim();

      if (!safeCamId) {
        return;
      }

      const timer =
        retryTimersRef.current[
          safeCamId
        ];

      if (timer) {

        clearTimeout(timer);

        delete retryTimersRef.current[
          safeCamId
        ];
      }

      delete streamRetryCountsRef.current[
        safeCamId
      ];

      if (!mountedRef.current) {
        return;
      }

      setStreamUrls(
        (previous) => {

          const next = {
            ...previous,
          };

          delete next[safeCamId];

          return next;
        }
      );

      setStreamStates(
        (previous) => ({
          ...previous,
          [safeCamId]: "idle",
        })
      );

      setStreamErrors(
        (previous) => {

          const next = {
            ...previous,
          };

          delete next[safeCamId];

          return next;
        }
      );

      setRetryingId(
        (current) =>
          current === safeCamId
            ? null
            : current
      );

    }, []);


  // ==========================================================
  // FETCH SYSTEM CAPACITY
  // ==========================================================

  const fetchSystemCapacity =
    useCallback(async () => {

      try {

        const response =
          await api.get(
            "/system/capacity"
          );

        if (
          !mountedRef.current
        ) {
          return;
        }

        setSystemCapacity(
          response?.data || null
        );

      } catch (error) {

        /*
         * Capacity is informational.
         * Don't break the camera UI if it fails.
         */

        console.error(
          "Failed to fetch system capacity:",
          error
        );
      }

    }, []);


  // ==========================================================
  // FETCH CAMERAS
  // ==========================================================

  const fetchCameras =
    useCallback(async () => {

      try {

        setLoading(true);

        const response =
          await api.get(
            "/cameras"
          );

        console.log(
          "GET /cameras RESPONSE:",
          response?.data
        );

        const rawCameraList =
          Array.isArray(response?.data)
            ? response.data
            : Array.isArray(
                response?.data?.cameras
              )
              ? response.data.cameras
              : [];

        const cameraList = rawCameraList
          .map(normalizeCamera)
          .filter(Boolean);

        if (
          !mountedRef.current
        ) {
          return;
        }

        setCameras(
          cameraList
        );

        const initialStates = {};

        cameraList.forEach(
          (camera) => {

            const camId =
              getCameraKey(
                camera
              );

            if (!camId) {
              return;
            }

            /*
             * Backend is_active indicates
             * backend state, but this browser
             * does not automatically have a
             * live stream.
             */

            const backendState = String(
              camera?.connection_state || camera?.status || "OFFLINE"
            ).trim().toUpperCase();
            initialStates[camId] = backendState;
          }
        );

        setStreamStates(
          initialStates
        );

      } catch (error) {

        console.error(
          "Failed to fetch cameras:",
          error
        );

        if (
          mountedRef.current
        ) {

          setCameras([]);

          toast.error(
            getErrorMessage(
              error,
              "Failed to load cameras"
            )
          );
        }

      } finally {

        if (
          mountedRef.current
        ) {
          setLoading(false);
        }
      }

    }, [
      getCameraKey,
      getErrorMessage,
      normalizeCamera,
    ]);


  // ==========================================================
  // REFRESH AUTHORITATIVE CAMERA HEALTH
  // ==========================================================

  const refreshCameraHealth = useCallback(async () => {
    try {
      const response = await api.get("/cameras");
      const rawLatest =
        Array.isArray(response?.data)
          ? response.data
          : Array.isArray(response?.data?.cameras)
            ? response.data.cameras
            : [];

      const latest = rawLatest
        .map(normalizeCamera)
        .filter(Boolean);

      if (!mountedRef.current) return;

      setCameras((previous) => {
        const previousById = new Map(
          previous.map((camera) => [getCameraKey(camera), camera])
        );
        return latest.map((camera) => {
          const id = getCameraKey(camera);
          return { ...(previousById.get(id) || {}), ...camera };
        });
      });
    } catch (error) {
      // Health refresh is non-blocking; retain the last known camera list.
      console.debug("Camera health refresh failed", error);
    }
  }, [getCameraKey, normalizeCamera]);


  // ==========================================================
  // CREATE SECURE STREAM SESSION
  // ==========================================================

  const createStreamSession =
    useCallback(
      async (camId) => {

        const safeCamId =
          String(camId || "").trim();

        if (!safeCamId) {
          throw new Error(
            "Invalid camera ID"
          );
        }

        /*
         * Prevent duplicate session requests.
         */

        if (
          streamSessionRequestsRef
            .current[safeCamId]
        ) {
          return (
            streamSessionRequestsRef
              .current[safeCamId]
          );
        }

        const request =
          (async () => {

            try {

              /*
               * Axios instance already has:
               *
               * withCredentials: true
               *
               * and CSRF handling.
               */

              await api.post(
                `/stream-session/camera/${encodeURIComponent(
                  safeCamId
                )}`,
                {}
              );

              if (
                !mountedRef.current
              ) {
                return "";
              }

              const streamURL =
                createStreamURL(
                  safeCamId
                );

              if (!streamURL) {
                throw new Error(
                  "Failed to create stream URL"
                );
              }

              setStreamUrls(
                (previous) => ({
                  ...previous,
                  [safeCamId]:
                    streamURL,
                })
              );

              setStreamErrors(
                (previous) => {

                  const next = {
                    ...previous,
                  };

                  delete next[
                    safeCamId
                  ];

                  return next;
                }
              );

              /*
               * The session exists, but the
               * actual camera is not ONLINE yet.
               */

              setCameraStreamState(
                safeCamId,
                "connecting"
              );

              return streamURL;

            } finally {

              delete streamSessionRequestsRef
                .current[safeCamId];
            }

          })();

        streamSessionRequestsRef.current[
          safeCamId
        ] = request;

        return request;
      },
      [
        createStreamURL,
        setCameraStreamState,
      ]
    );


  // ==========================================================
  // START CAMERA
  // ==========================================================

  const startCamera =
    useCallback(
      async (camera) => {

        const camId =
          getCameraKey(camera);

        if (!camId) {

          toast.error(
            "Invalid camera ID"
          );

          return;
        }

        /*
         * Prevent double-clicks and
         * duplicate start requests.
         */

        if (
          startRequestsRef.current[
            camId
          ]
        ) {
          return;
        }

        startRequestsRef.current[
          camId
        ] = true;

        try {

          setStartingId(camId);

          setCameraStreamState(
            camId,
            "starting"
          );

          setStreamErrors(
            (previous) => {

              const next = {
                ...previous,
              };

              delete next[
                camId
              ];

              return next;
            }
          );

          /*
           * Clear previous stream.
           */

          clearStream(camId);

          setCameraStreamState(
            camId,
            "starting"
          );


          // --------------------------------------------------
          // STEP 1: START BACKEND CAMERA WORKER
          // --------------------------------------------------

          try {

            await api.post(
              `/camera/${encodeURIComponent(
                camId
              )}/start`,
              {}
            );

          } catch (error) {

            const status =
              error?.response
                ?.status;

            let message =
              getErrorMessage(
                error,
                "Failed to start camera"
              );

            if (
              status === 401 ||
              status === 403
            ) {

              message =
                "Your session has expired or you are not authorized.";

            } else if (
              status === 404
            ) {

              message =
                "Camera was not found.";

            } else if (
              status === 409
            ) {

              message =
                "Camera is already running.";

            } else if (
              status === 429
            ) {

              message =
                "System camera capacity has been reached.";
            }

            throw new Error(
              message
            );
          }

          if (
            !mountedRef.current
          ) {
            return;
          }


          // --------------------------------------------------
          // STEP 2: CREATE SECURE STREAM SESSION
          // --------------------------------------------------

          try {

            await createStreamSession(
              camId
            );

          } catch (error) {

            console.error(
              "Failed to create stream session:",
              error
            );

            throw new Error(
              getErrorMessage(
                error,
                "Camera started, but secure stream session could not be created."
              )
            );
          }

          if (
            !mountedRef.current
          ) {
            return;
          }

          setCameraStreamState(
            camId,
            "connecting"
          );

          toast.success(
            `${getCameraName(camera)
            } is connecting...`
          );

          await fetchSystemCapacity();

        } catch (error) {

          console.error(
            "Camera startup failed:",
            error
          );

          if (
            mountedRef.current
          ) {

            const message =
              error instanceof Error
                ? error.message
                : getErrorMessage(
                    error,
                    "Camera failed to start"
                  );

            setCameraStreamState(
              camId,
              "error"
            );

            setStreamErrors(
              (previous) => ({
                ...previous,
                [camId]:
                  message,
              })
            );

            toast.error(
              message
            );

            clearStream(camId);

            setCameraStreamState(
              camId,
              "error"
            );

            setStreamErrors(
              (previous) => ({
                ...previous,
                [camId]:
                  message,
              })
            );
          }

        } finally {

          delete startRequestsRef.current[
            camId
          ];

          if (
            mountedRef.current
          ) {
            setStartingId(null);
          }
        }

      },
      [
        clearStream,
        createStreamSession,
        fetchSystemCapacity,
        getCameraKey,
        getCameraName,
        getErrorMessage,
        setCameraStreamState,
      ]
    );


  // ==========================================================
  // STOP CAMERA
  // ==========================================================

  const stopCamera =
    useCallback(
      async (camera) => {

        const camId =
          getCameraKey(camera);

        if (!camId) {
          return;
        }

        if (
          stoppingId === camId
        ) {
          return;
        }

        try {

          setStoppingId(camId);

          await api.post(
            `/camera/${encodeURIComponent(
              camId
            )}/stop`,
            {}
          );

          if (
            !mountedRef.current
          ) {
            return;
          }

          clearStream(camId);

          setCameraStreamState(
            camId,
            "idle"
          );

          toast.success(
            `${getCameraName(camera)
            } stopped`
          );

          await fetchSystemCapacity();

        } catch (error) {

          console.error(
            "Failed to stop camera:",
            error
          );

          if (
            mountedRef.current
          ) {

            toast.error(
              getErrorMessage(
                error,
                "Failed to stop camera"
              )
            );
          }

        } finally {

          if (
            mountedRef.current
          ) {
            setStoppingId(null);
          }
        }

      },
      [
        clearStream,
        fetchSystemCapacity,
        getCameraKey,
        getCameraName,
        getErrorMessage,
        setCameraStreamState,
        stoppingId,
      ]
    );


  // ==========================================================
  // DELETE CAMERA
  // ==========================================================

  const deleteCamera =
    useCallback(
      async (camId) => {

        const safeCamId =
          String(camId || "").trim();

        if (!safeCamId) {
          return;
        }

        const currentState =
          streamStates[
            safeCamId
          ];

        if (
          currentState ===
          "online"
        ) {

          toast.error(
            "Stop the camera before deleting it."
          );

          return;
        }

        if (
          deletingId === safeCamId
        ) {
          return;
        }

        try {

          setDeletingId(
            safeCamId
          );

          await api.delete(
            `/cameras/${encodeURIComponent(
              safeCamId
            )}`
          );

          if (
            !mountedRef.current
          ) {
            return;
          }

          clearStream(
            safeCamId
          );

          setCameras(
            (previous) =>
              previous.filter(
                (camera) =>
                  getCameraKey(
                    camera
                  ) !== safeCamId
              )
          );

          setStreamStates(
            (previous) => {

              const next = {
                ...previous,
              };

              delete next[
                safeCamId
              ];

              return next;
            }
          );

          toast.success(
            "Camera deleted"
          );

          await fetchSystemCapacity();

        } catch (error) {

          console.error(
            "Failed to delete camera:",
            error
          );

          if (
            mountedRef.current
          ) {

            toast.error(
              getErrorMessage(
                error,
                "Failed to delete camera"
              )
            );
          }

        } finally {

          if (
            mountedRef.current
          ) {
            setDeletingId(
              null
            );
          }
        }

      },
      [
        clearStream,
        deletingId,
        fetchSystemCapacity,
        getCameraKey,
        getErrorMessage,
        streamStates,
      ]
    );


  // ==========================================================
  // STREAM ERROR
  // ==========================================================

  const handleStreamError =
    useCallback(
      (camera) => {

        const camId =
          getCameraKey(camera);

        if (!camId) {
          return;
        }

        const retryCount =
          Number(
            streamRetryCountsRef
              .current[camId] || 0
          );

        /*
         * After several failures, don't
         * endlessly hammer the backend.
         */

        if (
          retryCount >=
          MAX_STREAM_RETRIES
        ) {

          if (
            mountedRef.current
          ) {

            setCameraStreamState(
              camId,
              "error"
            );

            setStreamErrors(
              (previous) => ({
                ...previous,
                [camId]:
                  "Live stream could not be opened. Check the camera source and backend worker.",
              })
            );

            setRetryingId(
              (current) =>
                current === camId
                  ? null
                  : current
            );
          }

          return;
        }

        const nextRetry =
          retryCount + 1;

        streamRetryCountsRef.current[
          camId
        ] = nextRetry;

        if (
          mountedRef.current
        ) {

          setCameraStreamState(
            camId,
            "connecting"
          );

          setRetryingId(
            camId
          );
        }

        const timer =
          retryTimersRef.current[
            camId
          ];

        if (timer) {
          clearTimeout(timer);
        }

        retryTimersRef.current[
          camId
        ] = setTimeout(
          async () => {

            try {

              /*
               * Recreate the secure session
               * before retrying the stream.
               */

              await createStreamSession(
                camId
              );

            } catch (error) {

              console.error(
                "Stream reconnect failed:",
                error
              );

              if (
                mountedRef.current
              ) {

                setCameraStreamState(
                  camId,
                  "error"
                );

                setStreamErrors(
                  (previous) => ({
                    ...previous,
                    [camId]:
                      getErrorMessage(
                        error,
                        "Secure stream session expired or could not be created."
                      ),
                  })
                );
              }

            } finally {

              delete retryTimersRef
                .current[camId];

              if (
                mountedRef.current
              ) {

                setRetryingId(
                  (current) =>
                    current === camId
                      ? null
                      : current
                );
              }
            }

          },
          STREAM_RETRY_DELAY_MS *
            nextRetry
        );
      },
      [
        createStreamSession,
        getCameraKey,
        getErrorMessage,
        setCameraStreamState,
      ]
    );


  // ==========================================================
  // SELECTED CAMERA SCROLL
  // ==========================================================

  useEffect(() => {

    const selectedId =
      selectedCamera?.cam_id ||
      selectedCamera?.camera_id ||
      selectedCamera?.id;

    if (!selectedId) {
      return;
    }

    const card =
      cameraCardRefs.current[
        String(selectedId)
      ];

    if (card) {

      card.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }

  }, [selectedCamera]);


  // ==========================================================
  // COMPONENT MOUNT / INITIAL DATA LOAD
  // ==========================================================

  useEffect(() => {

    mountedRef.current = true;

    /*
     * Load cameras and system capacity
     * whenever CameraList mounts.
     */

    fetchCameras();

    fetchSystemCapacity();

    const healthTimer = window.setInterval(() => {
      refreshCameraHealth();
      fetchSystemCapacity();
    }, 5000);

    return () => {
      window.clearInterval(healthTimer);

      mountedRef.current = false;

      /*
       * Clear pending stream retry timers.
       */

      Object.values(
        retryTimersRef.current
      ).forEach((timer) => {
        clearTimeout(timer);
      });

      retryTimersRef.current = {};

      streamRetryCountsRef.current = {};

      startRequestsRef.current = {};

      streamSessionRequestsRef.current = {};

    };

  }, [
    fetchCameras,
    fetchSystemCapacity,
    refreshCameraHealth,
  ]);


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

      {/* ==================================================== */}
      {/* HEADER */}
      {/* ==================================================== */}

      <div className="col-span-full flex flex-col md:flex-row md:items-center md:justify-between gap-3">

        <h2 className="text-2xl font-bold text-blue-400">
          All Cameras
        </h2>

        {systemCapacity && (
          <div className="bg-[#0d213e] border border-[#1f4b7a] rounded-lg px-4 py-3 text-sm text-white">

            <span className="font-bold text-blue-400">
              System Capacity:
            </span>{" "}

            Running Cameras:{" "}

            <span className="font-bold">
              {
                systemCapacity.active_live_cameras ??
                0
              }
            </span>

            {" / "}

            <span className="font-bold">
              {
                systemCapacity.max_live_cameras ??
                0
              }
            </span>

          </div>
        )}

      </div>


      {/* ==================================================== */}
      {/* LOADING */}
      {/* ==================================================== */}

      {loading && (
        <div className="col-span-full flex justify-center p-8">
          <Loading />
        </div>
      )}


      {/* ==================================================== */}
      {/* EMPTY */}
      {/* ==================================================== */}

      {!loading &&
        cameras.length === 0 && (
          <div className="col-span-full flex flex-col items-center justify-center gap-3 py-16">

            <CameraOff
              size={100}
              className="text-gray-500"
            />

            <p className="text-2xl text-white font-bold">
              No cameras found
            </p>

          </div>
        )}


      {/* ==================================================== */}
      {/* CAMERA CARDS */}
      {/* ==================================================== */}

      {!loading &&
        cameras.map((camera) => {

          const camId =
            getCameraKey(camera);

          if (!camId) {
            return null;
          }

          const state =
            streamStates[
              camId
            ] || "idle";

          const streamUrl =
            streamUrls[
              camId
            ];

          const streamError =
            streamErrors[
              camId
            ];

          const hasLocation =
            hasValidCoordinates(
              camera
            );

          const locationName =
            getLocationName(
              camera
            );

          const isStarting =
            startingId === camId;

          const isStopping =
            stoppingId === camId;

          const isDeleting =
            deletingId === camId;

          const isOnline =
            state === "online";

          const isConnecting =
            state === "connecting" ||
            state === "starting";

          const isError =
            state === "error";

          const isRetrying =
            retryingId === camId;

          const isSelected =
            String(
              selectedCamera?.cam_id ||
                ""
            ) === camId;

          const backendStatus = String(
            camera?.connection_state || camera?.status || "OFFLINE"
          ).toUpperCase();

          // Backend state is authoritative for camera health. Browser stream
          // state is only used to describe the local preview connection.
          const displayStatus = backendStatus;

          const isBackendOnline = backendStatus === "ONLINE";
          const isBackendProblem = !["ONLINE", "DEGRADED"].includes(backendStatus);

          const isExternallyManaged = Boolean(camera?.externally_managed);

          const streamResolution = camera?.stream_width && camera?.stream_height
            ? `${camera.stream_width}×${camera.stream_height}`
            : null;


          return (
            <div
              key={camId}
              ref={(element) => {

                if (element) {

                  cameraCardRefs.current[
                    camId
                  ] = element;

                } else {

                  delete cameraCardRefs.current[
                    camId
                  ];
                }
              }}
              className={`bg-[var(--s1)] rounded-xl p-4 text-white border transition ${
                isSelected
                  ? "border-blue-500 ring-2 ring-blue-500/30"
                  : "border-[#1f4b7a]"
              }`}
            >

              {/* ======================================== */}
              {/* CAMERA HEADER */}
              {/* ======================================== */}

              <div className="mb-3 flex justify-between gap-3">

                <div className="flex flex-wrap items-center gap-4 min-w-0">

                  <h2 className="text-xl uppercase font-bold text-blue-400 break-words">
                    {getCameraName(camera)}
                  </h2>

                  <p className="text-md font-medium text-gray-300 min-w-0">
                    <span className="uppercase">Location:</span>{" "}
                    <span className="text-white font-semibold break-words">
                      {locationName || "Not provided"}
                    </span>
                  </p>

                  <p
                    className={`text-sm font-bold ${
                      isOnline
                        ? "text-green-400"
                        : isError
                          ? "text-red-400"
                          : isConnecting
                            ? "text-yellow-400"
                            : "text-red-400"
                    }`}
                  >
                    {displayStatus}
                  </p>

                  {isExternallyManaged && (
                    <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2.5 py-1 text-xs font-bold uppercase text-cyan-300">
                      {camera.external_provider || "External"} managed
                    </span>
                  )}

                </div>


                {/* DELETE */}

                <button
                  type="button"
                  onClick={() =>
                    deleteCamera(
                      camId
                    )
                  }
                  disabled={
                    isDeleting ||
                    isOnline ||
                    isStarting ||
                    isStopping ||
                    isExternallyManaged
                  }
                  title={isExternallyManaged ? "Remove this camera from its catalogue integration instead" : "Delete camera"}
                  className="h-9 px-3 rounded-md text-sm font-bold bg-red-600 hover:bg-red-700 disabled:bg-gray-500 disabled:cursor-not-allowed shrink-0"
                >

                  {isDeleting
                    ? "Deleting..."
                    : "Delete"}

                </button>

              </div>

              {isExternallyManaged && (
                <div className="mb-3 grid grid-cols-2 gap-2 rounded-lg border border-cyan-500/20 bg-cyan-500/5 p-3 text-xs sm:grid-cols-4">
                  <div><p className="text-slate-500">External ID</p><p className="truncate font-mono text-cyan-200" title={camera.external_camera_id || ""}>{camera.external_camera_id || "—"}</p></div>
                  <div><p className="text-slate-500">Catalogue status</p><p className="font-semibold text-white">{camera.external_live_status || "UNKNOWN"}</p></div>
                  <div><p className="text-slate-500">Codec / resolution</p><p className="font-semibold text-white">{[camera.codec, streamResolution].filter(Boolean).join(" · ") || "—"}</p></div>
                  <div><p className="text-slate-500">Last synchronized</p><p className="font-semibold text-white">{formatIndianDateTime(camera.external_last_synced_at, "—")}</p></div>
                </div>
              )}


              {/* ======================================== */}
              {/* CAMERA LOCATION */}
              {/* ======================================== */}

              <div className="mb-3 bg-[#091728] border border-[#21456d] rounded-lg p-3">

                <div className="flex items-start justify-between gap-3">

                  {/* LOCATION INFORMATION */}

                  <div className="flex items-start gap-3 min-w-0">

                    <div className="shrink-0 mt-1">

                      <MapPin
                        size={20}
                        className={
                          hasLocation
                            ? "text-green-400"
                            : "text-gray-500"
                        }
                      />

                    </div>


                    <div className="min-w-0">

                      {/* GIS COORDINATES */}

                      {hasLocation ? (

                        <div className="mt-1">

                          <p className="text-xs text-gray-400">
                            Latitude:{" "}
                            <span className="text-gray-200 font-medium">
                              {Number(
                                camera.latitude
                              ).toFixed(6)}
                            </span>
                          </p>

                          <p className="text-xs text-gray-400">
                            Longitude:{" "}
                            <span className="text-gray-200 font-medium">
                              {Number(
                                camera.longitude
                              ).toFixed(6)}
                            </span>
                          </p>

                        </div>

                      ) : (

                        <p className="text-xs text-red-400 mt-1">
                          GIS coordinates unavailable
                        </p>

                      )}

                    </div>

                  </div>


                  {/* MAP BUTTON */}

                  {typeof onCameraMapFocus ===
                    "function" && (

                    <button
                      type="button"
                      onClick={() => {

                        if (
                          hasLocation
                        ) {
                          onCameraMapFocus(
                            camera
                          );
                        }

                      }}
                      disabled={
                        !hasLocation
                      }
                      title={
                        hasLocation
                          ? "Show camera on map"
                          : "Camera location coordinates unavailable"
                      }
                      className="flex items-center gap-2 px-3 py-2 rounded-md bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-sm font-bold shrink-0"
                    >

                      <Navigation
                        size={15}
                      />

                      Map

                    </button>
                  )}

                </div>

              </div>


              {/* ======================================== */}
              {/* VIDEO / STREAM AREA */}
              {/* ======================================== */}

              <div className="relative w-full h-[320px] rounded-lg bg-black border border-[#21456d] flex items-center justify-center overflow-hidden">

                {streamUrl ? (

                  <>

                    <img
                      key={streamUrl}
                      src={streamUrl}
                      alt={`Live stream for ${
                        getCameraName(camera)
                      }`}
                      className="w-full h-full object-contain"
                      crossOrigin="use-credentials"
                      referrerPolicy="no-referrer"
                      draggable="false"

                      onLoad={() => {

                        /*
                         * MJPEG stream is now
                         * actually delivering data.
                         */

                        if (
                          mountedRef.current
                        ) {

                          streamRetryCountsRef.current[
                            camId
                          ] = 0;

                          setCameraStreamState(
                            camId,
                            "online"
                          );

                          setStreamErrors(
                            (previous) => {

                              const next = {
                                ...previous,
                              };

                              delete next[
                                camId
                              ];

                              return next;
                            }
                          );

                          setRetryingId(
                            (current) =>
                              current ===
                                camId
                                ? null
                                : current
                          );
                        }
                      }}

                      onError={() => {

                        handleStreamError(
                          camera
                        );

                      }}
                    />


                    {/* ================================== */}
                    {/* CONNECTING */}
                    {/* ================================== */}

                    {isConnecting &&
                      !isRetrying && (

                        <div className="absolute inset-0 flex items-center justify-center bg-black/90">

                          <div className="flex flex-col items-center gap-3 text-center px-5">

                            <Loader2
                              size={48}
                              className="animate-spin text-blue-400"
                            />

                            <p className="text-lg font-bold text-white">
                              Connecting Camera...
                            </p>

                            <p className="text-sm text-gray-400">
                              Waiting for the camera worker to provide video
                            </p>

                          </div>

                        </div>
                      )}


                    {/* ================================== */}
                    {/* RETRYING */}
                    {/* ================================== */}

                    {isRetrying && (

                      <div className="absolute inset-0 flex items-center justify-center bg-black/80">

                        <div className="flex flex-col items-center gap-3">

                          <Loader2
                            size={35}
                            className="animate-spin text-blue-400"
                          />

                          <p className="text-white font-bold">
                            Reconnecting...
                          </p>

                        </div>

                      </div>
                    )}


                    {/* ================================== */}
                    {/* ERROR */}
                    {/* ================================== */}

                    {isError && (

                      <div className="absolute inset-0 flex items-center justify-center bg-black/95">

                        <div className="max-w-md px-5 text-center">

                          <AlertCircle
                            size={60}
                            className="mx-auto text-red-500 mb-3"
                          />

                          <p className="text-lg font-bold text-red-400">
                            Camera Source Unavailable
                          </p>

                          <p className="text-sm text-gray-300 mt-2">
                            {streamError ||
                              "The backend camera worker could not open the configured source."}
                          </p>

                          <p className="text-xs text-gray-500 mt-3">
                            Camera ID:{" "}
                            {camId}
                          </p>

                        </div>

                      </div>
                    )}

                  </>

                ) : (

                  <div className="flex flex-col items-center gap-2 text-gray-400">

                    <CameraOff
                      size={82}
                    />

                    <p className="text-lg font-bold">
                      Camera Offline
                    </p>

                    <p className="text-sm text-center">
                      Click Start Stream to view live feed
                    </p>

                  </div>

                )}

              </div>


              {/* ======================================== */}
              {/* ERROR MESSAGE */}
              {/* ======================================== */}

              {isError &&
                streamError && (

                  <div className="mt-3 flex items-start gap-2 rounded-md border border-red-500/30 bg-red-500/10 px-3 py-2">

                    <AlertCircle
                      size={18}
                      className="text-red-400 mt-0.5 shrink-0"
                    />

                    <p className="text-xs text-red-300">
                      {streamError}
                    </p>

                  </div>
                )}


              {/* ======================================== */}
              {/* START / STOP BUTTON */}
              {/* ======================================== */}

              <button
                type="button"
                onClick={() => {

                  if (isOnline) {

                    stopCamera(
                      camera
                    );

                  } else {

                    startCamera(
                      camera
                    );

                  }

                }}
                disabled={
                  isStarting ||
                  isStopping
                }
                className={`w-full mt-3 py-3 rounded-md font-bold transition flex items-center justify-center gap-2 ${
                  isStarting ||
                  isStopping
                    ? "bg-gray-500 cursor-not-allowed"
                    : isOnline
                      ? "bg-red-600 hover:bg-red-700"
                      : "bg-blue-600 hover:bg-blue-700"
                }`}
              >

                {isStarting ? (

                  <>
                    <Loader2
                      size={18}
                      className="animate-spin"
                    />
                    Starting...
                  </>

                ) : isStopping ? (

                  <>
                    <Loader2
                      size={18}
                      className="animate-spin"
                    />
                    Stopping...
                  </>

                ) : isOnline ? (

                  "Stop Stream"

                ) : isError ? (

                  <>
                    <RefreshCw
                      size={17}
                    />
                    Retry Stream
                  </>

                ) : (

                  "Start Stream"

                )}

              </button>

            </div>
          );
        })}

    </div>
  );
};


export default CameraList;
