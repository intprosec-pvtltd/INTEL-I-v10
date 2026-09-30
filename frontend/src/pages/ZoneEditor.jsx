import React, {
  useCallback,
  useEffect,
  useMemo,
  useState,
} from "react";

import toast from "react-hot-toast";

import {
  Camera,
  CheckCircle2,
  RefreshCw,
  Save,
  ShieldCheck,
  ShieldAlert,
  MapPin,
  Search,
} from "lucide-react";

import api from "../api/axios";


/* ============================================================
 * CONSTANTS
 * ============================================================ */

const NOT_RESTRICTED = "not_restricted";
const RESTRICTED = "restricted";


/* ============================================================
 * HELPERS
 * ============================================================ */

const getErrorMessage = (
  error,
  fallback = "Request failed"
) => {
  const detail =
    error?.response?.data?.detail;

  if (typeof detail === "string") {
    return detail;
  }

  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (typeof item === "string") {
          return item;
        }

        if (
          item &&
          typeof item.msg === "string"
        ) {
          return item.msg;
        }

        return "Validation error";
      })
      .join(", ");
  }

  if (
    detail &&
    typeof detail === "object" &&
    typeof detail.msg === "string"
  ) {
    return detail.msg;
  }

  return (
    error?.message ||
    fallback
  );
};


const normalizeCameraId = (
  value
) =>
  String(
    value ?? ""
  ).trim();


const getCameraName = (
  camera
) =>
  camera?.camera_name ||
  camera?.name ||
  camera?.camera_id ||
  camera?.cam_id ||
  "Unnamed Camera";


const getCameraLocation = (
  camera
) => {
  const latitude =
    camera?.latitude ??
    camera?.lat;

  const longitude =
    camera?.longitude ??
    camera?.lng ??
    camera?.lon;

  if (
    latitude === undefined ||
    longitude === undefined ||
    latitude === null ||
    longitude === null
  ) {
    return null;
  }

  return `${Number(latitude).toFixed(6)}, ${Number(
    longitude
  ).toFixed(6)}`;
};


/* ============================================================
 * COMPONENT
 * ============================================================ */

const ZoneEditor = () => {
  const [cameras, setCameras] =
    useState([]);

  const [selectedCamera, setSelectedCamera] =
    useState("");

  const [cameraZone, setCameraZone] =
    useState(NOT_RESTRICTED);

  const [loading, setLoading] =
    useState(true);

  const [refreshing, setRefreshing] =
    useState(false);

  const [saving, setSaving] =
    useState(false);

  const [search, setSearch] =
    useState("");

  const [errorMessage, setErrorMessage] =
    useState("");

  /* ==========================================================
   * AUTH
   * ========================================================== */

  const getAuthHeaders = useCallback(
    () => {
      const token =
        localStorage.getItem(
          "token"
        );

      return token
        ? {
            Authorization: `Bearer ${token}`,
          }
        : {};
    },
    []
  );


  /* ==========================================================
   * FILTERED CAMERAS
   * ========================================================== */

  const filteredCameras =
    useMemo(() => {
      const query =
        search.trim().toLowerCase();

      if (!query) {
        return cameras;
      }

      return cameras.filter(
        (camera) =>
          getCameraName(
            camera
          )
            .toLowerCase()
            .includes(query) ||
          normalizeCameraId(
            camera?.cam_id ??
              camera?.camera_id
          )
            .toLowerCase()
            .includes(query)
      );
    }, [
      cameras,
      search,
    ]);


  /* ==========================================================
   * SELECTED CAMERA OBJECT
   * ========================================================== */

  const selectedCameraObject =
    useMemo(
      () =>
        cameras.find(
          (camera) =>
            normalizeCameraId(
              camera?.cam_id ??
                camera?.camera_id
            ) ===
            selectedCamera
        ) || null,
      [
        cameras,
        selectedCamera,
      ]
    );


  /* ==========================================================
   * FETCH CAMERAS
   * ========================================================== */

  const fetchCameras = useCallback(
    async (
      silent = false
    ) => {
      if (silent) {
        setRefreshing(true);
      } else {
        setLoading(true);
      }

      setErrorMessage("");

      try {
        const response =
          await api.get(
            "/cameras",
            {
              headers:
                getAuthHeaders(),
              withCredentials: true,
            }
          );

        const nextCameras =
          Array.isArray(
            response?.data?.cameras
          )
            ? response.data.cameras
            : [];

        setCameras(
          nextCameras
        );

        /*
         * Preserve the currently selected camera.
         * Otherwise select the first available camera.
         */
        setSelectedCamera(
          (current) => {
            const stillExists =
              current &&
              nextCameras.some(
                (camera) =>
                  normalizeCameraId(
                    camera?.cam_id ??
                      camera?.camera_id
                  ) ===
                  current
              );

            if (
              stillExists
            ) {
              return current;
            }

            return nextCameras.length
              ? normalizeCameraId(
                  nextCameras[0]
                    ?.cam_id ??
                    nextCameras[0]
                      ?.camera_id
                )
              : "";
          }
        );
      } catch (error) {
        console.error(
          "Failed to fetch cameras:",
          error
        );

        setCameras([]);

        setErrorMessage(
          getErrorMessage(
            error,
            "Failed to load cameras"
          )
        );

        toast.error(
          getErrorMessage(
            error,
            "Failed to load cameras"
          )
        );
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [
      getAuthHeaders,
    ]
  );


  /* ==========================================================
   * INITIAL LOAD
   * ========================================================== */

  useEffect(() => {
    fetchCameras();
  }, [
    fetchCameras,
  ]);


  /* ==========================================================
   * CAMERA CHANGE
   * ========================================================== */

  const handleCameraChange = (
    camId
  ) => {
    const normalized =
      normalizeCameraId(
        camId
      );

    setSelectedCamera(
      normalized
    );

    const camera =
      cameras.find(
        (item) =>
          normalizeCameraId(
            item?.cam_id ??
              item?.camera_id
          ) ===
          normalized
      );

    if (camera) {
      setCameraZone(
        String(
          camera.zone ||
            camera.zone_type ||
            NOT_RESTRICTED
        ).toLowerCase() ===
          RESTRICTED
          ? RESTRICTED
          : NOT_RESTRICTED
      );
    } else {
      setCameraZone(
        NOT_RESTRICTED
      );
    }
  };


  /* ==========================================================
   * SAVE ZONE
   * ========================================================== */

  const saveZone = async () => {
    if (!selectedCamera) {
      toast.error(
        "Select a camera first"
      );
      return;
    }

    setSaving(true);

    try {
      await api.put(
        `/camera/${encodeURIComponent(
          selectedCamera
        )}/zone`,
        {
          zone: cameraZone,
        },
        {
          headers:
            getAuthHeaders(),
          withCredentials: true,
        }
      );

      toast.success(
        cameraZone === RESTRICTED
          ? "Camera marked as Restricted"
          : "Camera marked as Not Restricted"
      );

      /*
       * Refresh camera data so the saved state
       * shown by the UI comes directly from the backend.
       */
      await fetchCameras(
        true
      );
    } catch (error) {
      console.error(
        "Failed to update camera zone:",
        error
      );

      toast.error(
        getErrorMessage(
          error,
          "Failed to update camera zone"
        )
      );
    } finally {
      setSaving(false);
    }
  };


  /* ==========================================================
   * QUICK STATUS
   * ========================================================== */

  const isRestricted =
    cameraZone ===
    RESTRICTED;


  return (
    <div className="min-h-[calc(100vh-6rem)] w-full text-white">

      <div className="overflow-hidden rounded-2xl border border-slate-800 bg-[#07101b] shadow-[0_15px_50px_rgba(0,0,0,0.20)]">

        {/* ====================================================
         * HEADER
         * ==================================================== */}

        <div className="relative overflow-hidden border-b border-slate-800 bg-gradient-to-br from-[#091426] via-[#07101d] to-[#050b14] px-5 py-6 sm:px-7">

          <div className="pointer-events-none absolute -right-16 -top-20 h-56 w-56 rounded-full bg-blue-500/10 blur-3xl" />

          <div className="relative flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">

            <div>
              <div className="mb-3 flex items-center gap-2">
                <div className="rounded-lg border border-blue-400/20 bg-blue-400/10 p-2 text-blue-300">
                  <MapPin size={18} />
                </div>

                <span className="text-[10px] font-semibold uppercase tracking-[0.2em] text-blue-300/80">
                  Camera Configuration
                </span>
              </div>

              <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
                Zone Configuration
              </h1>

              <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">
                Configure whether a camera monitors a
                restricted zone and should generate
                restricted-zone alerts.
              </p>
            </div>


            <button
              type="button"
              onClick={() =>
                fetchCameras(
                  true
                )
              }
              disabled={
                loading ||
                refreshing ||
                saving
              }
              className="inline-flex items-center justify-center gap-2 rounded-lg border border-slate-700 bg-slate-900/70 px-4 py-2.5 text-sm font-semibold text-slate-300 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <RefreshCw
                size={16}
                className={
                  refreshing
                    ? "animate-spin"
                    : ""
                }
              />

              {refreshing
                ? "Refreshing..."
                : "Refresh Cameras"}
            </button>
          </div>
        </div>


        {/* ====================================================
         * BODY
         * ==================================================== */}

        <div className="grid min-h-[calc(100vh-16rem)] lg:grid-cols-[320px_minmax(0,1fr)]">

          {/* ==================================================
           * CAMERA LIST
           * ================================================== */}

          <aside className="border-b border-slate-800 bg-[#06101a] p-4 lg:border-b-0 lg:border-r xl:p-5">

            <div className="mb-4 flex items-center justify-between">
              <div>
                <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">
                  Cameras
                </div>

                <div className="mt-1 text-sm font-semibold text-slate-300">
                  Select camera
                </div>
              </div>

              <span className="rounded-full border border-slate-800 bg-slate-950 px-2.5 py-1 text-[10px] text-slate-500">
                {cameras.length}
              </span>
            </div>


            {/* SEARCH */}

            <div className="relative mb-3">
              <Search
                size={15}
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-600"
              />

              <input
                value={search}
                onChange={(event) =>
                  setSearch(
                    event.target.value
                  )
                }
                placeholder="Search camera..."
                className="w-full rounded-lg border border-slate-800 bg-slate-950/70 py-2.5 pl-9 pr-3 text-sm text-white outline-none placeholder:text-slate-600 focus:border-blue-500/50"
              />
            </div>


            {errorMessage && (
              <div className="mb-3 rounded-xl border border-red-500/20 bg-red-500/[0.04] p-3 text-xs leading-5 text-red-300">
                {errorMessage}
              </div>
            )}


            {loading ? (
              <div className="flex min-h-[260px] items-center justify-center">
                <div className="text-center">
                  <RefreshCw
                    size={24}
                    className="mx-auto mb-3 animate-spin text-blue-300"
                  />

                  <div className="text-xs text-slate-500">
                    Loading cameras...
                  </div>
                </div>
              </div>
            ) : filteredCameras.length === 0 ? (
              <div className="rounded-xl border border-dashed border-slate-800 bg-slate-950/50 p-7 text-center">
                <Camera
                  size={30}
                  className="mx-auto mb-3 text-slate-700"
                />

                <div className="text-sm font-medium text-slate-400">
                  No cameras found
                </div>

                <div className="mt-1 text-xs text-slate-600">
                  Check the camera configuration or search term.
                </div>
              </div>
            ) : (
              <div className="max-h-[580px] space-y-2 overflow-y-auto pr-1">

                {filteredCameras.map(
                  (camera) => {
                    const camId =
                      normalizeCameraId(
                        camera?.cam_id ??
                          camera?.camera_id
                      );

                    const active =
                      selectedCamera ===
                      camId;

                    const zone =
                      String(
                        camera?.zone ||
                          camera?.zone_type ||
                          NOT_RESTRICTED
                      ).toLowerCase() ===
                      RESTRICTED
                        ? RESTRICTED
                        : NOT_RESTRICTED;

                    return (
                      <button
                        key={
                          camId ||
                          getCameraName(
                            camera
                          )
                        }
                        type="button"
                        onClick={() =>
                          handleCameraChange(
                            camId
                          )
                        }
                        className={`w-full rounded-xl border p-3 text-left transition ${
                          active
                            ? "border-blue-500/50 bg-blue-500/[0.07]"
                            : "border-slate-800 bg-slate-950/40 hover:border-slate-700 hover:bg-slate-900/60"
                        }`}
                      >
                        <div className="flex items-start gap-3">

                          <div
                            className={`mt-0.5 flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border ${
                              active
                                ? "border-blue-500/30 bg-blue-500/10 text-blue-300"
                                : "border-slate-800 bg-slate-900 text-slate-600"
                            }`}
                          >
                            <Camera
                              size={18}
                            />
                          </div>


                          <div className="min-w-0 flex-1">

                            <div className="flex items-center justify-between gap-2">
                              <div className="truncate text-sm font-semibold text-slate-200">
                                {getCameraName(
                                  camera
                                )}
                              </div>

                              <span
                                className={`h-2 w-2 shrink-0 rounded-full ${
                                  zone ===
                                  RESTRICTED
                                    ? "bg-red-400"
                                    : "bg-emerald-400"
                                }`}
                              />
                            </div>


                            <div className="mt-1 truncate font-mono text-[10px] text-slate-600">
                              {camId ||
                                "NO CAMERA ID"}
                            </div>


                            <div className="mt-2">
                              <span
                                className={`inline-flex rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-wide ${
                                  zone ===
                                  RESTRICTED
                                    ? "border-red-500/20 bg-red-500/5 text-red-300"
                                    : "border-emerald-500/20 bg-emerald-500/5 text-emerald-300"
                                }`}
                              >
                                {zone ===
                                RESTRICTED
                                  ? "Restricted"
                                  : "Not Restricted"}
                              </span>
                            </div>
                          </div>
                        </div>
                      </button>
                    );
                  }
                )}
              </div>
            )}
          </aside>


          {/* ==================================================
           * CONFIGURATION PANEL
           * ================================================== */}

          <main className="min-w-0 w-full p-5 sm:p-7 lg:p-8">

            {!selectedCamera ? (
              <div className="flex min-h-[500px] items-center justify-center rounded-2xl border border-dashed border-slate-800 bg-slate-950/30 text-center">
                <div>
                  <div className="mx-auto mb-4 flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-800 bg-slate-950 text-slate-600">
                    <Camera size={29} />
                  </div>

                  <div className="text-base font-semibold text-slate-400">
                    Select a camera
                  </div>

                  <div className="mt-1 max-w-sm text-xs leading-5 text-slate-600">
                    Choose a camera from the list to configure
                    its restricted-zone monitoring state.
                  </div>
                </div>
              </div>
            ) : (
              <div className="w-full space-y-6 xl:max-w-none">

                {/* CAMERA INFO */}

                <div className="w-full rounded-xl border border-slate-800 bg-slate-950/40 p-4">
                  <div className="flex items-start gap-3">

                    <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl border border-blue-500/20 bg-blue-500/10 text-blue-300">
                      <Camera
                        size={20}
                      />
                    </div>

                    <div className="min-w-0">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">
                        Selected Camera
                      </div>

                      <div className="mt-1 truncate text-lg font-semibold text-white">
                        {getCameraName(
                          selectedCameraObject
                        )}
                      </div>

                      <div className="mt-1 font-mono text-xs text-slate-600">
                        {selectedCamera}
                      </div>

                      {getCameraLocation(
                        selectedCameraObject
                      ) && (
                        <div className="mt-2 flex items-center gap-1.5 text-xs text-slate-500">
                          <MapPin
                            size={13}
                          />

                          {getCameraLocation(
                            selectedCameraObject
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                </div>


                {/* ZONE OPTIONS */}

                <div>
                  <div className="mb-3">
                    <div className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">
                      Monitoring Mode
                    </div>

                    <h2 className="mt-1 text-lg font-semibold text-slate-200">
                      Camera Zone Status
                    </h2>

                    <p className="mt-1 text-xs leading-5 text-slate-600">
                      Choose whether the selected camera belongs to a
                      restricted monitoring zone.
                    </p>
                  </div>


                  <div className="grid w-full gap-4 md:grid-cols-2">

                    {/* NOT RESTRICTED */}

                    <button
                      type="button"
                      onClick={() =>
                        setCameraZone(
                          NOT_RESTRICTED
                        )
                      }
                      disabled={saving}
                      className={`relative rounded-2xl border p-4 text-left transition ${
                        cameraZone ===
                        NOT_RESTRICTED
                          ? "border-emerald-500/50 bg-emerald-500/[0.07] shadow-[0_10px_30px_rgba(16,185,129,0.06)]"
                          : "border-slate-800 bg-slate-950/40 hover:border-emerald-500/30 hover:bg-emerald-500/[0.03]"
                      }`}
                    >

                      <div className="flex items-start justify-between gap-3">

                        <div className="flex items-center gap-3">
                          <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-emerald-500/20 bg-emerald-500/10 text-emerald-300">
                            <ShieldCheck
                              size={21}
                            />
                          </div>

                          <div>
                            <div className="text-sm font-semibold text-slate-200">
                              Not Restricted
                            </div>

                            <div className="mt-1 text-[11px] text-slate-600">
                              Standard camera monitoring
                            </div>
                          </div>
                        </div>

                        {cameraZone ===
                          NOT_RESTRICTED && (
                          <CheckCircle2
                            size={18}
                            className="text-emerald-300"
                          />
                        )}
                      </div>

                      <div className="mt-4 text-xs leading-5 text-slate-500">
                        Restricted-zone alerts are not enabled
                        for this camera.
                      </div>
                    </button>


                    {/* RESTRICTED */}

                    <button
                      type="button"
                      onClick={() =>
                        setCameraZone(
                          RESTRICTED
                        )
                      }
                      disabled={saving}
                      className={`relative rounded-2xl border p-4 text-left transition ${
                        cameraZone ===
                        RESTRICTED
                          ? "border-red-500/50 bg-red-500/[0.07] shadow-[0_10px_30px_rgba(239,68,68,0.06)]"
                          : "border-slate-800 bg-slate-950/40 hover:border-red-500/30 hover:bg-red-500/[0.03]"
                      }`}
                    >

                      <div className="flex items-start justify-between gap-3">

                        <div className="flex items-center gap-3">
                          <div className="flex h-11 w-11 items-center justify-center rounded-xl border border-red-500/20 bg-red-500/10 text-red-300">
                            <ShieldAlert
                              size={21}
                            />
                          </div>

                          <div>
                            <div className="text-sm font-semibold text-slate-200">
                              Restricted
                            </div>

                            <div className="mt-1 text-[11px] text-slate-600">
                              Restricted-zone monitoring enabled
                            </div>
                          </div>
                        </div>

                        {cameraZone ===
                          RESTRICTED && (
                          <CheckCircle2
                            size={18}
                            className="text-red-300"
                          />
                        )}
                      </div>

                      <div className="mt-4 text-xs leading-5 text-slate-500">
                        The camera can generate alerts when
                        restricted-zone activity is detected.
                      </div>
                    </button>
                  </div>
                </div>


                {/* CURRENT STATE */}

                <div
                  className={`w-full rounded-xl border p-4 ${
                    isRestricted
                      ? "border-red-500/20 bg-red-500/[0.04]"
                      : "border-emerald-500/20 bg-emerald-500/[0.04]"
                  }`}
                >
                  <div className="flex items-start gap-3">

                    <div
                      className={`mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-lg ${
                        isRestricted
                          ? "bg-red-500/10 text-red-300"
                          : "bg-emerald-500/10 text-emerald-300"
                      }`}
                    >
                      {isRestricted ? (
                        <ShieldAlert
                          size={17}
                        />
                      ) : (
                        <ShieldCheck
                          size={17}
                        />
                      )}
                    </div>

                    <div className="min-w-0">
                      <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-slate-600">
                        Current Configuration
                      </div>

                      <div
                        className={`mt-1 text-sm font-semibold ${
                          isRestricted
                            ? "text-red-300"
                            : "text-emerald-300"
                        }`}
                      >
                        {isRestricted
                          ? "Restricted"
                          : "Not Restricted"}
                      </div>

                      <div className="mt-1 text-xs leading-5 text-slate-600">
                        {isRestricted
                          ? "Restricted-zone monitoring is enabled for this camera."
                          : "Restricted-zone monitoring is disabled for this camera."}
                      </div>
                    </div>
                  </div>
                </div>


                {/* SAVE */}

                <div className="flex flex-col gap-3 border-t border-slate-800 pt-5 sm:flex-row sm:items-center sm:justify-between">

                  <div className="text-[11px] leading-5 text-slate-600">
                    Changes are applied to camera{" "}
                    <span className="font-mono text-slate-400">
                      {selectedCamera}
                    </span>
                    .
                  </div>

                  <button
                    type="button"
                    onClick={
                      saveZone
                    }
                    disabled={
                      saving ||
                      refreshing
                    }
                    className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-5 py-2.5 text-sm font-semibold text-white hover:bg-blue-500 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    {saving ? (
                      <RefreshCw
                        size={16}
                        className="animate-spin"
                      />
                    ) : (
                      <Save
                        size={16}
                      />
                    )}

                    {saving
                      ? "Saving..."
                      : "Save Changes"}
                  </button>
                </div>
              </div>
            )}
          </main>
        </div>
      </div>
    </div>
  );
};


export default ZoneEditor;
