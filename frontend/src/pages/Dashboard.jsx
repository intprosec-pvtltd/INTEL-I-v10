import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import Alert from "../components/Alert";
import CameraList from "../components/CameraList";
import GISMap from "../components/GisMap";
import SystemHealth from "../components/SystemHealth";


import api from "../api/axios";
import toast from "react-hot-toast";


const Dashboard = () => {
  // ==========================================================
  // CAMERA STATE
  // ==========================================================

  const [cameras, setCameras] = useState([]);

  const [
    selectedCamera,
    setSelectedCamera,
  ] = useState(null);

  const [
    selectedLocation,
    setSelectedLocation,
  ] = useState(null);

  const [
    autoStartCameraId,
    setAutoStartCameraId,
  ] = useState(null);

  const [
    loadingCameras,
    setLoadingCameras,
  ] = useState(false);


  // ==========================================================
  // SYSTEM HEALTH
  // ==========================================================

  /*
   * IMPORTANT:
   *
   * System Health is NOT automatically opened when Dashboard
   * is entered.
   *
   * The health service is checked silently in the background.
   *
   * The operator can open the detailed System Health modal
   * manually using the System Health button.
   */

  const [
    isHealthModalOpen,
    setIsHealthModalOpen,
  ] = useState(false);

  const [
    isInitializing,
    setIsInitializing,
  ] = useState(false);

  const [
    health,
    setHealth,
  ] = useState(null);

  const [
    healthError,
    setHealthError,
  ] = useState("");

  const [
    healthLoading,
    setHealthLoading,
  ] = useState(false);

  const [
    healthRefreshing,
    setHealthRefreshing,
  ] = useState(false);

  const healthRequestInFlightRef =
    useRef(false);

  const healthRef =
    useRef(null);


  // ==========================================================
  // FETCH SYSTEM HEALTH
  // ==========================================================

  const fetchSystemHealth = useCallback(
    async ({
      startup = false,
      retry = false,
    } = {}) => {

      /*
       * Prevent duplicate health requests.
       */
      if (
        healthRequestInFlightRef.current
      ) {
        return;
      }

      healthRequestInFlightRef.current = true;


      /*
       * Startup health check runs silently.
       *
       * Manual refresh/open can show loading state.
       */
      if (startup) {
        setHealthLoading(true);
      } else {
        setHealthRefreshing(true);
      }


      try {

        const response =
          await api.get(
            "/api/system/health",
            {
              timeout: 10000,
            }
          );


        const payload =
          response?.data;


        /*
         * Validate response.
         */
        if (
          !payload ||
          typeof payload !== "object"
        ) {
          throw new Error(
            "Invalid system health response"
          );
        }


        /*
         * Determine whether backend is still
         * initializing.
         */
        const backendInitializing =
          payload?.initializing === true ||
          String(
            payload?.initialization?.status || ""
          ).toUpperCase() ===
          "INITIALIZING" ||
          String(
            payload?.overall || ""
          ).toUpperCase() ===
          "INITIALIZING";


        /*
         * Store latest health.
         */
        setHealth(payload);

        healthRef.current =
          payload;


        setHealthError("");

        setIsInitializing(
          backendInitializing
        );

      } catch (error) {

        console.error(
          "Failed to fetch INTEL-I system health:",
          error
        );


        /*
         * Convert backend/network errors
         * into safe user-facing messages.
         */
        setHealthError(
          error?.response?.status === 401
            ? "Authentication required to retrieve system health."

            : error?.response?.status === 403
              ? "Access denied for the system health service."

              : error?.response?.status === 404
                ? "System health endpoint was not found."

                : error?.response?.status >= 500
                  ? "Health service unavailable. The backend returned an error."

                  : error?.code ===
                    "ECONNABORTED"
                    ? "Health service request timed out."

                    : error?.message ||
                    "Unable to retrieve system health."
        );


        /*
         * If there is no previous health information,
         * represent the state as unavailable.
         */
        if (!healthRef.current) {
          setIsInitializing(false);
        }


        /*
         * retry/startup parameters are intentionally
         * not used to automatically open the modal.
         *
         * The modal is operator-controlled.
         */
        void retry;

      } finally {

        healthRequestInFlightRef.current =
          false;

        setHealthLoading(false);

        setHealthRefreshing(false);
      }
    },
    []
  );


  // ==========================================================
  // CLOSE SYSTEM HEALTH
  // ==========================================================

  const closeHealthModal =
    useCallback(() => {

      setIsHealthModalOpen(false);

    }, []);


  // ==========================================================
  // OPEN SYSTEM HEALTH
  // ==========================================================

  const openHealthModal =
    useCallback(() => {

      /*
       * Open the detailed health modal only
       * when the operator explicitly requests it.
       */
      setIsHealthModalOpen(true);


      /*
       * Perform a fresh health request.
       */
      fetchSystemHealth({
        startup: false,
        retry: true,
      });

    }, [
      fetchSystemHealth,
    ]);


  // ==========================================================
  // BACKGROUND SYSTEM HEALTH CHECK
  // ==========================================================

  useEffect(() => {

    let cancelled = false;


    const checkHealthInBackground =
      async () => {

        if (cancelled) {
          return;
        }

        /*
         * Silent background health check.
         *
         * IMPORTANT:
         * No modal is opened here.
         */
        await fetchSystemHealth({
          startup: true,
        });
      };


    checkHealthInBackground();


    /*
     * Cleanup.
     */
    return () => {

      cancelled = true;

    };

  }, [
    fetchSystemHealth,
  ]);


  // ==========================================================
  // GIS SECTION
  // ==========================================================

  const gisSectionRef =
    useRef(null);


  // ==========================================================
  // VALIDATE CAMERA COORDINATES
  // ==========================================================

  const hasValidCoordinates =
    useCallback(
      (camera) => {

        const latitude =
          Number(
            camera?.latitude
          );

        const longitude =
          Number(
            camera?.longitude
          );


        return (
          Number.isFinite(
            latitude
          ) &&
          Number.isFinite(
            longitude
          ) &&
          latitude >= -90 &&
          latitude <= 90 &&
          longitude >= -180 &&
          longitude <= 180
        );
      },
      []
    );


  // ==========================================================
  // FETCH CAMERAS
  // ==========================================================

  const fetchCameras =
    useCallback(
      async () => {

        try {

          setLoadingCameras(
            true
          );


          const response =
            await api.get(
              "/cameras"
            );


          const cameraList =
            Array.isArray(
              response?.data
                ?.cameras
            )
              ? response.data.cameras
              : [];


          setCameras(
            cameraList
          );


          /*
           * Keep selected camera synchronized
           * with latest backend data.
           */
          setSelectedCamera(
            (previous) => {

              if (
                !previous?.cam_id
              ) {
                return previous;
              }


              return (
                cameraList.find(
                  (camera) =>
                    camera?.cam_id ===
                    previous.cam_id
                ) || null
              );
            }
          );

        } catch (error) {

          console.error(
            "Failed to fetch cameras for GIS",
            error
          );


          toast.error(
            "Failed to load cameras"
          );


          setCameras([]);

        } finally {

          setLoadingCameras(
            false
          );
        }

      },
      []
    );


  // ==========================================================
  // CAMERA AUTO REFRESH
  // ==========================================================

  useEffect(() => {

    /*
     * Initial camera request.
     */
    fetchCameras();


    /*
     * Refresh camera information every
     * 30 seconds.
     */
    const intervalId =
      window.setInterval(
        () => {

          fetchCameras();

        },
        30000
      );


    return () => {

      window.clearInterval(
        intervalId
      );

    };

  }, [
    fetchCameras,
  ]);


  // ==========================================================
  // CAMERA SELECT
  // ==========================================================

  const handleCameraSelect =
    useCallback(
      (camera) => {

        if (!camera) {
          return;
        }


        if (
          !hasValidCoordinates(
            camera
          )
        ) {

          toast.error(
            "Camera location is not configured"
          );

          return;
        }


        setSelectedCamera(
          camera
        );


        setSelectedLocation(
          null
        );

      },
      [
        hasValidCoordinates,
      ]
    );


  // ==========================================================
  // CAMERA MAP FOCUS
  // ==========================================================

  const handleCameraMapFocus =
    useCallback(
      (camera) => {

        if (!camera) {
          return;
        }


        if (
          !hasValidCoordinates(
            camera
          )
        ) {

          toast.error(
            "Camera location is not configured"
          );

          return;
        }


        setSelectedCamera(
          camera
        );


        setSelectedLocation(
          null
        );


        /*
         * Focus GIS section.
         */
        window.requestAnimationFrame(
          () => {

            gisSectionRef.current?.scrollIntoView(
              {
                behavior:
                  "smooth",

                block:
                  "center",
              }
            );

          }
        );

      },
      [
        hasValidCoordinates,
      ]
    );


  // ==========================================================
  // AUTO START HANDLED
  // ==========================================================

  const handleAutoStartHandled =
    useCallback(
      () => {

        /*
         * Clear auto-start command after
         * CameraList has processed it.
         */
        setAutoStartCameraId(
          null
        );

      },
      []
    );


  // ==========================================================
  // ALERT → CAMERA + GIS
  // ==========================================================

  const handleAlertSelect =
    useCallback(
      (alert) => {

        const alertCamId =
          String(
            alert?.cam_id || ""
          ).trim();


        if (!alertCamId) {

          toast.error(
            "Alert has no camera ID"
          );

          return;
        }


        const camera =
          cameras.find(
            (item) =>
              String(
                item?.cam_id || ""
              ) === alertCamId
          );


        if (!camera) {

          toast.error(
            "Camera associated with this alert was not found"
          );

          return;
        }


        if (
          !hasValidCoordinates(
            camera
          )
        ) {

          toast.error(
            "Camera location is not configured"
          );

          return;
        }


        setSelectedCamera(
          camera
        );


        setSelectedLocation(
          null
        );


        setAutoStartCameraId(
          alertCamId
        );


        /*
         * Focus camera section.
         */
        window.requestAnimationFrame(
          () => {

            const element =
              document.getElementById(
                "dashboard-camera-section"
              );


            if (element) {

              element.scrollIntoView(
                {
                  behavior:
                    "smooth",

                  block:
                    "start",
                }
              );

            }

          }
        );


        /*
         * Focus GIS after camera section.
         */
        window.setTimeout(
          () => {

            gisSectionRef.current?.scrollIntoView(
              {
                behavior:
                  "smooth",

                block:
                  "center",
              }
            );

          },
          500
        );

      },
      [
        cameras,
        hasValidCoordinates,
      ]
    );


  // ==========================================================
  // GIS MAP CLICK
  // ==========================================================

  const handleMapClick =
    useCallback(
      (location) => {

        if (!location) {
          return;
        }


        const latitude =
          Number(
            location.latitude
          );


        const longitude =
          Number(
            location.longitude
          );


        if (
          !Number.isFinite(
            latitude
          ) ||
          !Number.isFinite(
            longitude
          ) ||
          latitude < -90 ||
          latitude > 90 ||
          longitude < -180 ||
          longitude > 180
        ) {

          toast.error(
            "Invalid map location"
          );

          return;
        }


        const normalized = {
          latitude:
            Number(
              latitude.toFixed(6)
            ),

          longitude:
            Number(
              longitude.toFixed(6)
            ),
        };


        setSelectedLocation(
          normalized
        );


        /*
         * Manual map selection is not
         * a camera selection.
         */
        setSelectedCamera(
          null
        );

      },
      []
    );


  // ==========================================================
  // MAPPED CAMERA COUNT
  // ==========================================================

  const mappedCameraCount =
    useMemo(
      () => {

        return cameras.filter(
          hasValidCoordinates
        ).length;

      },
      [
        cameras,
        hasValidCoordinates,
      ]
    );


  // ==========================================================
  // DASHBOARD METRICS
  // ==========================================================

  const totalCameras =
    cameras.length;


  const coveragePercent =
    totalCameras > 0
      ? Math.round(
        (
          mappedCameraCount /
          totalCameras
        ) *
        100
      )
      : 0;


  // ==========================================================
  // SYSTEM STATUS
  // ==========================================================

  const systemStatus =
    useMemo(() => {

      if (healthError) {
        return {
          label:
            "Health Unavailable",

          dotClass:
            "bg-red-400",

          textClass:
            "text-red-300",

          containerClass:
            "border-red-400/20 bg-red-400/5",
        };
      }


      if (isInitializing) {

        return {
          label:
            "System Initializing",

          dotClass:
            "animate-pulse bg-amber-400",

          textClass:
            "text-amber-300",

          containerClass:
            "border-amber-400/20 bg-amber-400/5",
        };
      }


      const overall =
        String(
          health?.overall || ""
        ).toUpperCase();


      if (
        overall ===
        "DEGRADED"
      ) {

        return {
          label:
            "System Degraded",

          dotClass:
            "bg-amber-400",

          textClass:
            "text-amber-300",

          containerClass:
            "border-amber-400/20 bg-amber-400/5",
        };
      }


      if (
        overall ===
        "UNHEALTHY" ||
        overall ===
        "CRITICAL" ||
        overall ===
        "DOWN"
      ) {

        return {
          label:
            "System Unhealthy",

          dotClass:
            "bg-red-400",

          textClass:
            "text-red-300",

          containerClass:
            "border-red-400/20 bg-red-400/5",
        };
      }


      /*
       * If no health response has arrived yet,
       * don't falsely claim the entire system is
       * operational.
       */
      if (!health) {

        return {
          label:
            "Checking System",

          dotClass:
            "animate-pulse bg-amber-400",

          textClass:
            "text-amber-300",

          containerClass:
            "border-amber-400/20 bg-amber-400/5",
        };
      }


      return {
        label:
          "System Operational",

        dotClass:
          "bg-emerald-400",

        textClass:
          "text-emerald-300",

        containerClass:
          "border-emerald-400/20 bg-emerald-400/5",
      };

    }, [
      health,
      healthError,
      isInitializing,
    ]);


  // ==========================================================
  // RENDER
  // ==========================================================

  return (
    <div className="min-h-[80vh] w-full rounded-2xl border border-slate-700/60 bg-[#07111f] p-3 text-white shadow-2xl shadow-black/20 sm:p-5">

      {/* ==================================================== */}
      {/* COMMAND HEADER */}
      {/* ==================================================== */}

      <section className="mb-5 overflow-hidden rounded-2xl border border-slate-700/60 bg-gradient-to-br from-[#0d1b2d] via-[#0b1727] to-[#07111f]">

        <div className="relative px-5 py-5 sm:px-6">

          <div className="absolute -right-20 -top-24 h-56 w-56 rounded-full bg-blue-500/10 blur-3xl" />

          <div className="absolute -bottom-24 left-1/3 h-48 w-48 rounded-full bg-cyan-500/5 blur-3xl" />


          <div className="relative flex flex-col gap-5 xl:flex-row xl:items-center xl:justify-between">

            <div>

              <div className="mb-2 flex items-center gap-2">

                <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400 shadow-[0_0_12px_rgba(52,211,153,0.8)]" />

                <span className="text-[11px] font-semibold uppercase tracking-[0.22em] text-emerald-400">
                  Operations Center
                </span>

              </div>


              <h1 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
                Security Intelligence Dashboard
              </h1>


              <p className="mt-1.5 max-w-2xl text-sm leading-6 text-slate-400">
                Monitor CCTV infrastructure, security alerts and geographic
                camera coverage from a single operational view.
              </p>

            </div>


            {/* ================================================= */}
            {/* HEADER ACTIONS */}
            {/* ================================================= */}

            <div className="flex flex-wrap items-center gap-2">

              {/* SYSTEM STATUS */}

              <div
                className={`flex items-center gap-2 rounded-xl border px-3.5 py-2.5 ${systemStatus.containerClass}`}
              >

                <span
                  className={`h-2 w-2 rounded-full ${systemStatus.dotClass}`}
                />

                <span
                  className={`text-xs font-semibold ${systemStatus.textClass}`}
                >
                  {systemStatus.label}
                </span>

              </div>


              {/* CAMERA REFRESH */}

              <button
                type="button"
                onClick={fetchCameras}
                disabled={loadingCameras}
                className="inline-flex items-center gap-2 rounded-xl border border-slate-600/70 bg-slate-800/70 px-3.5 py-2.5 text-xs font-semibold text-slate-200 transition hover:border-blue-400/50 hover:bg-slate-700/80 disabled:cursor-not-allowed disabled:opacity-60"
              >

                <svg
                  className={`h-4 w-4 ${loadingCameras
                      ? "animate-spin"
                      : ""
                    }`}
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                >

                  <path d="M20 11a8.1 8.1 0 0 0-14.9-4M4 5v4h4M4 13a8.1 8.1 0 0 0 14.9 4M20 19v-4h-4" />

                </svg>

                {
                  loadingCameras
                    ? "Refreshing"
                    : "Refresh"
                }

              </button>

            </div>

          </div>

        </div>


        {/* ==================================================== */}
        {/* KPI STRIP */}
        {/* ==================================================== */}

        <div className="grid grid-cols-2 border-t border-slate-700/60 lg:grid-cols-4">

          {/* TOTAL CAMERAS */}

          <div className="border-r border-slate-700/60 px-5 py-4">

            <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Total Cameras
            </p>

            <p className="mt-1 text-2xl font-bold text-white">
              {
                loadingCameras
                  ? "—"
                  : totalCameras
              }
            </p>

            <p className="mt-0.5 text-[11px] text-slate-500">
              Registered infrastructure
            </p>

          </div>


          {/* GIS COVERAGE */}

          <div className="border-r border-slate-700/60 px-5 py-4">

            <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              GIS Coverage
            </p>

            <p className="mt-1 text-2xl font-bold text-blue-400">
              {
                loadingCameras
                  ? "—"
                  : `${coveragePercent}%`
              }
            </p>

            <p className="mt-0.5 text-[11px] text-slate-500">
              {mappedCameraCount} cameras mapped
            </p>

          </div>


          {/* MAP STATUS */}

          <div className="border-r border-slate-700/60 px-5 py-4">

            <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Map Status
            </p>

            <div className="mt-2 flex items-center gap-2">

              <span className="h-2.5 w-2.5 rounded-full bg-emerald-400 shadow-[0_0_10px_rgba(52,211,153,0.55)]" />

              <span className="text-sm font-semibold text-emerald-300">
                Available
              </span>

            </div>

            <p className="mt-0.5 text-[11px] text-slate-500">
              Live location interface
            </p>

          </div>


          {/* DATA SYNC */}

          <div className="px-5 py-4">

            <p className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">
              Data Sync
            </p>

            <p className="mt-1 text-sm font-bold text-slate-200">
              Auto refresh
            </p>

            <p className="mt-0.5 text-[11px] text-slate-500">
              Camera data every 30 seconds
            </p>

          </div>

        </div>

      </section>


      {/* ==================================================== */}
      {/* SYSTEM HEALTH ACCESS */}
      {/* ==================================================== */}

      <div className="mb-5 flex items-center justify-end">

        <button
          type="button"
          onClick={openHealthModal}
          className="inline-flex items-center gap-2 rounded-xl border border-blue-400/20 bg-blue-400/5 px-3.5 py-2.5 text-xs font-semibold text-blue-200 transition hover:border-blue-400/40 hover:bg-blue-400/10 focus:outline-none focus:ring-2 focus:ring-blue-400/30"
          aria-label="Open System Health"
        >

          <span
            className={`h-2 w-2 rounded-full ${isInitializing
                ? "animate-pulse bg-amber-400"
                : healthError
                  ? "bg-red-400"
                  : health?.overall ===
                    "DEGRADED"
                    ? "bg-amber-400"
                    : "bg-emerald-400"
              }`}
          />

          System Health

        </button>

      </div>


      {/* ==================================================== */}
      {/* SYSTEM HEALTH MODAL */}
      {/* ==================================================== */}

      <SystemHealth
        isOpen={
          isHealthModalOpen
        }

        isInitializing={
          isInitializing
        }

        health={
          health
        }

        healthError={
          healthError
        }

        healthLoading={
          healthLoading
        }

        refreshing={
          healthRefreshing
        }

        /*
         * Startup countdown is intentionally
         * disabled.
         */
        countdown={
          null
        }

        startupAutoCloseActive={
          false
        }

        onClose={
          closeHealthModal
        }

        onRefresh={() =>
          fetchSystemHealth({
            startup: false,
            retry: true,
          })
        }
      />


      {/* ==================================================== */}
      {/* LIVE OPERATIONS */}
      {/* ==================================================== */}

      <div
        id="dashboard-camera-section"
        className="grid w-full grid-cols-1 gap-5 xl:grid-cols-4"
      >

        {/* ================================================== */}
        {/* CAMERA PANEL */}
        {/* ================================================== */}

        <section className="min-w-0 overflow-hidden rounded-2xl border border-slate-700/60 bg-[#0b1727] shadow-xl shadow-black/10 xl:col-span-3">

          <div className="flex flex-col gap-3 border-b border-slate-700/60 px-5 py-4 sm:flex-row sm:items-center sm:justify-between">

            <div className="flex items-center gap-3">

              <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-blue-400/20 bg-blue-400/10">

                <svg
                  className="h-5 w-5 text-blue-400"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                >

                  <path d="M3 8.5A2.5 2.5 0 0 1 5.5 6H14l3 3h2.5A1.5 1.5 0 0 1 21 10.5v7a1.5 1.5 0 0 1-1.5 1.5h-14A2.5 2.5 0 0 1 3 16.5v-8Z" />

                  <circle
                    cx="13"
                    cy="13.5"
                    r="3.2"
                  />

                </svg>

              </div>


              <div>

                <h2 className="text-base font-bold text-white">
                  Camera Operations
                </h2>

                <p className="text-xs text-slate-500">
                  Live camera feeds and operational controls
                </p>

              </div>

            </div>


            <span className="rounded-lg border border-slate-700 bg-slate-900/60 px-2.5 py-1.5 text-[11px] font-semibold text-slate-400">

              {
                loadingCameras
                  ? "Syncing..."
                  : `${totalCameras} cameras`
              }

            </span>

          </div>


          <div className="p-4 sm:p-5">

            <CameraList
              onCameraSelect={
                handleCameraSelect
              }

              onCameraMapFocus={
                handleCameraMapFocus
              }

              selectedCamera={
                selectedCamera
              }

              autoStartCameraId={
                autoStartCameraId
              }

              onAutoStartHandled={
                handleAutoStartHandled
              }
            />

          </div>

        </section>


        {/* ================================================== */}
        {/* ALERT PANEL */}
        {/* ================================================== */}

        <aside className="min-w-0 overflow-hidden rounded-2xl border border-slate-700/60 bg-[#0b1727] shadow-xl shadow-black/10">

          <div className="flex items-center justify-between border-b border-slate-700/60 px-5 py-4">

            <div className="flex items-center gap-3">

              <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-amber-400/20 bg-amber-400/10">

                <svg
                  className="h-5 w-5 text-amber-400"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                >

                  <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9Z" />

                  <path d="M10 21h4" />

                </svg>

              </div>


              <div>

                <h2 className="text-base font-bold text-white">
                  Security Alerts
                </h2>

                <p className="text-xs text-slate-500">
                  Recent events
                </p>

              </div>

            </div>


            <span className="rounded-full border border-amber-400/20 bg-amber-400/10 px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-amber-300">
              Live
            </span>

          </div>


          <div className="max-h-[680px] overflow-y-auto p-3">

            <Alert
              onAlertSelect={
                handleAlertSelect
              }
            />

          </div>

        </aside>

      </div>


      {/* ==================================================== */}
      {/* GIS INTELLIGENCE */}
      {/* ==================================================== */}

      <section
        ref={
          gisSectionRef
        }
        className="mt-5 overflow-hidden rounded-2xl border border-slate-700/60 bg-[#0b1727] shadow-xl shadow-black/10"
      >

        <div className="border-b border-slate-700/60 px-5 py-4 sm:px-6">

          <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">

            <div className="flex items-center gap-3">

              <div className="flex h-10 w-10 items-center justify-center rounded-xl border border-cyan-400/20 bg-cyan-400/10">

                <svg
                  className="h-5 w-5 text-cyan-400"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                >

                  <path d="M12 21s7-6.1 7-12A7 7 0 0 0 5 9c0 5.9 7 12 7 12Z" />

                  <circle
                    cx="12"
                    cy="9"
                    r="2.5"
                  />

                </svg>

              </div>


              <div>

                <div className="flex flex-wrap items-center gap-2">

                  <h2 className="text-base font-bold text-white sm:text-lg">
                    GIS Camera Intelligence
                  </h2>

                  <span className="rounded-md border border-cyan-400/20 bg-cyan-400/10 px-2 py-0.5 text-[9px] font-bold uppercase tracking-wider text-cyan-300">
                    Live Map
                  </span>

                </div>


                <p className="mt-0.5 text-xs text-slate-500">
                  Geographic camera coverage and security event locations
                </p>

              </div>

            </div>


            <div className="flex flex-wrap items-center gap-2">

              <div className="rounded-xl border border-slate-700 bg-slate-900/50 px-3.5 py-2">

                <p className="text-[9px] font-semibold uppercase tracking-wider text-slate-500">
                  Mapped Cameras
                </p>

                <p className="mt-0.5 text-lg font-bold text-blue-400">

                  {
                    loadingCameras
                      ? "..."
                      : mappedCameraCount
                  }

                  <span className="ml-1 text-xs font-medium text-slate-500">
                    / {totalCameras}
                  </span>

                </p>

              </div>


              {selectedLocation && (

                <div className="rounded-xl border border-blue-400/20 bg-blue-400/5 px-3.5 py-2">

                  <p className="text-[9px] font-semibold uppercase tracking-wider text-blue-300/70">
                    Selected Coordinates
                  </p>

                  <p className="mt-0.5 font-mono text-[11px] font-semibold text-slate-200">

                    {
                      selectedLocation.latitude.toFixed(
                        6
                      )
                    }

                    {" , "}

                    {
                      selectedLocation.longitude.toFixed(
                        6
                      )
                    }

                  </p>

                </div>

              )}

            </div>

          </div>

        </div>


        <div className="relative bg-[#07111f] p-3 sm:p-4">

          <div className="pointer-events-none absolute left-6 top-5 z-10 rounded-lg border border-slate-700/70 bg-[#07111f]/90 px-3 py-2 backdrop-blur-md">

            <p className="text-[9px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Coverage
            </p>

            <div className="mt-1 flex items-center gap-2">

              <span className="h-2 w-2 rounded-full bg-blue-400" />

              <span className="text-xs font-semibold text-slate-200">
                {coveragePercent}% mapped
              </span>

            </div>

          </div>


          <GISMap
            cameras={
              cameras
            }

            selectedCamera={
              selectedCamera
            }

            onCameraClick={
              handleCameraSelect
            }

            onMapClick={
              handleMapClick
            }
          />

        </div>
      </section>

    </div>
  );
};


export default Dashboard;