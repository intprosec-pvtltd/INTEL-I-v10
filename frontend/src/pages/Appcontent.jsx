import {
  useCallback,
  useEffect,
  useState,
} from "react";

import {
  Routes,
  Route,
  useLocation,
} from "react-router-dom";

import Navbar from "../components/Navbar";
import ProtectedRoute from "../components/ProtectedRoute";
import SuperAdminRoute from "../components/SuperAdminRoute";
import PermissionRoute from "../components/PermissionRoute";
import SystemStartupOverlay from "../components/SystemStartupOverlay";

import {
  useNavigationStack,
} from "../context/NavigationStackContext";

import Login from "../pages/Login";
import ResetPassword from "../pages/ResetPassword";
import Dashboard from "../pages/Dashboard";
import CameraSetup from "../pages/CameraSetup";
import RulesEngine from "../pages/RulesEngine";
import ZoneEditor from "../pages/ZoneEditor";
import AlertHistory from "../pages/AlertHistory";
import Watchlist from "../pages/Watchlist";
import PersonWatchlist from "../pages/PersonWatchlist";
import IncidentCenter from "../pages/IncidentCenter";
import IntelligenceAssistant from "../pages/IntelligenceAssistant";
import AnalyticsReports from "../pages/AnalyticsReports";
import UserManagement from "../pages/UserManagement";
import AccountSecurity from "../components/AccountSecurity";

/* =========================================================
   PUBLIC ROUTES
========================================================= */

const PUBLIC_AUTH_PATHS = [
  "/",
  "/reset-password",
];

/* =========================================================
   STARTUP FLAG
========================================================= */

const STARTUP_FLAG_KEY =
  "intel_i_show_startup";

/* =========================================================
   APP CONTENT
========================================================= */

const AppContent = () => {
  const location =
    useLocation();

  const {
    clearStack,
  } =
    useNavigationStack();

  const [
    showStartup,
    setShowStartup,
  ] = useState(false);

  const isPublicAuthPage =
    PUBLIC_AUTH_PATHS.includes(
      location.pathname,
    );

  /* =======================================================
     CLEAR OLD NAVIGATION STATE WHEN USER IS ON AUTH PAGE
  ======================================================= */

  useEffect(() => {
    if (
      location.pathname ===
      "/"
    ) {
      clearStack();
    }
  }, [
    location.pathname,
    clearStack,
  ]);

  /* =======================================================
     POST LOGIN STARTUP ANIMATION
  ======================================================= */

  useEffect(() => {
    if (
      isPublicAuthPage
    ) {
      return;
    }

    const shouldShowStartup =
      sessionStorage.getItem(
        STARTUP_FLAG_KEY,
      ) === "true";

    if (
      !shouldShowStartup
    ) {
      return;
    }

    /*
     * Consume flag immediately.
     *
     * Prevents startup animation from
     * appearing again on refresh.
     */
    sessionStorage.removeItem(
      STARTUP_FLAG_KEY,
    );

    setShowStartup(true);
  }, [
    location.pathname,
    isPublicAuthPage,
  ]);

  /* =======================================================
     STARTUP COMPLETE
  ======================================================= */

  const handleStartupComplete =
    useCallback(() => {
      setShowStartup(false);
    }, []);

  /* =======================================================
     UI
  ======================================================= */

  return (
    <div className="min-h-screen bg-[var(--bg)] text-[var(--txt)]">
      {/* =====================================================
          INTEL-I POST LOGIN INITIALIZATION
      ====================================================== */}

      {showStartup && (
        <SystemStartupOverlay
          onComplete={
            handleStartupComplete
          }
        />
      )}

      {/* =====================================================
          GLOBAL NAVBAR
      ====================================================== */}

      <Navbar />

      {/* =====================================================
          APPLICATION
      ====================================================== */}

      <div className="flex">
        <main
          className={
            isPublicAuthPage
              ? `
                min-h-[calc(100vh-85px)]
                w-full
                flex
                items-center
                justify-center
              `
              : `
                min-h-[calc(100vh-85px)]
                flex-1
                overflow-y-auto
                p-4
              `
          }
        >
          <Routes>
            {/* =================================================
                PUBLIC AUTHENTICATION
            ================================================= */}

            <Route
              path="/"
              element={
                <Login />
              }
            />

            <Route
              path="/reset-password"
              element={
                <ResetPassword />
              }
            />

            {/* =================================================
                DASHBOARD
            ================================================= */}

            <Route
              path="/dashboard"
              element={
                <ProtectedRoute>
                  <Dashboard />
                </ProtectedRoute>
              }
            />

            {/* =================================================
                CAMERA
            ================================================= */}

            <Route
              path="/camera-setup"
              element={
                <ProtectedRoute>
                  <CameraSetup />
                </ProtectedRoute>
              }
            />

            {/* =================================================
                RULE ENGINE
            ================================================= */}

            <Route
              path="/rules-engine"
              element={
                <ProtectedRoute>
                  <RulesEngine />
                </ProtectedRoute>
              }
            />

            {/* =================================================
                GIS / ZONES
            ================================================= */}

            <Route
              path="/zone-editor"
              element={
                <ProtectedRoute>
                  <ZoneEditor />
                </ProtectedRoute>
              }
            />

            {/* =================================================
                ALERT HISTORY
            ================================================= */}

            <Route
              path="/alert-history"
              element={
                <ProtectedRoute>
                  <AlertHistory />
                </ProtectedRoute>
              }
            />

            {/* =================================================
                VEHICLE WATCHLIST
            ================================================= */}

            <Route
              path="/watchlist"
              element={
                <ProtectedRoute>
                  <Watchlist />
                </ProtectedRoute>
              }
            />

            {/* =================================================
                PERSON WATCHLIST
            ================================================= */}

            <Route
              path="/person-watchlist"
              element={
                <ProtectedRoute>
                  <PersonWatchlist />
                </ProtectedRoute>
              }
            />

            {/* =================================================
                INCIDENT CENTER
            ================================================= */}

            <Route
              path="/incidents"
              element={
                <ProtectedRoute>
                  <IncidentCenter />
                </ProtectedRoute>
              }
            />



            {/* =================================================
                AI INTELLIGENCE
            ================================================= */}

            <Route
              path="/intelligence-assistant"
              element={
                <ProtectedRoute>
                  <IntelligenceAssistant />
                </ProtectedRoute>
              }
            />

            {/* =================================================
                ANALYTICS REPORTS
            ================================================= */}

            <Route
              path="/analytics-reports"
              element={
                <PermissionRoute permission="intelligence.view">
                  <AnalyticsReports />
                </PermissionRoute>
              }
            />

            {/* =================================================
                USER MANAGEMENT
            ================================================= */}

            <Route
              path="/user-management"
              element={
                <SuperAdminRoute>
                  <UserManagement />
                </SuperAdminRoute>
              }
            />

            {/* =================================================
                ACCOUNT SECURITY
            ================================================= */}

            <Route
              path="/account-security"
              element={
                <ProtectedRoute>
                  <AccountSecurity />
                </ProtectedRoute>
              }
            />
          </Routes>
        </main>
      </div>
    </div>
  );
};

export default AppContent;