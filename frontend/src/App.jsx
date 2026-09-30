import React from "react";

import {
  BrowserRouter,
} from "react-router-dom";

import {
  Toaster,
} from "react-hot-toast";

import AppContent from "./pages/Appcontent";

import {
  NavigationStackProvider,
} from "./context/NavigationStackContext";

export const API_URL =
  import.meta.env.VITE_BASE_URL;

/* =========================================================
   ROOT APPLICATION
========================================================= */

const App = () => {
  return (
    <BrowserRouter>
      {/*
       * NavigationStackProvider MUST stay
       * inside BrowserRouter because it uses
       * useNavigate() and useLocation().
       */}
      <NavigationStackProvider>
        <div className="min-h-screen bg-[var(--bg)] text-[var(--txt)]">
          <AppContent />

          <Toaster
            position="top-right"
            toastOptions={{
              style: {
                fontSize:
                  "13px",
              },
            }}
          />
        </div>
      </NavigationStackProvider>
    </BrowserRouter>
  );
};

export default App;