import { useEffect, useState } from "react";
import { Navigate } from "react-router-dom";
import api, { clearAuthStorage } from "../api/axios";
import { isSuperAdmin } from "../auth/rbac";
import Loading from "./Loading";
import ProtectedRoute from "./ProtectedRoute";

const AUTH_EXPIRY_KEY = "login_expiry";

export default function SuperAdminRoute({ children }) {
  const [state, setState] = useState("checking");

  useEffect(() => {
    let active = true;

    api
      .get("/auth/me")
      .then((response) => {
        if (!active) return;

        if (response.data?.id) {
          localStorage.setItem("user", JSON.stringify(response.data));
        }

        setState(isSuperAdmin(response.data) ? "allowed" : "denied");
      })
      .catch(() => {
        if (!active) return;
        clearAuthStorage();
        localStorage.removeItem(AUTH_EXPIRY_KEY);
        setState("unauthenticated");
      });

    return () => {
      active = false;
    };
  }, []);

  return (
    <ProtectedRoute>
      {state === "checking" && <Loading />}
      {state === "allowed" && children}
      {state === "denied" && <Navigate to="/dashboard" replace />}
      {state === "unauthenticated" && <Navigate to="/" replace />}
    </ProtectedRoute>
  );
}
