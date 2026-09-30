import React from "react";
import { Navigate } from "react-router-dom";

const AUTH_EXPIRY_KEY = "login_expiry";

const clearAuthStorage = () => {
  localStorage.removeItem("user");
  localStorage.removeItem("csrf_token");
  localStorage.removeItem(AUTH_EXPIRY_KEY);
};

const ProtectedRoute = ({ children }) => {
  const user = localStorage.getItem("user");
  const loginExpiry = Number(localStorage.getItem(AUTH_EXPIRY_KEY));

  if (!user || !loginExpiry || Date.now() >= loginExpiry) {
    clearAuthStorage();
    return <Navigate to="/" replace />;
  }

  return children;
};

export default ProtectedRoute;