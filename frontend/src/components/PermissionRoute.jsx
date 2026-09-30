import { Navigate } from "react-router-dom";
import ProtectedRoute from "./ProtectedRoute";
import { hasPermission } from "../auth/rbac";

export default function PermissionRoute({ permission, children }) {
  return <ProtectedRoute>{hasPermission(permission) ? children : <Navigate to="/dashboard" replace />}</ProtectedRoute>;
}
