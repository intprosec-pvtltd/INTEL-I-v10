export const getStoredUser = () => {
  try { return JSON.parse(localStorage.getItem("user") || "null"); }
  catch { return null; }
};

export const hasPermission = (permission) => {
  const permissions = getStoredUser()?.permissions || [];
  return permissions.includes("*") || permissions.includes(permission);
};

export const isSuperAdmin = (user = getStoredUser()) =>
  user?.role === "super_admin";
