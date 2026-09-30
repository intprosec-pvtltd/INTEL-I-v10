import { useEffect, useState } from "react";
import {
  Link,
  NavLink,
  useLocation,
  useNavigate,
} from "react-router-dom";
import {
  Bell,
  Bot,
  Camera,
  ChevronRight,
  ClipboardList,
  Eye,
  FileBarChart,
  LayoutDashboard,
  LogOut,
  MapPin,
  Menu,
  ShieldCheck,
  UserRound,
  Users,
  X,
} from "lucide-react";
import toast from "react-hot-toast";
import api from "../api/axios.js";
import logo from "../assets/logo.webp";
import campLogo from "../assets/comp_logo.webp";
import {
  getStoredUser,
  hasPermission,
  isSuperAdmin,
} from "../auth/rbac";

const AUTH_EXPIRY_KEY = "login_expiry";

const PUBLIC_AUTH_PATHS = [
  "/",
  "/reset-password",
];

const ROLE_LABELS = {
  super_admin: "Super Administrator",
  rto_admin: "RTO Administrator",
  rto_staff: "RTO Staff",
  crime_admin: "Crime Administrator",
  crime_staff: "Crime Staff",
  cyber_admin: "Cyber Administrator",
  cyber_staff: "Cyber Staff",
};

const NAVIGATION_LINKS = [
  {
    name: "Dashboard",
    shortName: "Dashboard",
    path: "/dashboard",
    icon: LayoutDashboard,
  },
  {
    name: "Camera Setup",
    shortName: "Cameras",
    path: "/camera-setup",
    icon: Camera,
    permission: "cameras.view",
  },
  {
    name: "Vehicle Watchlist",
    shortName: "Vehicles",
    path: "/watchlist",
    icon: Eye,
    permission: "vehicle.view",
  },
  {
    name: "Person Watchlist",
    shortName: "Persons",
    path: "/person-watchlist",
    icon: UserRound,
    permission: "person.view",
  },
  {
    name: "Zone Editor",
    shortName: "Zones",
    path: "/zone-editor",
    icon: MapPin,
    permission: "zones.manage",
  },
  {
    name: "Alert History",
    shortName: "Alerts",
    path: "/alert-history",
    icon: Bell,
    permission: "alerts.view",
  },
  {
    name: "Incidents",
    shortName: "Incidents",
    path: "/incidents",
    icon: ClipboardList,
    permission: "incidents.view",
  },
  {
    name: "AI Intelligence",
    shortName: "AI Intelligence",
    path: "/intelligence-assistant",
    icon: Bot,
    permission: "assistant.use",
  },
  {
    name: "Analytics Reports",
    shortName: "Reports",
    path: "/analytics-reports",
    icon: FileBarChart,
    permission: "intelligence.view",
  },
  {
    name: "User Management",
    shortName: "Users",
    path: "/user-management",
    icon: Users,
    superAdminOnly: true,
  },
  {
    name: "Account Security",
    shortName: "Security",
    path: "/account-security",
    icon: ShieldCheck,
  },
];

const clearAuthStorage = () => {
  localStorage.removeItem("user");
  localStorage.removeItem("csrf_token");
  localStorage.removeItem(AUTH_EXPIRY_KEY);
};

const getInitials = (name) =>
  String(name || "User")
    .trim()
    .split(/\s+/)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("") || "U";

const Navbar = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const [mobileMenuOpen, setMobileMenuOpen] =
    useState(false);

  /*
   * Public authentication pages must not display
   * authenticated INTEL-I navigation.
   *
   * "/"                -> Login / Forgot Password
   * "/reset-password"  -> Email password-reset flow
   */
  const isPublicAuthPage =
    PUBLIC_AUTH_PATHS.includes(
      location.pathname,
    );

  const showNavigation =
    !isPublicAuthPage;

  const storedUser =
    getStoredUser();

  const displayName =
    storedUser?.full_name ||
    "Authorized User";

  const roleLabel =
    ROLE_LABELS[storedUser?.role] ||
    storedUser?.role ||
    "User";

  const navigationLinks =
    NAVIGATION_LINKS.filter(
      (link) => {
        if (link.superAdminOnly) {
          return isSuperAdmin(
            storedUser,
          );
        }

        return (
          !link.permission ||
          hasPermission(
            link.permission,
          )
        );
      },
    );

  useEffect(() => {
    document.body.style.overflow =
      mobileMenuOpen
        ? "hidden"
        : "";

    return () => {
      document.body.style.overflow =
        "";
    };
  }, [mobileMenuOpen]);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname]);

  /*
   * Defensive cleanup:
   * if a public auth route is opened while the
   * mobile menu was active, force it closed.
   */
  useEffect(() => {
    if (isPublicAuthPage) {
      setMobileMenuOpen(false);
    }
  }, [isPublicAuthPage]);

  const closeMobileMenu = () =>
    setMobileMenuOpen(false);

  const handleLogout = async () => {
    try {
      await api.post(
        "/auth/logout",
      );
    } catch (error) {
      console.warn(
        "Backend logout request failed:",
        error,
      );
    } finally {
      clearAuthStorage();
      closeMobileMenu();

      toast.success(
        "Logged out successfully",
      );

      navigate("/", {
        replace: true,
      });
    }
  };

  return (
    <>
      <nav className="sticky top-0 z-[100] w-full border-b border-slate-800 bg-[#01050a]/97 shadow-xl shadow-black/25 backdrop-blur-xl">
        <div className="mx-auto w-full max-w-[1920px] px-3 sm:px-4 lg:px-6">
          <div className="flex min-h-[68px] items-center justify-between gap-4 lg:min-h-[74px]">
            <Link
              to={
                showNavigation
                  ? "/dashboard"
                  : "/"
              }
              className="flex shrink-0 items-center transition-opacity hover:opacity-90"
              aria-label="INTEL-I home"
            >
              <img
                src={logo}
                alt="INTEL-I"
                className="h-[46px] w-[132px] object-contain sm:h-[52px] sm:w-[155px] lg:h-[57px] lg:w-[180px]"
              />
            </Link>

            <div className="flex min-w-0 items-center gap-2 sm:gap-3">
              {showNavigation &&
                storedUser && (
                  <div className="hidden min-w-0 items-center gap-3 rounded-xl border border-slate-800 bg-[#07111d] px-3 py-2 lg:flex">
                    <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg border border-blue-500/25 bg-blue-500/10 text-xs font-bold text-blue-300">
                      {getInitials(
                        displayName,
                      )}
                    </div>

                    <div className="min-w-0 max-w-44">
                      <p className="truncate text-sm font-semibold text-white">
                        {displayName}
                      </p>

                      <p className="truncate text-[11px] text-slate-500">
                        {roleLabel}
                      </p>
                    </div>

                    {isSuperAdmin(
                      storedUser,
                    ) && (
                      <ShieldCheck
                        size={17}
                        className="shrink-0 text-cyan-400"
                        aria-label="Super Administrator"
                      />
                    )}
                  </div>
                )}

              <img
                src={campLogo}
                alt="IntProSec"
                className="hidden h-[44px] w-[128px] object-contain sm:block lg:h-[50px] lg:w-[155px] xl:w-[175px]"
              />

              {showNavigation && (
                <>
                  <button
                    type="button"
                    onClick={
                      handleLogout
                    }
                    title="Logout"
                    aria-label="Logout"
                    className="hidden h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-blue-500/30 bg-blue-600 text-white transition hover:bg-blue-500 hover:shadow-lg hover:shadow-blue-600/20 xl:flex"
                  >
                    <LogOut
                      size={18}
                    />
                  </button>

                  <button
                    type="button"
                    aria-label={
                      mobileMenuOpen
                        ? "Close navigation menu"
                        : "Open navigation menu"
                    }
                    aria-expanded={
                      mobileMenuOpen
                    }
                    onClick={() =>
                      setMobileMenuOpen(
                        (previous) =>
                          !previous,
                      )
                    }
                    className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-slate-700 bg-slate-900 text-slate-300 transition hover:border-blue-500/50 hover:bg-[#10233a] hover:text-white xl:hidden"
                  >
                    {mobileMenuOpen ? (
                      <X size={22} />
                    ) : (
                      <Menu
                        size={22}
                      />
                    )}
                  </button>
                </>
              )}
            </div>
          </div>

          {showNavigation && (
            <div className="hidden border-t border-slate-800/80 xl:block">
              <div className="flex min-w-0 items-center gap-1.5 overflow-x-auto py-2 [scrollbar-width:none] [&::-webkit-scrollbar]:hidden">
                {navigationLinks.map(
                  (item) => {
                    const Icon =
                      item.icon;

                    return (
                      <NavLink
                        key={
                          item.path
                        }
                        to={item.path}
                        title={
                          item.name
                        }
                        className={({
                          isActive,
                        }) =>
                          `group flex min-w-0 flex-1 items-center justify-center gap-2 whitespace-nowrap rounded-lg border px-2.5 py-2.5 text-xs font-semibold transition-all duration-200 2xl:text-sm ${
                            isActive
                              ? "border-blue-500/40 bg-blue-600 text-white shadow-lg shadow-blue-600/15"
                              : "border-transparent text-slate-400 hover:border-slate-800 hover:bg-[#0b1929] hover:text-white"
                          }`
                        }
                      >
                        <Icon
                          size={17}
                          strokeWidth={
                            2
                          }
                          className="shrink-0 transition-transform group-hover:scale-105"
                        />

                        <span className="2xl:hidden">
                          {
                            item.shortName
                          }
                        </span>

                        <span className="hidden 2xl:inline">
                          {
                            item.name
                          }
                        </span>
                      </NavLink>
                    );
                  },
                )}
              </div>
            </div>
          )}
        </div>

        {showNavigation &&
          mobileMenuOpen && (
            <div className="absolute right-0 left-0 top-full z-[110] max-h-[calc(100vh-68px)] overflow-y-auto border-t border-slate-800 bg-[#01050a] shadow-2xl shadow-black/50 xl:hidden">
              <div className="mx-auto w-full max-w-[1920px] p-3 sm:p-4">
                <div className="mb-3 flex items-center justify-between gap-3 rounded-xl border border-slate-800 bg-[#07111d] p-3.5">
                  <div className="flex min-w-0 items-center gap-3">
                    <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-blue-500/25 bg-blue-500/10 text-xs font-bold text-blue-300">
                      {getInitials(
                        displayName,
                      )}
                    </div>

                    <div className="min-w-0">
                      <p className="truncate text-sm font-semibold text-white">
                        {
                          displayName
                        }
                      </p>

                      <p className="truncate text-xs text-slate-500">
                        {
                          roleLabel
                        }
                      </p>
                    </div>
                  </div>

                  <div className="flex shrink-0 items-center gap-2 text-xs font-semibold text-emerald-400">
                    <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-400" />
                    Online
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {navigationLinks.map(
                    (item) => {
                      const Icon =
                        item.icon;

                      return (
                        <NavLink
                          key={
                            item.path
                          }
                          to={
                            item.path
                          }
                          onClick={
                            closeMobileMenu
                          }
                          className={({
                            isActive,
                          }) =>
                            `flex items-center gap-3 rounded-xl border px-4 py-3 text-sm font-semibold transition-all ${
                              isActive
                                ? "border-blue-500/40 bg-blue-600 text-white shadow-lg shadow-blue-600/10"
                                : "border-slate-800 bg-[#07111d] text-slate-300 hover:border-slate-700 hover:bg-[#10233a] hover:text-white"
                            }`
                          }
                        >
                          <Icon
                            size={19}
                            className="shrink-0"
                          />

                          <span className="min-w-0 flex-1 truncate">
                            {
                              item.name
                            }
                          </span>

                          <ChevronRight
                            size={16}
                            className="shrink-0 opacity-50"
                          />
                        </NavLink>
                      );
                    },
                  )}
                </div>

                <button
                  type="button"
                  onClick={
                    handleLogout
                  }
                  className="mt-3 flex w-full items-center justify-center gap-2 rounded-xl border border-red-500/25 bg-red-500/5 px-4 py-3 text-sm font-semibold text-red-400 transition hover:border-red-500/45 hover:bg-red-500/10 hover:text-red-300"
                >
                  <LogOut
                    size={18}
                  />
                  Logout
                </button>
              </div>
            </div>
          )}
      </nav>

      {showNavigation &&
        mobileMenuOpen && (
          <button
            type="button"
            aria-label="Close navigation menu"
            onClick={
              closeMobileMenu
            }
            className="fixed inset-0 z-[90] cursor-default bg-black/55 backdrop-blur-[2px] xl:hidden"
          />
        )}
    </>
  );
};

export default Navbar;
