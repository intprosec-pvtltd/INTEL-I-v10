import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useRef,
} from "react";

import {
  useLocation,
  useNavigate,
} from "react-router-dom";

const NavigationStackContext =
  createContext(null);

const NAVIGATION_STACK_KEY =
  "intel_i_navigation_stack_v1";

const PAGE_STATE_KEY =
  "intel_i_navigation_page_state_v1";

const MAX_STACK_SIZE = 40;

/* =========================================================
   SAFE SESSION STORAGE
========================================================= */

const readJson = (
  key,
  fallback,
) => {
  try {
    const raw =
      sessionStorage.getItem(key);

    if (!raw) {
      return fallback;
    }

    return JSON.parse(raw);
  } catch (error) {
    console.error(
      `Failed to read ${key}:`,
      error,
    );

    return fallback;
  }
};

const writeJson = (
  key,
  value,
) => {
  try {
    sessionStorage.setItem(
      key,
      JSON.stringify(value),
    );
  } catch (error) {
    console.error(
      `Failed to write ${key}:`,
      error,
    );
  }
};

/* =========================================================
   NORMALIZE ROUTE
========================================================= */

const normalizeRoute = (
  location,
) => {
  return `${location.pathname || "/"}${
    location.search || ""
  }${location.hash || ""}`;
};

/* =========================================================
   PROVIDER
========================================================= */

export const NavigationStackProvider = ({
  children,
}) => {
  const navigate =
    useNavigate();

  const location =
    useLocation();

  const stackRef =
    useRef(
      readJson(
        NAVIGATION_STACK_KEY,
        [],
      ),
    );

  /* =======================================================
     SAVE STACK
  ======================================================= */

  const persistStack =
    useCallback(() => {
      writeJson(
        NAVIGATION_STACK_KEY,
        stackRef.current.slice(
          -MAX_STACK_SIZE,
        ),
      );
    }, []);

  /* =======================================================
     SAVE PAGE STATE
  ======================================================= */

  const savePageState =
    useCallback(
      (
        route,
        state,
      ) => {
        if (!route) {
          return;
        }

        const existing =
          readJson(
            PAGE_STATE_KEY,
            {},
          );

        existing[route] = {
          ...(existing[route] ||
            {}),
          ...(state || {}),
          savedAt:
            Date.now(),
        };

        writeJson(
          PAGE_STATE_KEY,
          existing,
        );
      },
      [],
    );

  /* =======================================================
     READ PAGE STATE
  ======================================================= */

  const getPageState =
    useCallback(
      (route) => {
        if (!route) {
          return null;
        }

        const existing =
          readJson(
            PAGE_STATE_KEY,
            {},
          );

        return (
          existing[route] ||
          null
        );
      },
      [],
    );

  /* =======================================================
     REMOVE PAGE STATE
  ======================================================= */

  const removePageState =
    useCallback(
      (route) => {
        if (!route) {
          return;
        }

        const existing =
          readJson(
            PAGE_STATE_KEY,
            {},
          );

        delete existing[
          route
        ];

        writeJson(
          PAGE_STATE_KEY,
          existing,
        );
      },
      [],
    );

  /* =======================================================
     PUSH ROUTE
  ======================================================= */

  const pushRoute =
    useCallback(
      (
        target,
        options = {},
      ) => {
        if (!target) {
          return;
        }

        const currentRoute =
          normalizeRoute(
            location,
          );

        const previousEntry =
          stackRef.current[
            stackRef.current
              .length - 1
          ];

        /*
         * Do not push duplicate routes
         * repeatedly onto our custom stack.
         */
        if (
          !previousEntry ||
          previousEntry.route !==
            currentRoute
        ) {
          stackRef.current = [
            ...stackRef.current,
            {
              route:
                currentRoute,

              routerState:
                location.state ||
                null,

              returnState:
                options.returnState ||
                null,

              scrollY:
                window.scrollY ||
                0,

              createdAt:
                Date.now(),
            },
          ].slice(
            -MAX_STACK_SIZE,
          );

          persistStack();
        }

        navigate(
          target,
          {
            replace:
              options.replace ===
              true,

            state: {
              ...(options.state ||
                {}),

              intelINavigation:
                true,

              intelIFrom:
                currentRoute,

              intelINavigationTime:
                Date.now(),
            },
          },
        );
      },
      [
        location,
        navigate,
        persistStack,
      ],
    );

  /* =======================================================
     CUSTOM BACK
  ======================================================= */

  const goBack =
    useCallback(
      (
        fallback =
          "/dashboard",
      ) => {
        const stack = [
          ...stackRef.current,
        ];

        const previous =
          stack.pop();

        stackRef.current =
          stack;

        persistStack();

        if (
          previous?.route
        ) {
          navigate(
            previous.route,
            {
              /*
               * Replace current destination instead
               * of creating another history loop.
               */
              replace: true,

              state: {
                ...(previous.routerState ||
                  {}),

                ...(previous.returnState ||
                  {}),

                intelIRestored:
                  true,

                intelIRestoredAt:
                  Date.now(),
              },
            },
          );

          window.requestAnimationFrame(
            () => {
              window.scrollTo({
                top:
                  Number(
                    previous.scrollY,
                  ) || 0,

                behavior:
                  "auto",
              });
            },
          );

          return;
        }

        navigate(
          fallback,
          {
            replace: true,

            state: {
              intelIRestored:
                true,
            },
          },
        );
      },
      [
        navigate,
        persistStack,
      ],
    );

  /* =======================================================
     REPLACE CURRENT ROUTE
  ======================================================= */

  const replaceRoute =
    useCallback(
      (
        target,
        state = null,
      ) => {
        navigate(
          target,
          {
            replace: true,
            state,
          },
        );
      },
      [navigate],
    );

  /* =======================================================
     CLEAR NAVIGATION STACK
  ======================================================= */

  const clearStack =
    useCallback(() => {
      stackRef.current = [];

      try {
        sessionStorage.removeItem(
          NAVIGATION_STACK_KEY,
        );

        sessionStorage.removeItem(
          PAGE_STATE_KEY,
        );
      } catch (error) {
        console.error(
          "Failed to clear INTEL-I navigation state:",
          error,
        );
      }
    }, []);

  /* =======================================================
     STACK INFO
  ======================================================= */

  const getStack =
    useCallback(() => {
      return [
        ...stackRef.current,
      ];
    }, []);

  const canGoBack =
    useCallback(() => {
      return (
        stackRef.current
          .length > 0
      );
    }, []);

  /* =======================================================
     CONTEXT
  ======================================================= */

  const value =
    useMemo(
      () => ({
        pushRoute,
        goBack,
        replaceRoute,

        savePageState,
        getPageState,
        removePageState,

        clearStack,
        getStack,
        canGoBack,
      }),
      [
        pushRoute,
        goBack,
        replaceRoute,
        savePageState,
        getPageState,
        removePageState,
        clearStack,
        getStack,
        canGoBack,
      ],
    );

  return (
    <NavigationStackContext.Provider
      value={value}
    >
      {children}
    </NavigationStackContext.Provider>
  );
};

/* =========================================================
   HOOK
========================================================= */

export const useNavigationStack =
  () => {
    const context =
      useContext(
        NavigationStackContext,
      );

    if (!context) {
      throw new Error(
        "useNavigationStack must be used inside NavigationStackProvider",
      );
    }

    return context;
  };

export default NavigationStackContext;