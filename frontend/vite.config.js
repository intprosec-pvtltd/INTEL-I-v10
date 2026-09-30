import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/*
 * ============================================================
 * INTEL-I
 * LOCAL FRONTEND -> AWS L40S BACKEND
 * ============================================================
 *
 * Frontend:
 *   http://localhost:5173
 *
 * AWS Backend:
 *   http://52.78.151.3
 *
 * Browser
 *    |
 *    v
 * Vite localhost:5173
 *    |
 *    | HTTP / WebSocket proxy
 *    v
 * AWS Nginx :80
 *    |
 *    v
 * FastAPI 127.0.0.1:8000
 *
 * IMPORTANT:
 * Do not proxy "/camera" broadly because it can interfere
 * with the React route "/camera-setup".
 * ============================================================
 */

const BACKEND_URL = "http://16.184.9.123";

const proxyTarget = {
  target: BACKEND_URL,
  changeOrigin: true,
  secure: false,
};

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
  ],

  server: {
    host: "localhost",
    port: 5173,
    strictPort: true,

    proxy: {
      /*
       * --------------------------------------------------------
       * WebSocket
       * --------------------------------------------------------
       */
      "^/ws(?:/|$)": {
        ...proxyTarget,
        ws: true,
      },

      /*
       * --------------------------------------------------------
       * Authentication
       * --------------------------------------------------------
       */
      "^/auth(?:/|$)": {
        ...proxyTarget,
      },

      /*
       * --------------------------------------------------------
       * Main API
       * --------------------------------------------------------
       */
      "^/api(?:/|$)": {
        ...proxyTarget,
      },

      /*
       * --------------------------------------------------------
       * Cameras
       *
       * This regex matches:
       *   /camera
       *   /camera/...
       *
       * It DOES NOT match:
       *   /camera-setup
       *
       * Therefore the React camera-setup route remains safe.
       * --------------------------------------------------------
       */
      "^/camera(?:/|$)": {
        ...proxyTarget,
      },

      "^/cameras(?:/|$)": {
        ...proxyTarget,
      },

      /*
       * --------------------------------------------------------
       * Streaming
       * --------------------------------------------------------
       */
      "^/stream(?:/|$)": {
        ...proxyTarget,
      },

      "^/stream-session(?:/|$)": {
        ...proxyTarget,
      },

      "^/stream-upload(?:/|$)": {
        ...proxyTarget,
      },

      /*
       * --------------------------------------------------------
       * Video Upload
       * --------------------------------------------------------
       */
      "^/video-upload(?:/|$)": {
        ...proxyTarget,
      },

      /*
       * --------------------------------------------------------
       * System
       * --------------------------------------------------------
       */
      "^/system(?:/|$)": {
        ...proxyTarget,
      },

      /*
       * --------------------------------------------------------
       * Alerts
       * --------------------------------------------------------
       */
      "^/alerts(?:/|$)": {
        ...proxyTarget,
      },

      /*
       * --------------------------------------------------------
       * Evidence / Snapshots
       * --------------------------------------------------------
       */
      "^/snapshot(?:/|$)": {
        ...proxyTarget,
      },
    },
  },
});