import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

/*
 * ============================================================
 * INTEL-I
 * LOCAL FRONTEND -> LOCAL BACKEND
 * ============================================================
 *
 * Frontend:
 *   http://localhost:5173
 *
 * Local Backend:
 *   http://127.0.0.1:3000
 *
 * Browser
 *    |
 *    v
 * Vite localhost:5173
 *    |
 *    | HTTP / WebSocket proxy
 *    v
 * FastAPI 127.0.0.1:3000
 *
 * IMPORTANT:
 * Do not proxy "/camera" broadly because it can interfere
 * with the React route "/camera-setup".
 * ============================================================
 */

const BACKEND_URL = "http://127.0.0.1:3000";

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
      "^/ws(?:/|$)": {
        ...proxyTarget,
        ws: true,
      },

      "^/auth(?:/|$)": {
        ...proxyTarget,
      },

      "^/api(?:/|$)": {
        ...proxyTarget,
      },

      "^/camera(?:/|$)": {
        ...proxyTarget,
      },

      "^/cameras(?:/|$)": {
        ...proxyTarget,
      },

      "^/stream(?:/|$)": {
        ...proxyTarget,
      },

      "^/stream-session(?:/|$)": {
        ...proxyTarget,
      },

      "^/stream-upload(?:/|$)": {
        ...proxyTarget,
      },

      "^/video-upload(?:/|$)": {
        ...proxyTarget,
      },

      "^/system(?:/|$)": {
        ...proxyTarget,
      },

      "^/alerts(?:/|$)": {
        ...proxyTarget,
      },

      "^/snapshot(?:/|$)": {
        ...proxyTarget,
      },
    },
  },
});