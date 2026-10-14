import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run build` emits straight into openlab/app/static/, which the workspace
// server serves (see server.py: STATIC / index.html, else the placeholder).
//
// In dev, vite serves the SPA and proxies the API and both sockets to the
// FastAPI server on 8321, so `npm run dev` + `openlab app` is the whole loop.
export default defineConfig({
  plugins: [react()],
  // relative asset URLs: the bundle is served from the app root but must not
  // assume it. No CDN, no network font — everything ships in the bundle.
  base: "./",
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:8321", changeOrigin: false },
      "/ws": { target: "ws://127.0.0.1:8321", ws: true },
    },
  },
  build: {
    outDir: "../static",
    emptyOutDir: true,
    chunkSizeWarningLimit: 1200,
    rollupOptions: {
      output: {
        // the editor and the terminal are the heavy dependencies; split so the
        // shell paints first.
        manualChunks: {
          editor: ["@codemirror/view", "@codemirror/state", "@codemirror/commands", "@codemirror/search"],
          terminal: ["@xterm/xterm", "@xterm/addon-fit"],
        },
      },
    },
  },
});
