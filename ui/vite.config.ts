/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// VITE_CONNECTED controls whether the MCP settings pane is included in the
// bundle.  When unset (e.g. free-artifact builds pass VITE_CONNECTED= to vite),
// the McpSettingsPane import is never reached and tree-shaking drops the pane
// + mcp-settings/mcp-server-info state modules from the output bundle.
//
// The default is "1" (connected tier) so `npm run dev`, production builds, and
// vitest all keep current behaviour without any explicit env-var.
//
// To verify the free-build strip: VITE_CONNECTED= npm run build followed by
// grep -r "Enable MCP server" dist/ — the string must be absent.
const VITE_CONNECTED =
  process.env.VITE_CONNECTED !== undefined ? process.env.VITE_CONNECTED : "1";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
  },
  envPrefix: ["VITE_", "TAURI_"],
  define: {
    // Statically replace the flag so Vite's tree-shaker can eliminate dead
    // branches.  A JSON.stringify is required: Vite's define replaces the token
    // literally in source text, so the value must be a valid JS expression.
    "import.meta.env.VITE_CONNECTED": JSON.stringify(VITE_CONNECTED),
  },
  build: {
    target: "es2022",
    // Tauri uses Chromium on Windows and WebKit on macOS and Linux
    minify: !process.env.TAURI_DEBUG ? "esbuild" : false,
    sourcemap: !!process.env.TAURI_DEBUG,
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/__tests__/setup.ts"],
  },
});
