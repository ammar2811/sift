// vitest/config re-exports defineConfig with the `test` block typed; importing from
// "vite" leaves it unknown and fails the build.
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // The API is same-origin in production behind the container's reverse proxy, so
    // proxying in dev keeps request paths identical across environments.
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/ready": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test-setup.ts"],
  },
});
