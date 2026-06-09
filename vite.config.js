import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Standard Vite config. Output goes to /dist, which Firebase Hosting serves.
export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
  },
});
