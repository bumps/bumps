import { defineConfig } from "vite";
import svgLoader from "vite-svg-loader";
import vue from "@vitejs/plugin-vue";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [vue(), svgLoader()],
  base: "",
  define: {
    // By default, Vite doesn't include shims for NodeJS.
    // Plotly fails to load without this shim.
    global: {},
  },
});
