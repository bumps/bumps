import { join } from "node:path";
import { fileURLToPath, URL } from "node:url";
import { defineConfig } from "vite";
import svgLoader from "vite-svg-loader";
import vue from "@vitejs/plugin-vue";

// https://vitejs.dev/config/
export default ({ mode }) => {
  return defineConfig({
    plugins: [vue(), svgLoader()],
    base: "",
    resolve: {
      alias: {
        "@": fileURLToPath(new URL("./src", import.meta.url)),
      },
    },
    define: {
      // By default, Vite doesn't include shims for NodeJS/
      // necessary for segment analytics lib to work
      global: {},
    },
    worker: {
      format: "es",
      rollupOptions: {
        external: ["node-fetch"],
      },
    },
    build: {
      rollupOptions: {
        output: {
          // Default
          dir: join("dist", process.env.npm_package_version),
          entryFileNames: mode == "production" ? "assets/[name].js" : "assets/[name].[hash].js",
          assetFileNames: mode == "production" ? "assets/[name][extname]" : undefined,
          // chunkFileNames: "chunk-[name].js",
          // manualChunks: undefined,
        },
      },
    },
  });
};
