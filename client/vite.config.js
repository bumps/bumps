import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import generateFile from 'vite-plugin-generate-file'

// https://vitejs.dev/config/
export default ({mode}) => {
  return defineConfig({
    plugins: [
      vue(),
      generateFile([{
        type: 'yaml',
        output: 'VERSION',
        data: process.env.npm_package_version.toString(),
      }])
    ],
    base: '',
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url))
      }
    },
    define: {
      // By default, Vite doesn't include shims for NodeJS/
      // necessary for segment analytics lib to work
      global: {},
    },
    build: {
      rollupOptions: {
        output: {
          // Default
          // dir: 'dist',
          entryFileNames: (mode == 'production') ? 'assets/[name].js' : 'assets/[name].[hash].js',
          assetFileNames: (mode == 'production') ? 'assets/[name][extname]' : undefined,
          // chunkFileNames: "chunk-[name].js",
          // manualChunks: undefined,
        }
      }
    }
  })
}
