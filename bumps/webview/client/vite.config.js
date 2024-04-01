import { fileURLToPath, URL } from 'node:url'
import { join } from 'node:path'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default ({mode}) => {
  return defineConfig({
    plugins: [
      vue(),
    ],
    base: '',
    resolve: {
      alias: {
        '@': fileURLToPath(new URL('./src', import.meta.url)),
        './asyncSocket': (mode == 'standalone') ? join(process.cwd(), 'src', 'asyncWorkerSocket') : './asyncSocket',
        'socket.io-client': (mode == 'standalone') ? join(process.cwd(), 'src', 'asyncWorkerSocket') : 'socket.io-client',
      }
    },
    define: {
      // By default, Vite doesn't include shims for NodeJS/
      // necessary for segment analytics lib to work
      global: {},
    },
    worker: {
      format: 'es',
    },
    build: {
      rollupOptions: {
        output: {
          // Default
          dir: join('dist', process.env.npm_package_version),
          entryFileNames: (mode == 'production') ? 'assets/[name].js' : 'assets/[name].[hash].js',
          assetFileNames: (mode == 'production') ? 'assets/[name][extname]' : undefined,
          // chunkFileNames: "chunk-[name].js",
          // manualChunks: undefined,
        }
      }
    }
  })
}