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
  })
}
