import { defineConfig } from 'vite';

export default defineConfig({
  root: '.',
  build: {
    outDir: 'dist',
    target: 'esnext',
  },
  server: {
    port: 3000,
    // For local development with self-signed certs
    https: false,
    // Proxy WebSocket connections to the pool service
    proxy: {
      '/ws': {
        target: 'ws://localhost:9834',
        ws: true,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/ws/, ''),
      },
    },
  },
});
