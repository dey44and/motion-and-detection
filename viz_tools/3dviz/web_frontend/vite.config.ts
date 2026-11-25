import { defineConfig } from 'vite';
import { readFileSync } from 'node:fs';
export default defineConfig({
  base: './',
  plugins: [{
    name: 'third-party-license',
    generateBundle() {
      this.emitFile({
        type: 'asset',
        fileName: 'THIRD_PARTY.txt',
        source: 'This application bundles Three.js and its add-on modules.\n\n'
          + readFileSync(new URL('./node_modules/three/LICENSE', import.meta.url), 'utf8')
          + '\n\nThis application bundles the Source Sans Pro variable font from Adobe (2.045R).\n\n'
          + readFileSync(new URL('./src/fonts/OFL.txt', import.meta.url), 'utf8'),
      });
    },
  }],
  build: {outDir: '../src/threedviz/web/static', emptyOutDir: true, target: 'es2022', rollupOptions: {output: {manualChunks: {three: ['three', 'three/addons/controls/OrbitControls.js', 'three/addons/lines/LineSegments2.js', 'three/addons/lines/LineSegmentsGeometry.js', 'three/addons/lines/LineMaterial.js']}}}},
  server: {proxy: {'/api': 'http://127.0.0.1:8765'}},
});
