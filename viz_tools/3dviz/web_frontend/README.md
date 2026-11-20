# 3dviz browser frontend

The optional browser mode uses this TypeScript and Three.js client. Scientific
processing and dataset access stay in the Python package. The client requests a
session, polls background jobs, and receives the versioned binary frame packet.
No dataset-specific implementation is imported here.

From this directory, using Node.js 20.19+ or 22.12+:

```bash
npm ci
npm test
npm run build
```

`build` checks TypeScript and writes the complete offline runtime to
`../src/threedviz/web/static/`, including the Three.js and Inter licenses. Commit the built
assets alongside source changes. Installed users need Python and a browser;
Node.js is only needed to develop or rebuild the frontend.

For frontend development, start the Python browser server on port 8765, then run
`npm run dev`. Vite proxies `/api` to that server. The production Python server
serves the generated assets directly; there are no CDN, external font, image or
JavaScript requests.

The controls use the bundled Inter 4.001 upright variable font (weights 100–900,
optical sizes 14–32) from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/inter).
`src/fonts/InterVariable.woff2` retains the complete font character coverage,
including Latin, Greek, and Cyrillic, and was losslessly compressed from the
upstream TrueType file using FontTools. The SIL Open Font License is kept in
`src/fonts/OFL.txt` and included in the build's `THIRD_PARTY.txt`. Font loading
stays local and works offline.

The renderer uses fixed pixel point sizes and a shader with the server's shared
256-color lookup tables. Gray is evaluated continuously as `0.12 + 0.6 * t`;
Viridis and Concerto use the same floored lookup as the Python implementation.
Camera RGB uses its validity mask and distance-gray fallback. Class wireframes
use screen-space line widths. Style changes update uniforms and retained GPU
objects without another frame transfer. Rendering runs only after interactions
or scene changes, with all geometries and materials disposed on replacement.

Coordinates in binary packets are rebased before conversion to float32 by the
server. The client retains the float64 origin and exports camera position and
pivot in the original reference coordinates. Camera JSON follows the desktop
convention: degrees in `(roll, pitch, yaw)`, with `Rz @ Ry @ Rx`. Applying an
arbitrary orientation preserves roll and an independent orbit pivot; the next
mouse interaction orbits about that pivot with reference Z up.
