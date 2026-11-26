import './camera-review.css';
import { cameraImageUrl, fetchCameraPoints } from './api';
import type { Bootstrap, CameraBox, CameraObservationView, CameraReviewState, FramePacket, Scheme, SemanticClass, Vec3 } from './types';
import { rgbHex } from './viewer';
import type { ClassStyle } from './viewer';

export interface CameraReviewOptions {
  session: string;
  packet: FramePacket;
  palettes: Bootstrap['palettes'];
  maxDistance: number;
  scheme: Scheme;
  cameraId?: string;
  semanticLabels?: Uint8Array | null;
  semanticClasses?: readonly SemanticClass[];
  hiddenSemanticClasses?: ReadonlySet<number>;
  classStyles?: ReadonlyMap<string, ClassStyle>;
  predictionColor?: string;
  lineWidth?: number;
  load: (points: boolean) => Promise<CameraReviewState | null>;
}

interface CameraTile {
  view: CameraObservationView;
  button: HTMLButtonElement;
  canvas: HTMLCanvasElement;
  points: DataView | null;
  imageReady: boolean;
}

const cameraSlots = new Map([
  ['front_left', 0], ['front', 1], ['front_right', 2],
  ['rear_right', 3], ['rear', 4], ['rear_left', 5],
  ['back_right', 3], ['back', 4], ['back_left', 5], ['side_left', 3], ['side_right', 5],
]);
const schemeNames: Record<Scheme, string> = {
  gray: 'Gray', viridis: 'Viridis', concerto: 'Concerto', rgb: 'Camera RGB', semantic: 'Semantic class',
};

function orderedCameras(views: CameraObservationView[]) {
  const rolesKnown = views.length > 1 && views.some(view => view.role && cameraSlots.has(view.role));
  const columns = rolesKnown ? 3 : Math.max(1, Math.min(views.length, 3));
  const occupied = new Set<number>();
  const order = views.map(view => {
    const preferred = rolesKnown && view.role ? cameraSlots.get(view.role) : undefined;
    if (preferred === undefined || occupied.has(preferred)) return {view, slot: -1};
    occupied.add(preferred);
    return {view, slot: preferred};
  });
  let slot = 0;
  for (const item of order) if (item.slot < 0) {
    while (occupied.has(slot)) slot++;
    item.slot = slot++;
  }
  return {columns, items: order.sort((a, b) => a.slot - b.slot)};
}

function cssColor(color: number[]): string {return rgbHex(color as Vec3);}
function pointColors(options: CameraReviewOptions, scheme: Scheme): (row: number) => string | null {
  const {distance, rgb, rgbMask} = options.packet;
  const maxDistance = Math.max(1, options.maxDistance);
  if (scheme === 'semantic' && options.semanticLabels) {
    const colors = new Map(options.semanticClasses?.map(item => [item.id, rgbHex(item.color)]));
    const labels = options.semanticLabels;
    return row => options.hiddenSemanticClasses?.has(labels[row]) ? null : colors.get(labels[row]) ?? '#787c85';
  }
  const mapped = scheme === 'viridis' || scheme === 'concerto' ? options.palettes[scheme].map(cssColor) : null;
  const gray = (row: number) => {
    const component = Math.round((.12 + .6 * Math.min(1, distance[row] / maxDistance)) * 255);
    return `rgb(${component} ${component} ${component})`;
  };
  if (scheme === 'rgb' && rgb && rgbMask) return row => rgbMask[row] ? `rgb(${rgb[row * 3]} ${rgb[row * 3 + 1]} ${rgb[row * 3 + 2]})` : gray(row);
  if (mapped) return row => mapped[Math.min(255, Math.floor(distance[row] / maxDistance * 256))];
  return gray;
}

function drawBoxes(context: CanvasRenderingContext2D, boxes: CameraBox[], width: number, scale: number,
                   styles?: ReadonlyMap<string, ClassStyle>, color?: string, dashed = false) {
  context.lineWidth = width / scale;
  context.lineCap = 'round';
  context.setLineDash(dashed ? [5 / scale, 3 / scale] : []);
  for (const box of boxes) {
    const style = styles?.get(box.class_id);
    if (style?.visible === false) continue;
    context.strokeStyle = color ?? style?.color ?? '#4ce880';
    context.beginPath();
    for (const [u0, v0, u1, v1] of box.segments) {
      context.moveTo(u0, v0);
      context.lineTo(u1, v1);
    }
    context.stroke();
  }
}

export class CameraReviewModal {
  private readonly dialog = document.createElement('dialog');
  private readonly grid: HTMLDivElement;
  private readonly status: HTMLParagraphElement;
  private readonly controls: HTMLElement;
  private readonly groundTruth: HTMLInputElement;
  private readonly groundTruthOverride: HTMLInputElement;
  private readonly groundTruthColor: HTMLInputElement;
  private readonly predictions: HTMLInputElement;
  private readonly projected: HTMLInputElement;
  private readonly coloring: HTMLSelectElement;
  private readonly distance: HTMLParagraphElement;
  private readonly observer: ResizeObserver;
  private active: CameraReviewOptions | null = null;
  private controller: AbortController | null = null;
  private tiles: CameraTile[] = [];
  private drawing: number | null = null;
  private projecting = false;

  constructor() {
    this.dialog.id = 'camera-review';
    this.dialog.className = 'camera-review glass';
    this.dialog.setAttribute('aria-labelledby', 'camera-review-title');
    this.dialog.innerHTML = `
      <header class="camera-review-heading">
        <h2 id="camera-review-title">Camera images</h2>
        <button id="camera-review-close" class="icon-button" type="button" aria-label="Close camera images" autofocus>×</button>
      </header>
      <p id="camera-review-status" class="camera-review-status" role="status" hidden></p>
      <div class="camera-review-body">
        <div id="camera-review-grid" class="camera-review-grid" role="group" aria-label="Camera images"></div>
        <aside id="camera-review-controls" class="camera-review-controls" aria-label="Camera overlays" hidden>
          <h3>Overlays</h3>
          <label class="check-row"><span>Ground truth boxes</span><input id="camera-layer-gt" type="checkbox" checked></label>
          <div class="layer-color">
            <label class="check-row"><input id="camera-override-gt-color" type="checkbox"><span>Use one ground truth color</span></label>
            <input id="camera-ground-truth-color" type="color" value="#148c38" aria-label="Camera ground truth color" disabled>
          </div>
          <label class="check-row"><span>Predicted boxes</span><input id="camera-layer-predictions" type="checkbox" checked></label>
          <label class="check-row camera-point-switch"><span>Projected LiDAR points</span><input id="camera-layer-points" type="checkbox"></label>
          <label class="field-label" for="camera-point-colors">Point coloring</label>
          <select id="camera-point-colors" class="full-select"></select>
          <p id="camera-point-distance" class="hint" hidden></p>
        </aside>
      </div>`;
    const find = <T extends HTMLElement>(id: string) => this.dialog.querySelector<T>(`#${id}`)!;
    this.grid = find('camera-review-grid');
    this.status = find('camera-review-status');
    this.controls = find('camera-review-controls');
    this.groundTruth = find('camera-layer-gt');
    this.groundTruthOverride = find('camera-override-gt-color');
    this.groundTruthColor = find('camera-ground-truth-color');
    this.predictions = find('camera-layer-predictions');
    this.projected = find('camera-layer-points');
    this.coloring = find('camera-point-colors');
    this.distance = find('camera-point-distance');
    find('camera-review-close').addEventListener('click', () => this.close());
    this.dialog.addEventListener('close', () => {if (!this.dialog.open) this.reset();});
    this.dialog.addEventListener('keydown', event => event.stopPropagation());
    for (const input of [this.groundTruth, this.groundTruthOverride, this.predictions, this.coloring]) {
      input.addEventListener('change', () => {this.syncControls(); this.scheduleDraw();});
    }
    this.groundTruthColor.addEventListener('input', () => this.scheduleDraw());
    this.projected.addEventListener('change', () => {
      this.syncControls(); this.scheduleDraw();
      if (this.projected.checked) void this.loadPoints();
    });
    document.body.append(this.dialog);
    this.observer = new ResizeObserver(() => this.scheduleDraw());
    this.observer.observe(this.grid);
  }

  get isOpen() {return this.dialog.open;}
  close() {if (this.dialog.open) this.dialog.close(); this.reset();}

  open(options: CameraReviewOptions) {
    this.reset();
    this.active = options;
    this.controller = new AbortController();
    this.groundTruth.checked = this.predictions.checked = true;
    this.projected.checked = false;
    const schemes: Scheme[] = ['gray', 'viridis', 'concerto'];
    if (options.packet.rgb && options.packet.rgbMask) schemes.push('rgb');
    if (options.semanticLabels?.length === options.packet.header.count && options.semanticClasses?.length) schemes.push('semantic');
    this.coloring.replaceChildren(...schemes.map(scheme => new Option(schemeNames[scheme], scheme)));
    this.coloring.value = schemes.includes(options.scheme) ? options.scheme : 'gray';
    this.distance.textContent = `Distance scale · ${options.maxDistance} m`;
    this.setStatus('Loading camera images…');
    if (!this.dialog.open) this.dialog.showModal();
    this.dialog.scrollTop = 0;
    void this.loadViews(options);
  }

  private reset() {
    this.controller?.abort();
    if (this.drawing !== null) cancelAnimationFrame(this.drawing);
    this.active = null;
    this.controller = null;
    this.drawing = null;
    this.projecting = false;
    this.tiles = [];
    this.grid.replaceChildren();
    this.controls.hidden = true;
  }

  private setStatus(message: string) {
    this.status.textContent = message;
    this.status.hidden = !message;
  }

  private syncControls() {
    const calibrated = this.tiles.some(tile => tile.view.projection);
    this.groundTruth.disabled = this.predictions.disabled = this.projected.disabled = !calibrated;
    this.groundTruthOverride.disabled = !calibrated || !this.groundTruth.checked;
    this.groundTruthColor.disabled = this.groundTruthOverride.disabled || !this.groundTruthOverride.checked;
    this.coloring.disabled = !calibrated || !this.projected.checked;
    this.distance.hidden = this.coloring.disabled || ['rgb', 'semantic'].includes(this.coloring.value);
  }

  private async loadViews(options: CameraReviewOptions) {
    try {
      const review = await options.load(false);
      if (this.active !== options) return;
      if (!review || review.revision !== options.packet.header.revision) throw new Error('Camera images could not be loaded for the current frame.');
      if (!review.views.length) {
        this.setStatus('No camera images are available for this frame.');
        return;
      }
      const {columns, items} = orderedCameras(review.views);
      this.grid.style.setProperty('--camera-columns', String(columns));
      this.tiles = items.map(({view, slot}) => this.makeTile(view, slot, columns, options));
      this.grid.replaceChildren(...this.tiles.map(tile => tile.button));
      this.controls.hidden = false;
      this.syncControls();
      this.setStatus(this.tiles.some(tile => tile.view.projection) ? '' : 'Camera calibration is unavailable for overlays on this frame.');
      const selected = this.tiles.find(tile => tile.view.id === options.cameraId || tile.view.role === options.cameraId)
        ?? this.tiles.find(tile => tile.view.role === 'front') ?? this.tiles[0];
      this.selectTile(selected);
      selected.button.focus({preventScroll: true});
    } catch (error) {
      if (this.active === options) this.setStatus(error instanceof Error ? error.message : String(error));
    }
  }

  private makeTile(view: CameraObservationView, slot: number, columns: number, options: CameraReviewOptions): CameraTile {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'camera-review-tile';
    button.dataset.cameraId = view.id;
    if (view.role) button.dataset.cameraRole = view.role;
    button.setAttribute('aria-pressed', 'false');
    button.style.gridRowStart = String(Math.floor(slot / columns) + 1);
    button.style.gridColumnStart = String(slot % columns + 1);
    const preview = document.createElement('span');
    preview.className = 'camera-review-preview';
    preview.style.aspectRatio = `${view.width} / ${view.height}`;
    const image = document.createElement('img');
    image.alt = '';
    image.decoding = 'async';
    image.draggable = false;
    const canvas = document.createElement('canvas');
    canvas.setAttribute('aria-hidden', 'true');
    const fallback = document.createElement('span');
    fallback.className = 'camera-image-fallback';
    fallback.textContent = 'Loading image…';
    const caption = document.createElement('span');
    caption.className = 'camera-review-caption';
    caption.textContent = view.name;
    const tile: CameraTile = {view, button, canvas, points: null, imageReady: false};
    image.addEventListener('load', () => {
      fallback.hidden = true;
      tile.imageReady = true;
      this.scheduleDraw();
    });
    image.addEventListener('error', () => {
      image.hidden = true;
      fallback.textContent = 'Image unavailable';
    });
    preview.append(image, canvas, fallback);
    button.append(preview, caption);
    button.addEventListener('click', () => this.selectTile(tile));
    image.src = cameraImageUrl(options.session, options.packet.header.revision, view.id);
    return tile;
  }

  private selectTile(selected: CameraTile) {
    for (const tile of this.tiles) tile.button.setAttribute('aria-pressed', String(tile === selected));
  }

  private async loadPoints() {
    const options = this.active;
    const controller = this.controller;
    const pending = this.tiles.filter(tile => tile.view.projection && tile.points === null);
    if (!options || !controller || this.projecting || !pending.length) return;
    const {revision, count} = options.packet.header;
    this.projecting = true;
    this.setStatus('Projecting LiDAR points…');
    try {
      const review = await options.load(true);
      if (this.controller !== controller) return;
      if (!review || review.revision !== revision) throw new Error('Projected points could not be loaded for the current frame.');
      const byId = new Map(review.views.map(view => [view.id, view]));
      const results = await Promise.allSettled(pending.map(async tile => {
        const view = byId.get(tile.view.id);
        if (!view || view.point_count === null || view.width !== tile.view.width || view.height !== tile.view.height) {
          throw new Error('Projection is unavailable for this camera.');
        }
        const points = view.point_count === 0 ? new DataView(new ArrayBuffer(0))
          : await fetchCameraPoints(options.session, revision, view.id, view.point_count, count, controller.signal);
        if (!controller.signal.aborted) tile.points = points;
      }));
      if (this.controller !== controller) return;
      const failures = results.filter(result => result.status === 'rejected').length;
      this.setStatus(failures ? `Projected LiDAR points could not be loaded for ${failures} ${failures === 1 ? 'camera' : 'cameras'}.` : '');
      this.scheduleDraw();
    } catch (error) {
      if (this.controller === controller) this.setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      if (this.controller === controller) this.projecting = false;
    }
  }

  private scheduleDraw() {
    if (!this.active || !this.dialog.open || this.drawing !== null) return;
    this.drawing = requestAnimationFrame(() => {
      this.drawing = null;
      this.draw();
    });
  }

  private draw() {
    const options = this.active;
    if (!options) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const resolveColor = this.projected.checked ? pointColors(options, this.coloring.value as Scheme) : null;
    for (const tile of this.tiles) {
      if (!tile.imageReady) continue;
      const {canvas, points, view} = tile;
      const width = Math.round(canvas.clientWidth * dpr);
      const height = Math.round(canvas.clientHeight * dpr);
      if (!width || !height) continue;
      if (canvas.width !== width) canvas.width = width;
      if (canvas.height !== height) canvas.height = height;
      const context = canvas.getContext('2d');
      if (!context) continue;
      const sx = width / view.width, sy = height / view.height;
      context.setTransform(sx, 0, 0, sy, 0, 0);
      context.clearRect(0, 0, view.width, view.height);
      if (resolveColor && points) {
        const diameter = 2 * dpr, dx = diameter / sx, dy = diameter / sy;
        for (let offset = 0; offset < points.byteLength; offset += 12) {
          const color = resolveColor(points.getUint32(offset + 8, true));
          if (color === null) continue;
          context.fillStyle = color;
          context.fillRect(points.getFloat32(offset, true) - dx / 2, points.getFloat32(offset + 4, true) - dy / 2, dx, dy);
        }
      }
      const lineWidth = (options.lineWidth ?? 2) * dpr;
      if (this.groundTruth.checked) {
        const color = this.groundTruthOverride.checked ? this.groundTruthColor.value : undefined;
        drawBoxes(context, view.ground_truth, lineWidth, sx, options.classStyles, color);
      }
      if (this.predictions.checked) drawBoxes(context, view.predictions, lineWidth, sx, options.classStyles, options.predictionColor ?? '#e64008', true);
    }
  }
}
