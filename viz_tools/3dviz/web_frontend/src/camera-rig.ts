import {
  BufferGeometry, DoubleSide, Float32BufferAttribute, Group, LineBasicMaterial, LineSegments,
  Mesh, MeshBasicMaterial, Points, PointsMaterial, Raycaster, SRGBColorSpace, TextureLoader, Vector2, Vector3,
} from 'three';
import type { PerspectiveCamera, Scene, Texture } from 'three';
import type { CameraMarker } from './types';
import './camera-rig.css';

const MIN_PREVIEW_WIDTH = 3;
const MAX_PREVIEW_WIDTH = 12.5;
const HIT_RADIUS = 14;
const PLACEHOLDER_COLOR = '#45464a';

type ScreenPoint = {x: number; y: number};
interface Frustum {
  id: string;
  mount: Group;
  image: Mesh<BufferGeometry, MeshBasicMaterial>;
  edges: LineSegments<BufferGeometry, LineBasicMaterial>;
  corners: Vector3[];
  center: Vector3;
  rayWidth: number;
  button: HTMLButtonElement;
  texture: Texture | null;
  src: string | undefined;
  screen: {center: ScreenPoint; mount: ScreenPoint | null; corners: ScreenPoint[]} | null;
}

function segmentDistance(point: ScreenPoint, start: ScreenPoint, end: ScreenPoint) {
  const dx = end.x - start.x, dy = end.y - start.y;
  const squared = dx * dx + dy * dy;
  const fraction = squared ? Math.min(1, Math.max(0, ((point.x - start.x) * dx + (point.y - start.y) * dy) / squared)) : 0;
  return Math.hypot(point.x - start.x - fraction * dx, point.y - start.y - fraction * dy);
}

export class CameraRig {
  private readonly group = new Group();
  private readonly rigCenter = new Vector3();
  private readonly cameraOffset = new Vector3();
  private readonly layer = document.createElement('div');
  private readonly raycaster = new Raycaster();
  private readonly loader = new TextureLoader();
  private readonly point = new BufferGeometry().setAttribute('position', new Float32BufferAttribute([0, 0, 0], 3));
  private readonly pointMaterial = new PointsMaterial({color: '#292a2c', size: 4, sizeAttenuation: false, toneMapped: false});
  private frustums: Frustum[] = [];
  private camera: PerspectiveCamera | null = null;
  private press: {id: number; x: number; y: number; moved: boolean} | null = null;

  constructor(viewport: HTMLElement, private readonly canvas: HTMLCanvasElement, private readonly scene: Scene,
              private readonly select: (id: string) => void, private readonly invalidate: () => void) {
    this.layer.className = 'camera-rig';
    this.layer.setAttribute('role', 'group');
    this.layer.setAttribute('aria-label', 'Ego camera positions');
    viewport.append(this.layer);
    this.scene.add(this.group);
    canvas.addEventListener('pointerdown', this.pointerDown);
    canvas.addEventListener('pointermove', this.pointerMove);
    canvas.addEventListener('pointerup', this.pointerUp);
    canvas.addEventListener('pointercancel', this.pointerCancel);
  }

  setOrigin(origin: Vector3) {this.group.position.copy(origin).negate();}

  setPoses(poses: readonly CameraMarker[]) {
    this.clear();
    for (const pose of poses) {
      const rays = pose.corner_rays;
      if (rays?.length !== 4 || rays.some(ray => !ray.every(Number.isFinite))) continue;
      const corners = rays.map(ray => new Vector3(...ray));
      const rayWidth = (corners[0].distanceTo(corners[1]) + corners[3].distanceTo(corners[2])) / 2;
      if (!Number.isFinite(rayWidth) || rayWidth <= Number.EPSILON) continue;
      const center = corners.reduce((sum, corner) => sum.add(corner), new Vector3()).multiplyScalar(.25);
      const imageGeometry = new BufferGeometry();
      imageGeometry.setAttribute('position', new Float32BufferAttribute(corners.flatMap(corner => corner.toArray()), 3));
      imageGeometry.setAttribute('uv', new Float32BufferAttribute([0, 1, 1, 1, 1, 0, 0, 0], 2));
      imageGeometry.setIndex([0, 2, 1, 0, 3, 2]);
      const image = new Mesh(imageGeometry, new MeshBasicMaterial({side: DoubleSide, color: PLACEHOLDER_COLOR, toneMapped: false}));
      // The four rays use native image corners at one optical depth. Their
      // common endpoint is the calibrated mounting translation, at local zero.
      const edgeGeometry = new BufferGeometry();
      edgeGeometry.setAttribute('position', new Float32BufferAttribute(corners.flatMap(corner => [0, 0, 0, ...corner]), 3));
      const edges = new LineSegments(edgeGeometry, new LineBasicMaterial({color: '#292a2c', toneMapped: false}));
      const mount = new Group();
      mount.position.set(...pose.position);
      mount.add(image, edges, new Points(this.point, this.pointMaterial));
      this.group.add(mount);
      this.rigCenter.add(mount.position);
      const button = document.createElement('button');
      button.className = 'camera-marker';
      button.type = 'button';
      button.dataset.cameraId = pose.id;
      button.setAttribute('aria-label', `Open ${pose.name} camera`);
      button.setAttribute('aria-haspopup', 'dialog');
      button.hidden = true;
      button.addEventListener('click', () => this.select(pose.id));
      this.layer.append(button);
      this.frustums.push({id: pose.id, mount, image, edges, corners, center, rayWidth, button, texture: null, src: undefined, screen: null});
    }
    if (this.frustums.length) this.rigCenter.divideScalar(this.frustums.length);
  }

  updateScale(camera: PerspectiveCamera) {
    if (!this.frustums.length) return;
    const distance = this.cameraOffset.copy(this.rigCenter).add(this.group.position).distanceTo(camera.position);
    // Use a common image width at this zoom level. Wide-angle cameras meet it
    // at a shallower optical depth while keeping their calibrated corner rays.
    const width = Math.min(MAX_PREVIEW_WIDTH, Math.max(MIN_PREVIEW_WIDTH, distance / 32));
    for (const frustum of this.frustums) frustum.mount.scale.setScalar(width / frustum.rayWidth);
  }

  setImages(images: Readonly<Record<string, string>>) {
    for (const frustum of this.frustums) {
      const src = Object.hasOwn(images, frustum.id) ? images[frustum.id] : undefined;
      if (src === frustum.src) continue;
      frustum.src = src;
      frustum.texture?.dispose();
      frustum.texture = null;
      const material = frustum.image.material;
      material.map = null;
      material.color.set(PLACEHOLDER_COLOR);
      material.needsUpdate = true;
      if (!src) continue;
      frustum.texture = this.loader.load(src, texture => {
        if (frustum.texture !== texture) return;
        texture.colorSpace = SRGBColorSpace;
        material.map = texture;
        material.color.set('#fff');
        material.needsUpdate = true;
        this.invalidate();
      });
    }
    this.invalidate();
  }

  render(camera: PerspectiveCamera) {
    this.camera = camera;
    const width = this.layer.clientWidth, height = this.layer.clientHeight;
    const project = (point: Vector3, mount: Group): ScreenPoint | null => {
      const clip = point.clone().applyMatrix4(mount.matrixWorld).project(camera);
      if (!Number.isFinite(clip.x) || !Number.isFinite(clip.y) || Math.abs(clip.z) > 1) return null;
      return {x: (clip.x + 1) * width / 2, y: (1 - clip.y) * height / 2};
    };
    const zero = new Vector3();
    for (const frustum of this.frustums) {
      frustum.screen = null;
      const center = project(frustum.center, frustum.mount);
      if (!center || !width || !height) {frustum.button.hidden = true; continue;}
      const corners = frustum.corners.map(corner => project(corner, frustum.mount)).filter(point => point !== null);
      const bounds = [center, ...corners];
      const left = Math.min(...bounds.map(point => point.x));
      const top = Math.min(...bounds.map(point => point.y));
      const right = Math.max(...bounds.map(point => point.x));
      const bottom = Math.max(...bounds.map(point => point.y));
      if (left > width || top > height || right < 0 || bottom < 0) {frustum.button.hidden = true; continue;}
      const targetWidth = Math.min(width, Math.max(HIT_RADIUS * 2, Math.min(width, right) - Math.max(0, left)));
      const targetHeight = Math.min(height, Math.max(HIT_RADIUS * 2, Math.min(height, bottom) - Math.max(0, top)));
      const targetX = Math.max(0, Math.min(width - targetWidth, center.x - targetWidth / 2));
      const targetY = Math.max(0, Math.min(height - targetHeight, center.y - targetHeight / 2));
      frustum.screen = {center, mount: project(zero, frustum.mount), corners};
      Object.assign(frustum.button.style, {left: `${targetX}px`, top: `${targetY}px`, width: `${targetWidth}px`, height: `${targetHeight}px`});
      frustum.button.hidden = false;
    }
  }

  private pointerDown = (event: PointerEvent) => {
    if (this.press) {this.press.moved = true; return;}
    if (event.button === 0 && this.frustums.length) this.press = {id: event.pointerId, x: event.clientX, y: event.clientY, moved: false};
  };
  private pointerMove = (event: PointerEvent) => {
    const press = this.press;
    if (press?.id === event.pointerId && Math.hypot(event.clientX - press.x, event.clientY - press.y) > 4) press.moved = true;
    this.canvas.classList.toggle('camera-frustum-hit', !press && !!this.pick(event.clientX, event.clientY));
  };
  private pointerCancel = (event: PointerEvent) => {if (this.press?.id === event.pointerId) this.press = null;};
  private pointerUp = (event: PointerEvent) => {
    const press = this.press;
    if (!press || press.id !== event.pointerId) return;
    this.press = null;
    if (press.moved || Math.hypot(event.clientX - press.x, event.clientY - press.y) > 4) return;
    const hit = this.pick(event.clientX, event.clientY);
    if (hit) this.select(hit.id);
  };

  private pick(clientX: number, clientY: number) {
    if (!this.camera || !this.frustums.length) return;
    const bounds = this.canvas.getBoundingClientRect();
    const point = {x: clientX - bounds.left, y: clientY - bounds.top};
    if (!bounds.width || !bounds.height || point.x < 0 || point.x > bounds.width || point.y < 0 || point.y > bounds.height) return;
    const ndc = new Vector2(point.x / bounds.width * 2 - 1, 1 - point.y / bounds.height * 2);
    this.raycaster.near = this.camera.near;
    this.raycaster.far = this.camera.far;
    this.raycaster.setFromCamera(ndc, this.camera);
    const hit = this.raycaster.intersectObjects(this.frustums.map(frustum => frustum.image), false)[0];
    const exact = this.frustums.find(frustum => frustum.image === hit?.object);
    if (exact) return exact;
    let nearest: Frustum | undefined;
    let shortest = HIT_RADIUS;
    for (const frustum of this.frustums) {
      const screen = frustum.screen;
      if (!screen) continue;
      let distance = Math.hypot(point.x - screen.center.x, point.y - screen.center.y);
      if (screen.mount) for (const corner of screen.corners) distance = Math.min(distance, segmentDistance(point, screen.mount, corner));
      if (distance < shortest) {nearest = frustum; shortest = distance;}
    }
    return nearest;
  }

  private clear() {
    this.press = null;
    this.canvas.classList.remove('camera-frustum-hit');
    for (const frustum of this.frustums) {
      const {image, edges} = frustum;
      frustum.texture?.dispose();
      // Retire the texture reference so queued image callbacks cannot publish.
      frustum.texture = null;
      image.geometry.dispose(); image.material.dispose();
      edges.geometry.dispose(); edges.material.dispose();
    }
    this.frustums = [];
    this.rigCenter.set(0, 0, 0);
    this.group.clear();
    this.layer.replaceChildren();
  }

  dispose() {
    this.clear();
    this.point.dispose();
    this.pointMaterial.dispose();
    this.scene.remove(this.group);
    this.layer.remove();
    this.canvas.removeEventListener('pointerdown', this.pointerDown);
    this.canvas.removeEventListener('pointermove', this.pointerMove);
    this.canvas.removeEventListener('pointerup', this.pointerUp);
    this.canvas.removeEventListener('pointercancel', this.pointerCancel);
  }
}
