import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';
import { absolute, rotationAngles, rotationQuaternion } from './camera';
import { CameraRig } from './camera-rig';
import type { Box, CameraMarker, CameraView, FramePacket, Scheme, SemanticClass, Vec3 } from './types';

const EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
export interface ClassStyle {visible: boolean; color: string}
export interface PointStyle {scheme: Scheme; size: number; maxDistance: number}
export interface BoxStyle {width: number; groundTruth: boolean; predictions: boolean; groundTruthColor: string | null; predictionColor: string}
export class Viewer {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(60, 1, 0.1, 5000);
  readonly controls: OrbitControls;
  readonly origin = new THREE.Vector3();
  private points: THREE.Mesh<THREE.InstancedBufferGeometry, THREE.ShaderMaterial> | null = null;
  private readonly boxes = new Map<string, LineSegments2>();
  private readonly predictions = new Map<string, LineSegments2>();
  private readonly axes: LineSegments2;
  private readonly textures = new Map<string, THREE.DataTexture>();
  private readonly semanticPalette = new THREE.DataTexture(new Uint8Array(256 * 4), 256, 1);
  private bounds = new THREE.Box3(new THREE.Vector3(-10,-10,-2), new THREE.Vector3(10,10,2));
  private animation: number | null = null;
  private axesEnabled = true;
  private readonly observer: ResizeObserver;
  private readonly rig: CameraRig;
  private style: PointStyle = {scheme: 'gray', size: 3, maxDistance: 80};
  onCameraChange: (() => void) | null = null;
  onCameraSelect: ((id: string) => void) | null = null;
  constructor(readonly element: HTMLElement, palettes: Record<string, number[][]>) {
    this.renderer = new THREE.WebGLRenderer({antialias: true, alpha: false, powerPreference: 'high-performance'});
    this.renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0xffffff, 1);
    this.renderer.domElement.setAttribute('aria-label', 'Interactive 3D point cloud. Drag to rotate, right-drag to pan, scroll to zoom.');
    this.renderer.domElement.setAttribute('tabindex', '0');
    this.element.append(this.renderer.domElement);
    this.rig = new CameraRig(element, this.renderer.domElement, this.scene, id => this.onCameraSelect?.(id), () => this.invalidate());
    this.camera.up.set(0,0,1);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = false;
    this.controls.screenSpacePanning = true;
    this.controls.minDistance = 0.1;
    this.controls.maxDistance = 100000;
    this.controls.addEventListener('change', () => {this.invalidate(); this.onCameraChange?.();});
    this.semanticPalette.magFilter = this.semanticPalette.minFilter = THREE.NearestFilter;
    this.setSemanticClasses([], new Set());
    for (const [key, values] of Object.entries(palettes)) {
      if (values.length !== 256 || values.some(row => row.length !== 3 || row.some(v => !Number.isFinite(v) || v < 0 || v > 1))) throw new Error(`Invalid ${key} palette from the server.`);
      const data = new Float32Array(values.flatMap(row => [...row, 1]));
      const texture = new THREE.DataTexture(data, 256, 1, THREE.RGBAFormat, THREE.FloatType);
      texture.magFilter = texture.minFilter = THREE.NearestFilter;
      texture.needsUpdate = true; this.textures.set(key, texture);
    }
    const geometry = new LineSegmentsGeometry().setPositions([0,0,0,3,0,0, 0,0,0,0,3,0, 0,0,0,0,0,3]);
    geometry.setColors([1,0.12,0.16,1,0.12,0.16, 0.05,0.65,0.35,0.05,0.65,0.35, 0.13,0.35,1,0.13,0.35,1]);
    this.axes = new LineSegments2(geometry, new LineMaterial({linewidth: 2.5, vertexColors: true}));
    this.axes.frustumCulled = false; this.axes.visible = false; this.scene.add(this.axes);
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(element);
    this.resetCamera(); this.resize();
  }
  private resize() {
    const {width, height} = this.element.getBoundingClientRect();
    if (!width || !height) return;
    this.renderer.setSize(width, height);
    this.camera.aspect = width / height; this.camera.updateProjectionMatrix();
    if (this.points) this.points.material.uniforms.viewportHeight.value = this.renderer.domElement.height;
    this.axes.material.resolution.set(width, height);
    for (const line of this.boxes.values()) line.material.resolution.set(width, height);
    for (const line of this.predictions.values()) line.material.resolution.set(width, height);
    this.invalidate();
  }
  private invalidate() {
    if (this.animation !== null) return;
    this.animation = requestAnimationFrame(() => {
      this.animation = null;
      this.rig.updateScale(this.camera);
      this.renderer.render(this.scene, this.camera);
      this.rig.render(this.camera);
    });
  }
  setCameraMarkers(markers: readonly CameraMarker[]) {
    this.rig.setPoses(markers);
    this.invalidate();
  }
  setCameraImages(images: Readonly<Record<string, string>>) {this.rig.setImages(images);}
  setFrame(packet: FramePacket, reset: boolean) {
    const oldView = this.captureCamera();
    const {header, positions, distance, rgb, rgbMask} = packet;
    const origin = new THREE.Vector3(...header.origin);
    // Prepare replacement geometry before disposing the previous frame.
    // Share one 20-face shape across the cloud without per-point objects or matrices.
    const shape = new THREE.IcosahedronGeometry(1, 0);
    const geometry = new THREE.InstancedBufferGeometry();
    geometry.setAttribute('position', shape.getAttribute('position'));
    geometry.setAttribute('normal', shape.getAttribute('normal'));
    geometry.instanceCount = header.count;
    geometry.setAttribute('pointCenter', new THREE.InstancedBufferAttribute(positions, 3));
    geometry.setAttribute('distanceFromOrigin', new THREE.InstancedBufferAttribute(distance, 1));
    geometry.setAttribute('cameraRgb', new THREE.InstancedBufferAttribute(rgb ?? new Uint8Array(header.count * 3), 3, true));
    geometry.setAttribute('cameraValid', new THREE.InstancedBufferAttribute(rgbMask ?? new Uint8Array(header.count), 1));
    geometry.setAttribute('semanticId', new THREE.InstancedBufferAttribute(new Uint8Array(header.count).fill(255), 1));
    const material = new THREE.ShaderMaterial({
      uniforms: {pointSize: {value: this.style.size * this.renderer.getPixelRatio()}, viewportHeight: {value: this.renderer.domElement.height}, maxDistance: {value: this.style.maxDistance}, colorMode: {value: 0}, palette: {value: this.textures.get('viridis')}, semanticPalette: {value: this.semanticPalette}},
      vertexShader: `attribute vec3 pointCenter; attribute float distanceFromOrigin; attribute vec3 cameraRgb; attribute float cameraValid; attribute float semanticId;
        uniform float pointSize; uniform float viewportHeight;
        varying float vDistance; varying vec3 vRgb; varying float vValid; varying vec3 vNormal; varying float vSemantic;
        void main(){ vDistance=distanceFromOrigin; vRgb=cameraRgb; vValid=cameraValid; vSemantic=semanticId;
          vec4 center=modelViewMatrix*vec4(pointCenter,1.0);
          vNormal=normalMatrix*normal;
          if(center.z>=0.0){gl_Position=vec4(2.0,2.0,2.0,1.0);return;}
          vec3 vertex=mat3(modelViewMatrix)*position;
          // Keep the glyph's diameter in pixels, including at wide FOVs and screen edges.
          float radius=pointSize*max(-center.z,0.0)/(viewportHeight*projectionMatrix[1][1]);
          float depthRadius=min(radius,-center.z*0.25);
          vec4 clipCenter=projectionMatrix*center;
          center.z+=vertex.z*depthRadius;
          vec4 clipVertex=projectionMatrix*center;
          vec2 offset=vertex.xy*(pointSize/viewportHeight);
          offset.x*=projectionMatrix[0][0]/projectionMatrix[1][1];
          clipVertex.xy=(clipCenter.xy/clipCenter.w+offset)*clipVertex.w;
          gl_Position=clipVertex; }`,
      fragmentShader: `precision highp float; uniform float maxDistance; uniform int colorMode; uniform sampler2D palette; uniform sampler2D semanticPalette;
        varying float vDistance; varying vec3 vRgb; varying float vValid; varying vec3 vNormal; varying float vSemantic;
        void main(){float t=clamp(vDistance/maxDistance,0.0,1.0); vec3 gray=vec3(0.12+0.6*t);
          float index=min(floor(t*256.0),255.0); vec3 mapped=texture2D(palette,vec2((index+0.5)/256.0,0.5)).rgb;
          vec3 rgb=colorMode==0?gray:(colorMode==2?(vValid>0.5?vRgb:gray):mapped);
          if(colorMode==3){vec4 semantic=texture2D(semanticPalette,vec2((vSemantic+0.5)/256.0,0.5));
            if(semantic.a<0.5) discard;
            rgb=semantic.rgb;}
          float light=0.72+0.28*max(dot(normalize(vNormal),normalize(vec3(-0.4,0.6,1.0))),0.0);
          gl_FragColor=vec4(rgb*light,1.0);}`,
    });
    const points = new THREE.Mesh(geometry, material);
    // Centers and pixel-sized radii are applied in the shader, outside the base shape's bounds.
    points.frustumCulled = false;
    if (this.points) {this.scene.remove(this.points); this.points.geometry.dispose(); this.points.material.dispose();}
    this.points = points; this.scene.add(points);
    this.origin.copy(origin);
    this.rig.setPoses([]);
    this.rig.setOrigin(origin);
    this.replaceBoxes(this.boxes, header.boxes, false);
    this.replaceBoxes(this.predictions, [], true);
    this.axes.position.copy(origin).negate();
    this.axes.visible = this.axesEnabled;
    this.bounds.set(new THREE.Vector3(...header.bounds.min).sub(origin), new THREE.Vector3(...header.bounds.max).sub(origin));
    this.setPointStyle(this.style);
    if (reset) this.resetCamera(); else this.applyCamera(oldView);
    this.resize(); this.invalidate();
  }
  private replaceBoxes(layer: Map<string, LineSegments2>, boxes: readonly Box[], dashed: boolean) {
    for (const line of layer.values()) {this.scene.remove(line); line.geometry.dispose(); line.material.dispose();}
    layer.clear();
    const grouped = new Map<string, number[]>();
    for (const box of boxes) {
      const coordinates = grouped.get(box.class_id) ?? [];
      for (const [a,b] of EDGES) for (const index of [a,b]) {
        const [x, y, z] = box.corners[index];
        coordinates.push(x - this.origin.x, y - this.origin.y, z - this.origin.z);
      }
      grouped.set(box.class_id, coordinates);
    }
    const resolution = this.renderer.getSize(new THREE.Vector2());
    for (const [id, coordinates] of grouped) {
      const material = new LineMaterial({dashed, dashSize: .6, gapSize: .34});
      material.resolution.copy(resolution);
      const line = new LineSegments2(new LineSegmentsGeometry().setPositions(coordinates), material);
      if (dashed) line.computeLineDistances();
      layer.set(id, line);
      this.scene.add(line);
    }
  }
  setPredictionBoxes(boxes: readonly Box[]) {
    this.replaceBoxes(this.predictions, boxes, true);
    this.invalidate();
  }
  setSemanticLabels(labels: Uint8Array | null) {
    if (!this.points) return;
    const attribute = this.points.geometry.getAttribute('semanticId');
    const target = attribute.array as Uint8Array;
    if (labels && labels.length === target.length) target.set(labels);
    else target.fill(255);
    attribute.needsUpdate = true;
    this.invalidate();
  }
  setSemanticClasses(classes: readonly SemanticClass[], hidden: ReadonlySet<number>) {
    const colors = this.semanticPalette.image.data as Uint8Array;
    for (let index = 0; index < 256; index++) colors.set([120, 124, 133, 255], index * 4);
    for (const item of classes) colors.set([...item.color.map(c => Math.round(c * 255)), hidden.has(item.id) ? 0 : 255], item.id * 4);
    this.semanticPalette.needsUpdate = true;
    this.invalidate();
  }
  showPoints(visible: boolean) {
    if (this.points) this.points.visible = visible;
    this.invalidate();
  }
  setPointStyle(style: PointStyle) {
    this.style = {...style};
    if (this.points) {
      const uniforms = this.points.material.uniforms;
      uniforms.pointSize.value = style.size * this.renderer.getPixelRatio();
      uniforms.maxDistance.value = style.maxDistance;
      uniforms.colorMode.value = {gray: 0, viridis: 1, concerto: 1, rgb: 2, semantic: 3}[style.scheme];
      uniforms.palette.value = this.textures.get(style.scheme === 'concerto' ? 'concerto' : 'viridis');
    }
    this.invalidate();
  }
  setBoxStyle(styles: Map<string, ClassStyle>, appearance: BoxStyle) {
    for (const [id, line] of this.boxes) {
      const style = styles.get(id);
      line.visible = appearance.groundTruth && (style?.visible ?? true);
      line.material.color.set(appearance.groundTruthColor ?? style?.color ?? '#1959bf');
      line.material.linewidth = appearance.width;
    }
    for (const [id, line] of this.predictions) {
      line.visible = appearance.predictions && (styles.get(id)?.visible ?? true);
      line.material.color.set(appearance.predictionColor);
      line.material.linewidth = appearance.width;
    }
    this.invalidate();
  }
  showAxes(value: boolean) {this.axesEnabled = value; this.axes.visible = Boolean(this.points) && value; this.invalidate();}
  clearFrame() {
    if (this.points) {this.scene.remove(this.points); this.points.geometry.dispose(); this.points.material.dispose(); this.points = null;}
    this.rig.setPoses([]);
    this.replaceBoxes(this.boxes, [], false);
    this.replaceBoxes(this.predictions, [], true);
    this.axes.visible = false; this.invalidate();
  }
  resetCamera() {
    const center = this.bounds.getCenter(new THREE.Vector3());
    const radius = Math.max(this.bounds.getSize(new THREE.Vector3()).length() / 2, 5);
    const distance = radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 1.1;
    this.controls.target.copy(center);
    this.camera.up.set(0,0,1);
    this.camera.position.copy(center).add(new THREE.Vector3(-1,-1,0.85).normalize().multiplyScalar(distance));
    this.camera.near = Math.max(0.02, radius / 10000);
    this.camera.far = Math.max(5000, distance + radius * 10);
    this.camera.updateProjectionMatrix();
    this.controls.update(); this.invalidate(); this.onCameraChange?.();
  }
  captureCamera(): CameraView {
    return {position: absolute(this.camera.position, this.origin), rotation: rotationAngles(this.camera.quaternion), pivot: absolute(this.controls.target, this.origin), fov: this.camera.fov, near: this.camera.near, far: this.camera.far};
  }
  applyCamera(view: CameraView) {
    this.camera.position.set(...view.position).sub(this.origin);
    this.controls.target.set(...view.pivot).sub(this.origin);
    this.camera.quaternion.copy(rotationQuaternion(view.rotation));
    this.camera.up.set(0,0,1);
    this.camera.fov = view.fov; this.camera.near = view.near; this.camera.far = view.far;
    this.camera.updateProjectionMatrix();
    // OrbitControls updates only on interaction. Do not replace an explicitly
    // pasted orientation with lookAt(pivot): the pivot may be off the view axis.
    this.invalidate(); this.onCameraChange?.();
  }
  dispose() {
    this.rig.dispose();
    this.observer.disconnect(); this.controls.dispose();
    if (this.animation !== null) cancelAnimationFrame(this.animation);
    this.points?.geometry.dispose(); this.points?.material.dispose();
    for (const line of this.boxes.values()) {line.geometry.dispose(); line.material.dispose();}
    for (const line of this.predictions.values()) {line.geometry.dispose(); line.material.dispose();}
    this.axes.geometry.dispose(); this.axes.material.dispose();
    for (const texture of this.textures.values()) texture.dispose();
    this.semanticPalette.dispose();
    this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
export function rgbHex(color: Vec3): string {return '#' + color.map(v => Math.round(Math.max(0, Math.min(1,v)) * 255).toString(16).padStart(2, '0')).join('');}
