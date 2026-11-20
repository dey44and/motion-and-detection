import './style.css';
import { bootstrap, closeSession, createSession, fetchFrame, runJob } from './api';
import { parseCamera } from './camera';
import { Viewer, rgbHex } from './viewer';
import type { ClassStyle } from './viewer';
import type { Bootstrap, FramePacket, JobRequest, Scheme, ViewerState } from './types';

const brandLogo = new URL('./assets/3dviz-logo.png', import.meta.url).href;
const icon = (name: string, size = 18) => {
  const paths: Record<string, string> = {
    cube: '<path d="m12 3 8 4.5v9L12 21l-8-4.5v-9L12 3Z"/><path d="m4 7.5 8 4.5 8-4.5M12 12v9"/>',
    chevron: '<path d="m9 5 7 7-7 7"/>',
    sliders: '<path d="M4 7h16M4 17h16M8 4v6M16 14v6"/>',
    reset: '<path d="M4 10a8 8 0 1 1 1 7M4 4v6h6"/>',
    copy: '<rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V4a1 1 0 0 0-1-1H4a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h4"/>',
    close: '<path d="m6 6 12 12M6 18 18 6"/>',
    folder: '<path d="M3 7V5h7l2 2h9v13H3V7Z"/>',
    mouse: '<rect x="6" y="2" width="12" height="20" rx="6"/><path d="M12 2v6"/>',
    arrow: '<path d="M5 12h14m-6-6 6 6-6 6"/>',
    check: '<path d="m5 12 4 4L19 6"/>',
  };
  return `<svg width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name] ?? paths.cube}</svg>`;
};
const app = document.querySelector<HTMLDivElement>('#app')!;
app.innerHTML = `
  <main id="viewport" aria-label="3D scene"></main>
  <header class="navbar glass" aria-label="Dataset and frame navigation">
    <a class="brand" href="/" aria-label="3dviz home"><img class="brand-logo" src="${brandLogo}" width="42" height="25" alt=""><span>3dviz</span></a>
    <div class="nav-divider"></div>
    <label class="nav-field dataset-field"><span>Dataset</span><select id="dataset" aria-label="Dataset"><option>Connecting…</option></select></label>
    <label class="nav-field split-field"><span>Split</span><select id="split" aria-label="Split" disabled><option>—</option></select></label>
    <label class="nav-field scene-field"><span>Scene</span><select id="scene" aria-label="Scene" disabled><option>Load a dataset</option></select></label>
    <div class="frame-nav" role="group" aria-labelledby="frame-caption"><span id="frame-caption">Frame</span><input id="frame-slider" class="frame-slider" type="range" min="0" max="0" value="0" aria-label="Frame" disabled><button id="previous" class="icon-button previous" aria-label="Previous frame" title="Previous frame · ←" disabled>${icon('chevron')}</button><label class="frame-label"><span class="sr-only">Frame number</span><div><input id="frame-number" type="number" min="1" value="1" aria-label="Frame number" disabled><span id="frame-total">/ —</span></div></label><button id="next" class="icon-button" aria-label="Next frame" title="Next frame · →" disabled>${icon('chevron')}</button></div>
    <div id="load-strip" class="load-strip" role="progressbar" aria-label="Loading dataset" hidden><span></span></div>
  </header>
  <aside id="sidebar" class="sidebar glass is-loading" aria-label="Visualization controls">
    <div class="panel-heading"><h1>Scene controls</h1><button id="menu-toggle" class="icon-button menu-toggle" type="button" aria-label="Minimize scene controls" aria-expanded="true" aria-controls="sidebar-content" title="Minimize scene controls">${icon('chevron')}</button></div>
    <div id="sidebar-content">
    <details open id="source-section"><summary><span>${icon('folder')} Data source</span></summary><div class="section-content">
      <form id="source-form"><label class="field-label" for="root">Dataset root</label><div class="input-wrap"><input id="root" autocomplete="off" spellcheck="false" placeholder="/path/to/dataset" aria-describedby="source-hint"></div>
      <div class="source-row"><label class="compact-field">Release<input id="version" placeholder="auto" autocomplete="off" spellcheck="false"></label><button id="connect" class="button primary" type="submit">Load dataset ${icon('arrow',16)}</button></div></form>
      <p id="source-hint" class="hint">Paths are read on this computer.</p><p id="scene-description" class="scene-description" hidden></p>
    </div></details>
    <details open><summary><span>${icon('cube')} Point cloud</span></summary><div class="section-content">
      <div class="control-heading"><label for="point-size">Point size</label><output id="point-size-value"><input id="point-size-number" type="number" min="1" max="12" step="0.5" value="3" aria-label="Point size in pixels"> px</output></div><input id="point-size" type="range" min="1" max="12" step="0.5" value="3">
      <label class="field-label palette-label" for="point-colors">Coloring</label><select id="point-colors" class="full-select"><option value="gray">Gray</option><option value="viridis">Viridis</option><option value="rgb" disabled>Camera RGB</option><option value="concerto">Concerto</option></select>
      <div class="control-heading scale-label"><label for="max-distance">Color scale max</label><output id="max-distance-value"><input id="max-distance-number" type="number" min="1" max="1000" value="80" aria-label="Maximum distance in meters"> m</output></div><div class="scale-row"><span>0</span><input id="max-distance" type="range" min="1" max="1000" step="1" value="80"><span>1000 m</span></div>
      <p id="rgb-coverage" class="hint" hidden></p>
      <label class="check-row raw-row"><input id="raw-points" type="checkbox" aria-describedby="raw-hint" disabled><span>Undo ego motion compensation</span></label><p id="raw-hint" class="hint">Load a dataset to check availability.</p>
    </div></details>
    <details open><summary><span><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M4 4h16v16H4zM4 4l5 5h11M9 9v11M4 20l5-5h11"/></svg> 3D bounding boxes</span><span id="box-total" class="section-count">—</span></summary><div class="section-content">
      <div class="control-heading"><label for="box-width">Line width</label><output id="box-width-value"><input id="box-width-number" type="number" min="1" max="10" step="0.5" value="2" aria-label="Box line width in pixels"> px</output></div><input id="box-width" type="range" min="1" max="10" step="0.5" value="2">
      <div class="text-actions"><button id="show-all" type="button">Show all</button><span>·</span><button id="hide-all" type="button">Hide all</button></div><div id="classes" class="classes"><p class="hint">Object classes appear after loading a dataset.</p></div><p id="annotation-note" class="hint" hidden></p>
    </div></details>
    <details><summary><span>${icon('sliders')} View</span></summary><div class="section-content">
      <label class="check-row"><input id="axes" type="checkbox" checked><span>Show coordinate axes <small>3 m</small></span></label>
    </div></details>
    <div id="status-footer" class="panel-footer"><span><span id="status">Connecting to local server</span><span class="loading-dots" aria-hidden="true">...</span></span></div>
    </div>
  </aside>
  <div id="right-overlays" class="right-overlays">
    <details id="camera-panel" class="camera-panel glass" open aria-label="Camera view controls"><summary><span>Camera view</span></summary><div class="camera-content">
      <label class="sr-only" for="camera-json">Full camera view JSON</label><textarea id="camera-json" rows="8" spellcheck="false" aria-describedby="camera-hint" placeholder="Load a scene to copy or apply a camera view."></textarea>
      <div class="camera-actions"><button id="reset-camera" class="icon-button" type="button" aria-label="Reset camera" title="Reset camera">${icon('reset',16)}</button><button id="copy-camera" class="button secondary" type="button">${icon('copy',14)} Copy</button><button id="paste-camera" class="button secondary" type="button">Paste</button><button id="apply-camera" class="button primary" type="button">Apply</button></div><p id="camera-hint" class="hint" role="status" aria-live="polite" hidden></p>
    </div></details>
    <div id="error" class="error-toast glass" role="alert" hidden><div><strong>Unable to complete the request</strong><p id="error-text"></p></div><button id="dismiss-error" class="icon-button" aria-label="Dismiss error">${icon('close')}</button></div>
  </div>
  <section id="empty-state" class="empty-state"><h2>Load a dataset</h2><p>Choose a dataset and enter its root directory in Data source.</p><button id="try-demo" class="button secondary">Load demo ${icon('arrow',13)}</button></section>
  <div class="viewport-footer"><div class="interaction-hint">${icon('mouse',16)}<span>Drag to rotate <b>·</b> Scroll to zoom <b>·</b> Right drag to pan</span></div><div id="frame-stats" class="frame-stats" hidden></div></div>
  <div id="legend" class="legend" hidden><div class="legend-caption"><span>Distance from origin</span><span id="legend-max">80 m</span></div><div id="legend-ramp"></div><div class="legend-labels"><span>0 m</span><span id="legend-far">80 m</span></div></div>
`;
function el<T extends HTMLElement = HTMLElement>(id: string): T {return document.getElementById(id) as T;}
const dataset = el<HTMLSelectElement>('dataset'), root = el<HTMLInputElement>('root'), version = el<HTMLInputElement>('version');
const split = el<HTMLSelectElement>('split'), scene = el<HTMLSelectElement>('scene');
const color = el<HTMLSelectElement>('point-colors'), raw = el<HTMLInputElement>('raw-points');
const cameraJson = el<HTMLTextAreaElement>('camera-json');
let info: Bootstrap | null = null;
let state: ViewerState | null = null;
let packet: FramePacket | null = null;
let viewer: Viewer | null = null;
let session: string | null = null;
let busy = false;
let disposed = false;
let generation = 0;
let cameraDirty = false;
let committedColor: Scheme = 'gray';
let cameraSync: ReturnType<typeof setTimeout> | null = null;
const classStyles = new Map<string, ClassStyle>();

function error(message: unknown) {el('error-text').textContent = message instanceof Error ? message.message : String(message); el('error').hidden = false;}
function setStatus(message: string) {
  el('status').textContent = message.replace(/(?:…|\.{3,})\s*$/, '');
  el('status-footer').hidden = !message;
  if (message) el('announcements').textContent = message;
}
function cameraMessage(message = '', invalid?: boolean) {
  el('camera-hint').textContent = message;
  el('camera-hint').hidden = !message;
  if (invalid !== undefined) cameraJson.setAttribute('aria-invalid', String(invalid));
}
function setBusy(value: boolean, message = '') {
  busy = value; el('load-strip').hidden = !value;
  app.setAttribute('aria-busy', String(value));
  if (message) setStatus(message.replace(/[_-]/g,' '));
  if (!value && packet && el('error').hidden) setStatus('');
  el('sidebar').classList.toggle('is-loading', value);
  syncDisabled();
}
function syncDisabled() {
  const locked = state?.dataset.configured ?? info?.initial.configured ?? false;
  dataset.disabled = busy || locked;
  root.readOnly = locked; version.readOnly = locked;
  root.disabled = busy || dataset.value === 'demo';
  version.disabled = busy || dataset.value === 'demo';
  el<HTMLButtonElement>('connect').disabled = busy || !session;
  el<HTMLButtonElement>('try-demo').disabled = busy || !session || locked;
  const pendingProvider = Boolean(state && dataset.value !== state.dataset.name);
  split.disabled = busy || pendingProvider || !state || state.metadata.splits.length < 2;
  scene.disabled = busy || pendingProvider || !state || !state.scenes.length;
  const length = state?.frames.length ?? 0, index = state?.frame_index ?? 0;
  el<HTMLButtonElement>('previous').disabled = busy || pendingProvider || !length || index <= 0;
  el<HTMLButtonElement>('next').disabled = busy || pendingProvider || !length || index >= length - 1;
  el<HTMLInputElement>('frame-number').disabled = busy || pendingProvider || !length;
  el<HTMLInputElement>('frame-slider').disabled = busy || pendingProvider || length < 2;
  raw.disabled = busy || pendingProvider || !supportsRaw();
  color.disabled = busy || pendingProvider;
  for (const id of ['copy-camera','paste-camera','apply-camera','reset-camera']) el<HTMLButtonElement>(id).disabled = !packet;
}
function supports(name: string) {return state?.metadata.capabilities.includes(name) ?? false;}
function supportsRaw() {return supports('acquisition_geometry') && state?.metadata.feature_coverage.acquisition_geometry !== 'partial';}
function option(select: HTMLSelectElement, entries: {value: string; text: string}[], value: string) {
  select.replaceChildren(...entries.map(entry => {const node = document.createElement('option'); node.value = entry.value; node.textContent = entry.text; return node;})); select.value = value;
}
function updateLegend() {
  const max = el<HTMLInputElement>('max-distance').value;
  el('legend').hidden = !packet || color.value === 'rgb';
  el('legend-max').textContent = `${max} m`; el('legend-far').textContent = `${max} m`;
  const palette = info?.palettes[color.value as 'gray' | 'viridis' | 'concerto'];
  const stops = color.value === 'gray' ? ['rgb(31,31,31)', 'rgb(184,184,184)'] : palette?.filter((_, i) => i % 16 === 0 || i === 255).map(c => rgbHex(c as [number,number,number])) ?? [];
  el('legend-ramp').style.background = `linear-gradient(90deg,${stops.join(',')})`;
}
function updatePointStyle() {
  const size = Number(el<HTMLInputElement>('point-size').value), maxDistance = Number(el<HTMLInputElement>('max-distance').value);
  el<HTMLInputElement>('point-size-number').value = String(size);
  el<HTMLInputElement>('max-distance-number').value = String(maxDistance);
  const scheme = color.value === 'rgb' && !state?.display.rgb ? committedColor : color.value as Scheme;
  viewer?.setPointStyle({scheme, size, maxDistance});
  el<HTMLInputElement>('max-distance').disabled = color.value === 'rgb';
  el<HTMLInputElement>('max-distance-number').disabled = color.value === 'rgb';
  if (color.value !== 'rgb' || state?.display.rgb) committedColor = color.value as Scheme;
  el('rgb-coverage').hidden = color.value !== 'rgb';
  updateLegend();
}
function updateBoxStyle() {const width = Number(el<HTMLInputElement>('box-width').value); el<HTMLInputElement>('box-width-number').value = String(width); viewer?.setBoxStyle(classStyles, width);}
function refreshCamera() {
  if (!viewer || !packet || cameraDirty) return;
  const fields = Object.entries(viewer.captureCamera()).map(([key, value]) => `  "${key}": ${JSON.stringify(value)}`);
  cameraJson.value = `{\n${fields.join(',\n')}\n}`;
}
function queueCameraRefresh() {if (cameraSync) clearTimeout(cameraSync); cameraSync = setTimeout(refreshCamera, 80);}
function renderClasses() {
  const classes = state?.metadata.classes ?? [];
  const groups = new Map<string, typeof classes>();
  for (const item of classes) {const key = item.group || item.name; const group = groups.get(key) ?? []; group.push(item); groups.set(key, group); if (!classStyles.has(item.id)) classStyles.set(item.id, {visible:true, color:rgbHex(item.color)});}
  const container = el('classes'); container.replaceChildren();
  const available = packet?.header.annotation_status.toLowerCase() === 'available';
  el('box-total').textContent = packet ? (available ? String(packet.header.boxes.length) : '—') : '—';
  for (const [name, items] of groups) {
    const row = document.createElement('div'); row.className = 'class-row';
    const label = document.createElement('label'); label.className = 'class-label';
    const checkbox = document.createElement('input'); checkbox.type = 'checkbox'; checkbox.checked = items.every(c => classStyles.get(c.id)?.visible); checkbox.indeterminate = !checkbox.checked && items.some(c => classStyles.get(c.id)?.visible); checkbox.setAttribute('aria-label', `Show ${name.replaceAll('_',' ')} boxes`);
    const text = document.createElement('span'); text.textContent = name.replaceAll('_',' ');
    const count = document.createElement('span'); count.className = 'class-count'; count.textContent = available ? String(packet!.header.boxes.filter(box => items.some(c => c.id === box.class_id)).length) : '—';
    label.append(checkbox, text, count);
    const picker = document.createElement('input'); picker.type = 'color'; picker.value = classStyles.get(items[0].id)!.color; picker.setAttribute('aria-label', `${name.replaceAll('_',' ')} box color`); picker.title = `${name.replaceAll('_',' ')} box color`;
    checkbox.addEventListener('change', () => {for (const item of items) classStyles.get(item.id)!.visible = checkbox.checked; checkbox.indeterminate = false; updateBoxStyle();});
    picker.addEventListener('input', () => {for (const item of items) classStyles.get(item.id)!.color = picker.value; updateBoxStyle();});
    row.append(label,picker); container.append(row);
  }
  if (!classes.length) {const hint = document.createElement('p'); hint.className = 'hint'; hint.textContent = 'No annotation classes in this dataset.'; container.append(hint);}
  el('annotation-note').hidden = !packet || available;
  el('annotation-note').textContent = packet?.header.annotation_status.toLowerCase() === 'withheld' ? 'Ground truth is withheld for this frame.' : 'Ground truth is not provided for this frame.';
}
function renderState() {
  if (!state) return;
  option(dataset, info!.datasets.map(value => ({value, text: value === 'demo' ? 'Synthetic demo' : value === 'nuscenes' ? 'nuScenes' : value === 'waymo' ? 'Waymo' : value})), state.dataset.name);
  root.value = state.dataset.root; version.value = state.dataset.version;
  option(split, state.metadata.splits.map(value => ({value, text:value})), state.split);
  option(scene, state.scenes.map(s => ({value:s.id, text:s.name || s.id})), state.scene_id);
  const currentScene = state.scenes.find(s => s.id === state!.scene_id);
  el('scene-description').textContent = currentScene?.description || ''; el('scene-description').hidden = !currentScene?.description;
  el('source-hint').textContent = state.dataset.configured ? 'Source settings are supplied by the launch configuration.' : 'Paths are read on this computer.';
  el('source-hint').hidden = state.dataset.name === 'demo' && !state.dataset.configured;
  const input = el<HTMLInputElement>('frame-number'); input.value = String(state.frame_index + 1); input.max = String(Math.max(1,state.frames.length));
  el('frame-total').textContent = `/ ${state.frames.length || '—'}`;
  const slider = el<HTMLInputElement>('frame-slider');
  slider.min = '0'; slider.max = String(Math.max(0,state.frames.length-1)); slider.value = String(state.frame_index); slider.disabled = busy || state.frames.length < 2;
  slider.setAttribute('aria-valuetext', `${state.frame_index + 1} of ${state.frames.length}`);
  color.querySelector<HTMLOptionElement>('[value="rgb"]')!.disabled = !supports('rgb_coloring');
  if (color.value === 'rgb' && !supports('rgb_coloring')) color.value = 'gray';
  raw.checked = state.display.raw;
  el('raw-hint').hidden = supportsRaw();
  el('raw-hint').textContent = supportsRaw() ? 'Show points at their acquisition-time positions.' : supports('acquisition_geometry') ? 'Only part of this scan has acquisition geometry.' : 'Acquisition geometry is unavailable for this dataset.';
  const count = packet?.header.count ?? 0;
  el('rgb-coverage').textContent = packet ? `Camera RGB: ${packet.header.rgb_count.toLocaleString()} / ${count.toLocaleString()} points. Unmatched points use gray.` : '';
  el('empty-state').hidden = Boolean(packet);
  el('frame-stats').hidden = !packet;
  if (packet) {
    const total = document.createElement('b'); total.textContent = count.toLocaleString();
    el('frame-stats').replaceChildren(total, ' points');
  }
  renderClasses(); updatePointStyle(); updateBoxStyle(); refreshCamera(); syncDisabled();
}
async function execute(body: JobRequest) {
  if (busy || !session || disposed) return;
  const token = ++generation;
  const priorState = state;
  el('error').hidden = true; setBusy(true, body.op === 'connect' ? 'Opening dataset…' : body.op === 'prepare' ? 'Preparing points…' : 'Loading frame…');
  try {
    const next = await runJob(session, body, stage => {if (token === generation) setStatus(stage.replace(/[_-]/g,' '));});
    if (disposed || token !== generation) return;
    let nextPacket: FramePacket | null = null;
    if (next.has_frame) {setStatus('Transferring point cloud…'); nextPacket = await fetchFrame(session, next.revision);}
    if (disposed || token !== generation) return;
    const reset = !priorState || priorState.dataset.name !== next.dataset.name || priorState.dataset.root !== next.dataset.root || priorState.scene_id !== next.scene_id || body.op === 'connect';
    if (!priorState || priorState.dataset.name !== next.dataset.name || priorState.dataset.root !== next.dataset.root || body.op === 'connect') classStyles.clear();
    for (const c of next.metadata.classes) if (!classStyles.has(c.id)) classStyles.set(c.id,{visible:true,color:rgbHex(c.color)});
    if (nextPacket) viewer!.setFrame(nextPacket, next.metadata.classes, classStyles, reset);
    else viewer!.clearFrame();
    state = next; packet = nextPacket;
    cameraDirty = false;
    cameraMessage('', false);
    if (body.op === 'prepare' && body.rgb !== undefined) color.value = body.rgb ? 'rgb' : color.value;
    renderState();
    if (!nextPacket) setStatus('No frames are available for this selection.');
    else el('announcements').textContent = `Frame ${state.frame_index + 1} of ${state.frames.length} loaded.`;
  } catch (issue) {
    if (disposed || token !== generation) return;
    error(issue);
    color.value = committedColor;
    if (state) {raw.checked = state.display.raw; renderState();}
    setStatus(packet ? 'Previous frame retained.' : 'Ready to load a dataset.');
  } finally {if (!disposed && token === generation) setBusy(false);}
}
function desiredDisplay() {return {rgb:color.value === 'rgb', raw:raw.checked};}
function selectFrame(index: number) {if (state && dataset.value !== state.dataset.name) return; if (!state || !Number.isInteger(index) || index < 0 || index >= state.frames.length) {renderState(); return;} if (index !== state.frame_index) void execute({op:'select', frame_index:index, ...desiredDisplay()});}
function connect() {if (dataset.value === 'demo' && color.value === 'rgb') color.value = 'gray'; raw.checked = false; void execute({op:'connect', dataset:dataset.value, root:root.value, version:version.value, config_path:info?.initial.configured ? info.initial.config_path : null, split:dataset.value === state?.dataset.name ? split.value : !state && dataset.value === info?.initial.dataset ? info.initial.split : 'all', ...desiredDisplay()});}
function bindNumericControl(rangeId: string, numberId: string, update: () => void) {
  const range = el<HTMLInputElement>(rangeId), number = el<HTMLInputElement>(numberId);
  range.addEventListener('input', update);
  number.addEventListener('change', () => {
    if (!Number.isFinite(number.valueAsNumber) || !number.validity.valid) {
      number.value = range.value;
      return;
    }
    range.value = number.value;
    update();
  });
}

el('source-form').addEventListener('submit', e => {e.preventDefault(); connect();});
dataset.addEventListener('change', () => {
  if (dataset.value === 'demo') {root.value = ''; version.value = ''; connect();}
  else {
    root.value = dataset.value === info?.initial.dataset ? info.initial.root : '';
    version.value = dataset.value === info?.initial.dataset ? info.initial.version : '';
    el<HTMLDetailsElement>('source-section').open = true; syncDisabled(); root.focus();
    el('source-hint').hidden = false;
    el('source-hint').textContent = `Enter your ${dataset.value === 'nuscenes' ? 'nuScenes' : dataset.value} dataset root, then load the dataset.`;
  }
});
split.addEventListener('change', () => void execute({op:'select',split:split.value,...desiredDisplay()}));
scene.addEventListener('change', () => void execute({op:'select',scene_id:scene.value,frame_index:0,...desiredDisplay()}));
el('previous').addEventListener('click', () => selectFrame((state?.frame_index ?? 0)-1));
el('next').addEventListener('click', () => selectFrame((state?.frame_index ?? 0)+1));
el<HTMLInputElement>('frame-number').addEventListener('change', e => selectFrame(Number((e.target as HTMLInputElement).value)-1));
el<HTMLInputElement>('frame-slider').addEventListener('change', e => selectFrame(Number((e.target as HTMLInputElement).value)));
el<HTMLInputElement>('frame-slider').addEventListener('input', e => {
  const slider = e.target as HTMLInputElement, frame = Number(slider.value) + 1;
  el<HTMLInputElement>('frame-number').value = String(frame);
  slider.setAttribute('aria-valuetext', `${frame} of ${state?.frames.length ?? 0}`);
});
bindNumericControl('point-size', 'point-size-number', updatePointStyle);
bindNumericControl('max-distance', 'max-distance-number', updatePointStyle);
bindNumericControl('box-width', 'box-width-number', updateBoxStyle);
color.addEventListener('change', () => {
  if (color.value === 'rgb' && !state?.display.rgb) void execute({op:'prepare',...desiredDisplay()});
  else updatePointStyle();
});
raw.addEventListener('change', () => void execute({op:'prepare',...desiredDisplay()}));
for (const [id,visible] of [['show-all',true],['hide-all',false]] as const) el(id).addEventListener('click', () => {for (const style of classStyles.values()) style.visible = visible; renderClasses(); updateBoxStyle();});
el<HTMLInputElement>('axes').addEventListener('change', e => viewer?.showAxes((e.target as HTMLInputElement).checked));
el('reset-camera').addEventListener('click', () => {cameraDirty=false; viewer?.resetCamera(); refreshCamera(); cameraMessage('', false);});
cameraJson.addEventListener('input', () => {cameraDirty=true; cameraMessage('Apply to move the camera to this view.', false);});
el('apply-camera').addEventListener('click', () => {
  try {
    viewer!.applyCamera(parseCamera(cameraJson.value));
    cameraDirty=false; refreshCamera(); cameraMessage('Camera view applied.', false);
  } catch (issue) {
    cameraMessage(issue instanceof Error ? issue.message : String(issue), true);
  }
});
el('copy-camera').addEventListener('click', async () => {
  if (!viewer) return;
  refreshCamera();
  const invalid = cameraJson.getAttribute('aria-invalid') === 'true';
  try {await navigator.clipboard.writeText(cameraJson.value); cameraMessage('Camera view copied.', invalid);}
  catch {cameraJson.focus(); cameraJson.select(); cameraMessage('Select and copy the camera JSON with your keyboard.', invalid);}
});
el('paste-camera').addEventListener('click', async () => {
  let text: string;
  try {text = await navigator.clipboard.readText();}
  catch {cameraJson.focus(); cameraMessage('Paste camera JSON into the field, then Apply.'); return;}
  try {parseCamera(text);}
  catch (issue) {cameraMessage(`Clipboard: ${issue instanceof Error ? issue.message : String(issue)}`); return;}
  cameraJson.value=text; cameraDirty=true; cameraMessage('View pasted · Apply to move the camera.', false);
});
let inspectorScroll = {panel: 0, content: 0};
function showInspector(visible: boolean) {
  const panel = el('sidebar'), content = el('sidebar-content'), toggle = el<HTMLButtonElement>('menu-toggle');
  if (!visible && !content.hidden) {
    inspectorScroll = {panel: panel.scrollTop, content: content.scrollTop};
    if (content.contains(document.activeElement)) toggle.focus({preventScroll: true});
  }
  content.inert = !visible;
  content.hidden = !visible;
  panel.classList.toggle('collapsed', !visible);
  toggle.setAttribute('aria-expanded', String(visible));
  const label = visible ? 'Minimize scene controls' : 'Expand scene controls';
  toggle.setAttribute('aria-label', label); toggle.title = label;
  if (visible) {panel.scrollTop = inspectorScroll.panel; content.scrollTop = inspectorScroll.content;}
}
const narrowLayout = window.matchMedia('(max-width: 620px)');
const cameraPanel = el<HTMLDetailsElement>('camera-panel');
cameraPanel.open = !narrowLayout.matches;
el('menu-toggle').addEventListener('click', () => {
  const visible = el('sidebar-content').hidden;
  if (visible && narrowLayout.matches) cameraPanel.open = false;
  showInspector(visible);
});
cameraPanel.addEventListener('toggle', () => {
  if (cameraPanel.open && narrowLayout.matches) showInspector(false);
});
narrowLayout.addEventListener('change', e => {cameraPanel.open = !e.matches;});
el('dismiss-error').addEventListener('click', () => {el('error').hidden=true; if (packet && !busy) setStatus('');});
el('try-demo').addEventListener('click', () => {dataset.value='demo'; root.value=''; version.value=''; connect();});
document.addEventListener('keydown', e => {
  if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement || e.ctrlKey || e.altKey || e.metaKey) return;
  if (e.key === 'ArrowRight') {e.preventDefault(); selectFrame((state?.frame_index??0)+1);}
  if (e.key === 'ArrowLeft') {e.preventDefault(); selectFrame((state?.frame_index??0)-1);}
});
window.addEventListener('pagehide', () => {disposed=true; generation++; if (cameraSync) clearTimeout(cameraSync); viewer?.dispose(); if(session) void closeSession(session);}, {once:true});
async function start() {
  try {
    info = await bootstrap();
    option(dataset,info.datasets.map(value=>({value,text:value==='demo'?'Synthetic demo':value==='nuscenes'?'nuScenes':value==='waymo'?'Waymo':value})),info.initial.dataset);
    root.value=info.initial.root||''; version.value=info.initial.version||'';
    viewer=new Viewer(el('viewport'),info.palettes); viewer.onCameraChange=queueCameraRefresh;
    session=(await createSession()).id; syncDisabled(); el('sidebar').classList.remove('is-loading'); setStatus('Ready to load a dataset.');
    if(info.initial.autoload) await execute({op:'connect',dataset:info.initial.dataset,root:info.initial.root,version:info.initial.version,config_path:info.initial.config_path,split:info.initial.split,scene_id:info.initial.scene||undefined,frame_index:info.initial.frame,...desiredDisplay()});
  } catch(issue) {el('sidebar').classList.remove('is-loading'); error(issue); setStatus('Unable to connect to the local viewer.');}
}
void start();
