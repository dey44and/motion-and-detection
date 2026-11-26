import type { DetectionFilters, DetectionState, NmsMode, ReviewJob, SegmentationFormat, SegmentationMode, SegmentationState, Vec3 } from './types';
import { rgbHex } from './viewer';

export const predictionSources = `
  <form id="detection-source" class="attachment-form">
    <label class="field-label" for="prediction-path">Detection export</label>
    <div class="input-wrap"><input id="prediction-path" autocomplete="off" spellcheck="false" placeholder="/path/to/results.json or .pkl" required></div>
    <div class="attachment-actions">
      <select id="prediction-format" aria-label="Detection export format"><option value="auto">Detect format</option></select>
      <button class="button secondary" type="submit">Attach</button><button id="detach-predictions" class="button quiet" type="button" disabled>Detach</button>
    </div>
    <p id="detection-source-error" class="hint" role="status" hidden></p>
  </form>
  <form id="segmentation-source" class="attachment-form">
    <label class="field-label" for="segmentation-path">LiDAR segmentation export</label>
    <div class="input-wrap"><input id="segmentation-path" autocomplete="off" spellcheck="false" placeholder="/path/to/labels or .pkl" required></div>
    <div class="attachment-actions">
      <select id="segmentation-format" aria-label="LiDAR segmentation export format"><option value="auto">Detect format</option><option value="official">nuScenes lidarseg</option><option value="sonata">Sonata export</option></select>
      <button class="button secondary" type="submit">Attach</button><button id="detach-segmentation" class="button quiet" type="button" disabled>Detach</button>
    </div>
  </form>`;

export const detectionPanel = `
  <div class="review-section">
    <div class="review-run" id="detection-run" hidden><strong id="detection-run-name"></strong><span id="detection-run-stage"></span></div>
    <p id="detection-status" class="review-status" role="status"></p>
    <fieldset id="detection-controls" class="review-controls" disabled>
      <div class="control-heading"><label for="confidence">Minimum confidence</label><output id="confidence-value" for="confidence">0.10</output></div>
      <input id="confidence" type="range" min="0" max="1" step="0.01" value="0.1">
      <dl id="detection-counts" class="stage-counts" aria-label="Detection filter counts" hidden>
        <div><dt>Loaded</dt><dd data-count="loaded"></dd></div><div><dt>Confidence</dt><dd data-count="score"></dd></div><div><dt>Spatial</dt><dd data-count="spatial"></dd></div>
        <div><dt>Candidates</dt><dd data-count="pre_nms"></dd></div><div><dt>After NMS</dt><dd data-count="nms"></dd></div><div><dt>Retained</dt><dd data-count="output"></dd></div>
      </dl>
      <details id="advanced-detection">
        <summary><span>Advanced filters</span></summary>
        <form id="detection-filters">
          <div class="filter-grid">
            <label class="compact-field">Min distance (m)<input id="min-distance" type="number" min="0" step="any" value="0" required></label>
            <label class="compact-field">Max distance (m)<input id="max-box-distance" type="number" min="0" step="any" placeholder="Any"></label>
            <label class="compact-field">Min height (m)<input id="min-height" type="number" step="any" placeholder="Any"></label>
            <label class="compact-field">Max height (m)<input id="max-height" type="number" step="any" placeholder="Any"></label>
            <label class="compact-field">Candidate limit<input id="candidate-limit" type="number" min="0" step="1" placeholder="Unlimited"></label>
            <label class="compact-field">Output limit<input id="output-limit" type="number" min="0" step="1" placeholder="Unlimited"></label>
            <label class="compact-field">BEV NMS<select id="nms-mode"><option value="off">Off</option><option value="class">Per class</option><option value="agnostic">Class agnostic</option><option value="groups">CenterPoint groups</option></select></label>
            <label class="compact-field">IoU threshold<input id="nms-iou" type="number" min="0" max="1" step="any" value="0.2" required></label>
          </div>
          <details class="class-overrides"><summary><span>Per-class confidence</span></summary><div id="class-overrides"></div></details>
          <div class="review-actions"><button class="button primary" type="submit">Apply filters</button><button id="reset-filters" class="button quiet" type="button">Reset</button></div>
        </form>
      </details>
      <div class="review-actions"><button id="load-preset" class="button secondary" type="button">Load preset</button><button id="save-preset" class="button secondary" type="button">Save preset</button><input id="preset-file" type="file" accept="application/json,.json" hidden></div>
    </fieldset>
  </div>`;

export const segmentationPanel = `
  <div class="review-section">
    <div class="field-label" id="label-source-title">Point labels</div>
    <div class="segmentation-options" role="group" aria-labelledby="label-source-title">
      <button id="labels-ground-truth" class="button secondary" type="button" aria-pressed="false" disabled>Ground truth</button>
      <button id="labels-prediction" class="button secondary" type="button" aria-pressed="false" disabled>Prediction</button>
    </div>
    <p id="segmentation-status" class="review-status" role="status"></p>
    <p id="segmentation-run" class="hint" hidden></p>
    <div id="semantic-legend" hidden>
      <div class="semantic-heading"><strong>nuScenes evaluation classes</strong><span>Points</span></div>
      <div id="semantic-classes" class="classes" aria-label="Visible semantic classes"></div>
    </div>
  </div>`;

const modeLabels: Record<string, string> = {pre_nms: 'Pre-NMS', post_nms: 'Post-NMS', unknown: 'Stage unknown'};
const formatLabels = new Map([['auto', 'Detect format'], ['nuscenes_json', 'nuScenes JSON'], ['pcdet_pickle', 'OpenPCDet pickle']]);
const filterFields = {
  min_distance: 'min-distance', max_distance: 'max-box-distance', min_height: 'min-height', max_height: 'max-height',
  pre_nms_limit: 'candidate-limit', max_detections: 'output-limit', nms_iou_threshold: 'nms-iou',
} as const;
const defaultFilters: DetectionFilters = {
  score_threshold: .1, per_class_scores: {}, min_distance: 0, max_distance: null,
  min_height: null, max_height: null, pre_nms_limit: null, max_detections: null,
  nms_mode: 'off', nms_iou_threshold: .2,
  nms_groups: [['car'], ['truck', 'construction_vehicle'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']],
};

export interface ReviewAppearance {gt_color: Vec3; detection_color: Vec3; line_width: number}
interface ReviewActions {
  run: (body: ReviewJob) => Promise<boolean>;
  appearance: () => ReviewAppearance;
  applyAppearance: (appearance: ReviewAppearance) => void;
  setClassVisibility: (hidden: ReadonlySet<number>) => void;
  error: (message: unknown) => void;
}
function el<T extends HTMLElement = HTMLElement>(id: string) {return document.getElementById(id) as T;}
function field(id: string) {return el<HTMLInputElement>(id);}
function disabled(id: string, value: boolean) {el<HTMLButtonElement>(id).disabled = value;}
function isObject(value: unknown): value is Record<string, unknown> {return typeof value === 'object' && value !== null && !Array.isArray(value);}
function colorValue(value: unknown): value is Vec3 {return Array.isArray(value) && value.length === 3 && value.every(c => typeof c === 'number' && Number.isFinite(c) && c >= 0 && c <= 1);}

export function wireReviewControls(actions: ReviewActions) {
  let detection: DetectionState | undefined;
  let segmentation: SegmentationState | undefined;
  let busy = false;
  let predictionFormats: string[] = [];
  let launchPrediction: {path: string | null; format: string} | undefined;
  const hidden = new Set<number>();
  const score = field('confidence');
  const predictionFormat = el<HTMLSelectElement>('prediction-format');
  const filterForm = el<HTMLFormElement>('detection-filters');

  function showScore() {
    const threshold = Number(score.value);
    el('confidence-value').textContent = threshold === Math.round(threshold * 100) / 100 ? threshold.toFixed(2) : String(threshold);
  }
  function renderAvailability() {
    const rootOpen = !!detection?.available;
    el<HTMLFieldSetElement>('detection-controls').disabled = busy || !rootOpen;
    for (const id of ['prediction-path', 'prediction-format', 'detach-predictions']) disabled(id, busy || !rootOpen || (id === 'detach-predictions' && !detection?.source?.path));
    el<HTMLFormElement>('detection-source').querySelector<HTMLButtonElement>('[type="submit"]')!.disabled = busy || !rootOpen;
    for (const id of ['segmentation-path', 'segmentation-format', 'detach-segmentation']) disabled(id, busy || !segmentation?.available || (id === 'detach-segmentation' && !segmentation.path));
    el<HTMLFormElement>('segmentation-source').querySelector<HTMLButtonElement>('[type="submit"]')!.disabled = busy || !segmentation?.available;
    disabled('labels-ground-truth', busy || !segmentation?.available);
    disabled('labels-prediction', busy || !segmentation?.available);
  }
  function renderFilters() {
    const filters = detection?.filters ?? defaultFilters;
    score.step = filters.score_threshold === Math.round(filters.score_threshold * 100) / 100 ? '0.01' : 'any';
    score.value = String(filters.score_threshold);
    showScore();
    for (const [key, id] of Object.entries(filterFields)) field(id).value = String(filters[key as keyof typeof filterFields] ?? '');
    el<HTMLSelectElement>('nms-mode').value = filters.nms_mode;
    const scores = el('class-overrides');
    scores.replaceChildren();
    const classes = new Map(detection?.classes.map(item => [item.id, item.name]));
    for (const id of Object.keys(filters.per_class_scores)) if (!classes.has(id)) classes.set(id, id);
    for (const [id, name] of classes) {
      const row = document.createElement('label'); row.className = 'override-row';
      const label = document.createElement('span'); label.textContent = name;
      const input = document.createElement('input');
      input.type = 'number'; input.min = '0'; input.max = '1'; input.step = 'any'; input.placeholder = 'Global';
      input.value = String(filters.per_class_scores[id] ?? ''); input.dataset.classId = id;
      row.append(label, input); scores.append(row);
    }
    if (!classes.size) scores.textContent = 'Class thresholds appear with a dataset or attached detection export.';
  }
  function showPredictionFormat(selected: string) {
    const values = [...new Set(['auto', ...predictionFormats, selected])];
    predictionFormat.replaceChildren(...values.map(value => new Option(formatLabels.get(value) ?? value, value)));
    predictionFormat.value = selected;
  }
  function renderDetection(next?: DetectionState) {
    const source = next?.source;
    const previous = detection?.source;
    const sourceError = next?.status === 'invalid' && !source ? next.message : '';
    if (source?.path && (previous?.path !== source.path || previous.format !== source.format)) {
      field('prediction-path').value = source.path;
      showPredictionFormat(source.format);
    } else if (sourceError && !detection?.source && !field('prediction-path').value && launchPrediction?.path) {
      field('prediction-path').value = launchPrediction.path;
      showPredictionFormat(launchPrediction.format);
    } else if (previous?.path && !source?.path) {
      field('prediction-path').value = sourceError ? launchPrediction?.path ?? '' : '';
      showPredictionFormat(sourceError ? launchPrediction?.format ?? 'auto' : 'auto');
    }
    const classesChanged = detection?.classes.map(c => c.id).join('\n') !== next?.classes.map(c => c.id).join('\n');
    const filtersChanged = JSON.stringify(detection?.filters) !== JSON.stringify(next?.filters);
    detection = next;
    el('detection-run').hidden = !next?.source;
    el('detection-run-name').textContent = next?.source?.name ?? '';
    el('detection-run-stage').textContent = modeLabels[next?.source?.stage ?? ''] ?? '';
    el('detection-status').textContent = next?.message || (next?.status === 'ready' ? '' : 'Attach a detection export in Sources.');
    el('detection-source-error').textContent = sourceError;
    el('detection-source-error').hidden = !sourceError;
    const counts = next?.counts;
    el('detection-counts').hidden = !counts;
    if (counts) for (const output of el('detection-counts').querySelectorAll<HTMLElement>('[data-count]')) {
      output.textContent = counts[output.dataset.count as keyof typeof counts].toLocaleString();
    }
    if (filtersChanged || classesChanged) renderFilters();
    renderAvailability();
  }
  function renderSegmentation(next?: SegmentationState, failure = '') {
    if (segmentation?.path !== next?.path) {
      field('segmentation-path').value = next?.path ?? '';
      el<HTMLSelectElement>('segmentation-format').value = next?.format === 'official' || next?.format === 'sonata' ? next.format : 'auto';
    }
    segmentation = next;
    for (const [id, mode] of [['labels-ground-truth', 'ground_truth'], ['labels-prediction', 'prediction']]) {
      el(id).setAttribute('aria-pressed', String(mode === next?.mode));
    }
    el('segmentation-run').hidden = next?.mode !== 'prediction' || !next.path;
    el('segmentation-run').textContent = next?.path?.split('/').pop() ?? '';
    const message = failure || next?.message || (
      next?.status === 'ready' ? '' : next?.available ? 'Choose a label source to color this scan.' : 'LiDAR segmentation is unavailable for this dataset.'
    );
    el('segmentation-status').textContent = message;
    el('segmentation-status').hidden = !message;
    const list = el('semantic-classes'); list.replaceChildren();
    const ready = next?.status === 'ready' && !failure;
    el('semantic-legend').hidden = !ready;
    if (ready) for (const item of next.classes) {
      const row = document.createElement('label'); row.className = 'class-label semantic-class';
      const check = document.createElement('input'); check.type = 'checkbox'; check.checked = !hidden.has(item.id);
      const name = item.name.replaceAll('_', ' ');
      check.setAttribute('aria-label', `Show ${name} points`);
      check.addEventListener('change', () => {if (check.checked) hidden.delete(item.id); else hidden.add(item.id); actions.setClassVisibility(hidden);});
      const swatch = document.createElement('i'); swatch.className = 'semantic-swatch'; swatch.style.background = rgbHex(item.color);
      const text = document.createElement('span'); text.textContent = name;
      const count = document.createElement('span'); count.className = 'class-count'; count.textContent = item.count.toLocaleString();
      row.append(check, swatch, text, count); list.append(row);
    }
    actions.setClassVisibility(hidden);
    renderAvailability();
  }
  function submitDetection(filters: Partial<DetectionFilters>) {return actions.run({op: 'detection', filters});}

  score.addEventListener('input', showScore);
  score.addEventListener('change', async () => {
    if (!await submitDetection({score_threshold: Number(score.value)})) renderFilters();
  });
  filterForm.addEventListener('submit', async event => {
    event.preventDefault();
    const filters: Partial<DetectionFilters> = {nms_mode: el<HTMLSelectElement>('nms-mode').value as NmsMode, per_class_scores: {}};
    for (const [key, id] of Object.entries(filterFields)) Object.assign(filters, {[key]: field(id).value === '' ? null : field(id).valueAsNumber});
    for (const input of el('class-overrides').querySelectorAll<HTMLInputElement>('input')) {
      if (input.value !== '') Object.defineProperty(filters.per_class_scores!, input.dataset.classId!, {value: input.valueAsNumber, enumerable: true});
    }
    await submitDetection(filters);
  });
  el('reset-filters').addEventListener('click', () => void submitDetection({...defaultFilters, score_threshold: 0}));
  el('detection-source').addEventListener('submit', event => {
    event.preventDefault();
    const path = field('prediction-path').value.trim();
    if (path) void actions.run({op: 'detection', prediction_path: path, prediction_format: predictionFormat.value});
  });
  el('detach-predictions').addEventListener('click', () => void actions.run({op: 'detection', prediction_path: ''}));
  el('segmentation-source').addEventListener('submit', event => {
    event.preventDefault();
    const path = field('segmentation-path').value.trim();
    if (path) void actions.run({op: 'segmentation', segmentation_mode: 'prediction', segmentation_path: path, segmentation_format: el<HTMLSelectElement>('segmentation-format').value as SegmentationFormat});
  });
  el('detach-segmentation').addEventListener('click', () => void actions.run({op: 'segmentation', segmentation_path: ''}));
  const chooseMode = (segmentation_mode: SegmentationMode) => void actions.run({op: 'segmentation', segmentation_mode});
  el('labels-ground-truth').addEventListener('click', () => chooseMode('ground_truth'));
  el('labels-prediction').addEventListener('click', () => chooseMode('prediction'));

  el('load-preset').addEventListener('click', () => field('preset-file').click());
  field('preset-file').addEventListener('change', async () => {
    const file = field('preset-file').files?.[0];
    field('preset-file').value = '';
    if (!file) return;
    try {
      const preset: unknown = JSON.parse(await file.text());
      if (!isObject(preset) || preset.schema_version !== 1 || !isObject(preset.filters)) throw new Error('Choose a version 1 detection preset JSON file.');
      const appearance = preset.appearance;
      if (appearance !== undefined && !isObject(appearance)) throw new Error('Preset appearance must be a JSON object.');
      const defaults = actions.appearance();
      const gt_color = appearance?.gt_color ?? defaults.gt_color;
      const detection_color = appearance?.detection_color ?? defaults.detection_color;
      const line_width = appearance?.line_width ?? defaults.line_width;
      if (!colorValue(gt_color) || !colorValue(detection_color) || typeof line_width !== 'number' || !Number.isFinite(line_width) || line_width < 1 || line_width > 10) throw new Error('Preset colors or line width are invalid.');
      if (await submitDetection({...defaultFilters, ...preset.filters})) actions.applyAppearance({gt_color, detection_color, line_width});
    } catch (error) {actions.error(error);}
  });
  el('save-preset').addEventListener('click', () => {
    const payload = {schema_version: 1, filters: detection?.filters ?? defaultFilters, appearance: actions.appearance()};
    const href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], {type: 'application/json'}));
    const link = document.createElement('a');
    link.href = href; link.download = 'detection-preset.json'; link.click();
    setTimeout(() => URL.revokeObjectURL(href), 1000);
  });
  renderAvailability();
  return {
    renderDetection, renderSegmentation,
    setDetectionDefaults(formats: string[], path: string | null, format: string) {
      predictionFormats = formats;
      launchPrediction = {path, format};
      field('prediction-path').value = path ?? '';
      showPredictionFormat(format);
    },
    setBusy(value: boolean) {busy = value; renderAvailability();},
    resetClasses() {hidden.clear();},
  };
}
