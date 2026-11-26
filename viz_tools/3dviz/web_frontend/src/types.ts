export type Vec3 = [number, number, number];
export type Scheme = 'gray' | 'viridis' | 'rgb' | 'concerto' | 'semantic';
export type SegmentationMode = 'off' | 'ground_truth' | 'prediction';
export type SegmentationFormat = 'auto' | 'official' | 'sonata';
export type NmsMode = 'off' | 'class' | 'agnostic' | 'groups';
export interface DetectionFilters {
  score_threshold: number;
  per_class_scores: Record<string, number>;
  min_distance: number;
  max_distance: number | null;
  min_height: number | null;
  max_height: number | null;
  pre_nms_limit: number | null;
  max_detections: number | null;
  nms_mode: NmsMode;
  nms_iou_threshold: number;
  nms_groups: string[][];
}
export interface DetectionState {
  available: boolean;
  status: 'detached' | 'ready' | 'missing' | 'empty' | 'filtered' | 'invalid';
  message: string;
  source: {name: string; path: string | null; format: string; stage: string} | null;
  filters: DetectionFilters | null;
  counts: {loaded: number; score: number; spatial: number; pre_nms: number; nms: number; output: number} | null;
  classes: DatasetClass[];
  boxes: (Box & {score: number})[];
}
export interface SemanticClass {id: number; name: string; color: Vec3; count: number}
export interface SegmentationState {
  available: boolean;
  mode: SegmentationMode;
  status: 'off' | 'ready' | 'missing' | 'invalid' | 'unsupported';
  message: string;
  path: string | null;
  format: string;
  scan_token: string | null;
  classes: SemanticClass[];
  generation: number;
}
export interface Bootstrap {
  datasets: string[];
  prediction_formats: string[];
  initial: {
    dataset: string; root: string; version: string; split: string; scene?: string | null; frame: number;
    autoload: boolean; config_path?: string | null; configured?: boolean;
    prediction_path: string | null; prediction_format: string;
  };
  palettes: Record<'gray' | 'viridis' | 'concerto', number[][]>;
}
export interface DatasetClass {id: string; name: string; group: string; color: Vec3}
export interface ViewerState {
  dataset: {name: string; root: string; version: string; configured: boolean};
  metadata: {display_name: string; capabilities: string[]; feature_coverage: Record<string, string>; classes: DatasetClass[]; splits: string[]};
  split: string;
  scenes: {id: string; name: string; description: string; count: number | null}[];
  scene_id: string;
  frames: {id: string; index: number; timestamp_us: string | null}[];
  frame_index: number;
  revision: number;
  has_frame: boolean;
  display: {rgb: boolean; raw: boolean};
  detection?: DetectionState;
  segmentation?: SegmentationState;
}
export interface Box {id: string; class_id: string; group: string; corners: Vec3[]}
export interface FrameHeader {
  protocol: number; revision: number; count: number; origin: Vec3;
  bounds: {min: Vec3; max: Vec3};
  frame: {id: string; scene_id: string; index: number; timestamp_us: string | null};
  source_id: string; reference_frame_id: string; annotation_status: string;
  boxes: Box[]; rgb_count: number; display: {rgb: boolean; raw: boolean};
  buffers: {name: string; dtype: string; shape: number[]; offset: number; length: number}[];
}
export interface FramePacket {
  header: FrameHeader; positions: Float32Array; distance: Float32Array;
  rgb: Uint8Array | null; rgbMask: Uint8Array | null;
}
export interface CameraView {position: Vec3; rotation: Vec3; pivot: Vec3; fov: number; near: number; far: number}
export interface FrameJob {
  op: 'connect' | 'select' | 'prepare'; dataset?: string; root?: string;
  version?: string; config_path?: string | null; split?: string; scene_id?: string;
  frame_index?: number; rgb?: boolean; raw?: boolean;
}
export type ReviewJob = {
  op: 'detection'; prediction_path?: string; prediction_format?: string; filters?: Partial<DetectionFilters>;
} | {
  op: 'segmentation'; segmentation_mode?: SegmentationMode; segmentation_path?: string; segmentation_format?: SegmentationFormat;
};
export type JobRequest = FrameJob | (ReviewJob & {revision: number});
