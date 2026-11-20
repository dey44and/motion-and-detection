export type Vec3 = [number, number, number];
export type Scheme = 'gray' | 'viridis' | 'rgb' | 'concerto';
export interface Bootstrap {
  datasets: string[];
  initial: {dataset: string; root: string; version: string; split: string; scene?: string | null; frame: number; autoload: boolean; config_path?: string | null; configured?: boolean};
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
export interface JobRequest {
  op: 'connect' | 'select' | 'prepare'; dataset?: string; root?: string;
  version?: string; config_path?: string | null; split?: string; scene_id?: string;
  frame_index?: number; rgb?: boolean; raw?: boolean;
}
