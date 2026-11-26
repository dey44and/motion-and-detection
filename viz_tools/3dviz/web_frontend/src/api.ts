import type { Bootstrap, JobRequest, ViewerState } from './types';
import { decodeFrame } from './packet';
async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, {...options, headers: {'Content-Type': 'application/json', ...options.headers}});
  if (!response.ok) {
    let message = `Request failed (${response.status}).`;
    try {const body = await response.json(); message = body.error || body.detail || message;} catch { /* HTTP status remains meaningful. */ }
    throw new Error(typeof message === 'string' ? message : JSON.stringify(message));
  }
  return response.json() as Promise<T>;
}
export const bootstrap = () => request<Bootstrap>('/api/bootstrap');
export const createSession = () => request<{id: string}>('/api/sessions', {method: 'POST', body: '{}'});
export const closeSession = (id: string) => fetch(`/api/sessions/${encodeURIComponent(id)}`, {method: 'DELETE', keepalive: true});
export async function runJob(session: string, body: JobRequest, onStage: (stage: string) => void): Promise<ViewerState> {
  const root = `/api/sessions/${encodeURIComponent(session)}`;
  const {id} = await request<{id: string}>(`${root}/jobs`, {method: 'POST', body: JSON.stringify(body)});
  for (;;) {
    const result = await request<{status: string; stage: string; state?: ViewerState; error?: string}>(`${root}/jobs/${encodeURIComponent(id)}`);
    onStage(result.stage || 'Loading');
    if (result.status === 'error') throw new Error(result.error || 'The dataset could not be loaded.');
    if (result.status === 'done') {
      if (!result.state) throw new Error('The server returned no viewer state.');
      return result.state;
    }
    await new Promise(resolve => setTimeout(resolve, 150));
  }
}
export async function fetchFrame(session: string, revision: number) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(session)}/frame.bin?revision=${revision}`);
  if (!response.ok) {
    let message = `Point-cloud transfer failed (${response.status}).`;
    try {const body = await response.json(); message = body.error || body.detail || message;} catch { /* Use HTTP status. */ }
    throw new Error(message);
  }
  const packet = decodeFrame(await response.arrayBuffer());
  if (packet.header.revision !== revision) throw new Error('The scene changed during the point-cloud transfer. Please retry.');
  return packet;
}

export async function fetchSegmentation(session: string, revision: number, generation: number, count: number) {
  const response = await fetch(`/api/sessions/${encodeURIComponent(session)}/segmentation.bin?revision=${revision}&generation=${generation}`);
  if (!response.ok) throw new Error(`Point labels could not be transferred (${response.status}).`);
  const labels = new Uint8Array(await response.arrayBuffer());
  if (labels.length !== count || labels.some(label => label > 16)) throw new Error('The point labels do not match this LiDAR scan.');
  return labels;
}

export async function fetchCameraPreviews(session: string, revision: number, signal: AbortSignal) {
  const previews = await request<Record<string, string>>(
    `/api/sessions/${encodeURIComponent(session)}/camera-previews?revision=${revision}`, {signal},
  );
  return Object.fromEntries(Object.entries(previews).map(([id, jpeg]) => [id, `data:image/jpeg;base64,${jpeg}`]));
}

function cameraUrl(session: string, revision: number, camera: string, resource: 'camera-image' | 'camera-points') {
  return `/api/sessions/${encodeURIComponent(session)}/${resource}/${encodeURIComponent(camera)}?revision=${revision}`;
}

export function cameraImageUrl(session: string, revision: number, camera: string) {
  return cameraUrl(session, revision, camera, 'camera-image');
}

export async function fetchCameraPoints(session: string, revision: number, camera: string, count: number, rows: number, signal: AbortSignal) {
  const response = await fetch(cameraUrl(session, revision, camera, 'camera-points'), {signal});
  if (!response.ok) throw new Error(`Projected points could not be transferred (${response.status}).`);
  const buffer = await response.arrayBuffer();
  if (buffer.byteLength !== count * 12) throw new Error('The projected point count does not match the camera response.');
  const points = new DataView(buffer);
  for (let offset = 0; offset < buffer.byteLength; offset += 12) {
    if (!Number.isFinite(points.getFloat32(offset, true)) || !Number.isFinite(points.getFloat32(offset + 4, true)) || points.getUint32(offset + 8, true) >= rows) {
      throw new Error('The projected points do not match the current LiDAR scan.');
    }
  }
  return points;
}
