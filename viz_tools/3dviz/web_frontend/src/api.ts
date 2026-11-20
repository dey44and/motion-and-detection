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
