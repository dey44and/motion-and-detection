import { Euler, Matrix4, Quaternion, Vector3 } from 'three';
import type { CameraView, Vec3 } from './types';

const degrees = 180 / Math.PI;
export function rotationAngles(quaternion: Quaternion): Vec3 {
  const m = new Matrix4().makeRotationFromQuaternion(quaternion).elements;
  const horizontal = Math.hypot(m[0], m[1]);
  const pitch = Math.atan2(-m[2], horizontal);
  const roll = horizontal > 1e-6 ? Math.atan2(m[6], m[10]) : Math.atan2(-m[9], m[5]);
  const yaw = horizontal > 1e-6 ? Math.atan2(m[1], m[0]) : 0;
  return [roll * degrees, pitch * degrees, yaw * degrees];
}
export function rotationQuaternion(rotation: Vec3): Quaternion {
  // Three's ZYX Euler is the desktop Rz(yaw) @ Ry(pitch) @ Rx(roll).
  return new Quaternion().setFromEuler(new Euler(...rotation.map(v => v / degrees) as Vec3, 'ZYX'));
}
export function parseCamera(text: string): CameraView {
  const view = JSON.parse(text) as CameraView;
  if (!view || typeof view !== 'object') throw new Error('The view must be a JSON object.');
  const allowed = ['position', 'rotation', 'pivot', 'fov', 'near', 'far'];
  if (Object.keys(view).some(key => !allowed.includes(key))) throw new Error('Unknown camera field. Use position, rotation, pivot, fov, near, and far.');
  for (const key of ['position', 'rotation', 'pivot'] as const) {
    if (!Array.isArray(view[key]) || view[key].length !== 3 || !view[key].every(v => typeof v === 'number' && Number.isFinite(v))) throw new Error(`${key} requires three finite numbers.`);
  }
  for (const key of ['fov', 'near', 'far'] as const) if (typeof view[key] !== 'number' || !Number.isFinite(view[key])) throw new Error(`${key} must be finite.`);
  if (!(view.fov > 0 && view.fov < 180)) throw new Error('Field of view must be between 0 and 180 degrees.');
  if (!(view.near > 0 && view.far > view.near)) throw new Error('Clipping distances must satisfy 0 < near < far.');
  return view;
}
export function absolute(local: Vector3, origin: Vector3): Vec3 {return local.clone().add(origin).toArray() as Vec3;}
