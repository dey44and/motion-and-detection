import type { FrameHeader, FramePacket } from './types';

/** Decode zero-copy little-endian arrays while checking the complete wire contract. */
export function decodeFrame(packet: ArrayBuffer): FramePacket {
  if (packet.byteLength < 4) throw new Error('The point-cloud response is incomplete.');
  const headerLength = new DataView(packet).getUint32(0, true);
  const payloadStart = 4 + Math.ceil(headerLength / 4) * 4;
  if (headerLength < 2 || payloadStart > packet.byteLength) throw new Error('Invalid point-cloud header length.');
  const header = JSON.parse(new TextDecoder('utf-8', {fatal: true}).decode(new Uint8Array(packet, 4, headerLength))) as FrameHeader;
  const vec3 = (value: unknown): boolean => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
  if (header.protocol !== 1 || !Number.isSafeInteger(header.count) || header.count < 0 || !Number.isSafeInteger(header.revision) || !vec3(header.origin) || !vec3(header.bounds?.min) || !vec3(header.bounds?.max) || !Array.isArray(header.buffers) || !Array.isArray(header.boxes)) {
    throw new Error('Unsupported or invalid point-cloud packet.');
  }
  const seen = new Set<string>();
  const spans: [number, number][] = [];
  for (const b of header.buffers) {
    if (seen.has(b.name) || !Array.isArray(b.shape) || !b.shape.every(n => Number.isSafeInteger(n) && n >= 0) || !Number.isSafeInteger(b.offset) || b.offset < 0 || !Number.isSafeInteger(b.length) || b.length < 0 || b.offset + b.length > packet.byteLength - payloadStart) throw new Error('Invalid point-cloud buffer.');
    if (!['<f4', 'u1'].includes(b.dtype) || (b.dtype === '<f4' && b.offset % 4 !== 0)) throw new Error('Unsupported point-cloud buffer type.');
    const expected = b.shape.reduce((a, n) => a * n, 1) * (b.dtype === '<f4' ? 4 : 1);
    if (!Number.isSafeInteger(expected) || b.length !== expected) throw new Error('Point-cloud buffer size does not match its shape.');
    if (b.length && spans.some(([start, end]) => b.offset < end && start < b.offset + b.length)) throw new Error('Overlapping point-cloud buffers.');
    spans.push([b.offset, b.offset + b.length]); seen.add(b.name);
  }
  function buffer(name: string, dtype: string, shape: number[], optional = false): Float32Array | Uint8Array | null {
    const b = header.buffers.find(item => item.name === name);
    if (!b && optional) return null;
    if (!b || b.dtype !== dtype || b.shape.length !== shape.length || b.shape.some((v, i) => v !== shape[i])) throw new Error(`Invalid ${name} point-cloud buffer.`);
    return dtype === '<f4' ? new Float32Array(packet, payloadStart + b.offset, b.length / 4) : new Uint8Array(packet, payloadStart + b.offset, b.length);
  }
  const positions = buffer('positions', '<f4', [header.count, 3]) as Float32Array;
  const distance = buffer('distance', '<f4', [header.count]) as Float32Array;
  const rgb = buffer('rgb', 'u1', [header.count, 3], true) as Uint8Array | null;
  const rgbMask = buffer('rgb_mask', 'u1', [header.count], true) as Uint8Array | null;
  if (Boolean(rgb) !== Boolean(rgbMask)) throw new Error('RGB colors and their validity mask must be paired.');
  for (const box of header.boxes) if (!Array.isArray(box.corners) || box.corners.length !== 8 || !box.corners.every(vec3)) throw new Error('Invalid bounding-box geometry.');
  return {header, positions, distance, rgb, rgbMask};
}
