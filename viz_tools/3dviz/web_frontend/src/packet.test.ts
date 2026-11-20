import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeFrame } from './packet';
import type { FrameHeader } from './types';
function makePacket(change?: (header: FrameHeader)=>void) {
  const header: FrameHeader={protocol:1,revision:7,count:2,origin:[640000,4900000,0],bounds:{min:[640000,4900000,0],max:[640001,4900001,1]},frame:{id:'frame',scene_id:'scene',index:0,timestamp_us:'1470000000000000'},source_id:'source',reference_frame_id:'reference',annotation_status:'available',boxes:[],rgb_count:1,display:{rgb:true,raw:false},buffers:[{name:'positions',dtype:'<f4',shape:[2,3],offset:0,length:24},{name:'distance',dtype:'<f4',shape:[2],offset:24,length:8},{name:'rgb',dtype:'u1',shape:[2,3],offset:32,length:6},{name:'rgb_mask',dtype:'u1',shape:[2],offset:38,length:2}]};
  change?.(header);
  const json=new TextEncoder().encode(JSON.stringify(header)); const length=Math.ceil(json.length/4)*4;
  const packet=new ArrayBuffer(4+length+40);new DataView(packet).setUint32(0,length,true);
  new Uint8Array(packet,4,length).fill(32);new Uint8Array(packet,4,json.length).set(json);
  new Float32Array(packet,4+length,6).set([0,0,0,1,1,1]);new Float32Array(packet,4+length+24,2).set([5,80]);new Uint8Array(packet,4+length+32,8).set([255,0,127,0,0,0,1,0]);return packet;
}
test('wire packet keeps rebased arrays zero-copy and global origin intact',()=>{
  const raw=makePacket(), packet=decodeFrame(raw);
  assert.equal(packet.positions.buffer,raw);assert.deepEqual([...packet.positions],[0,0,0,1,1,1]);
  assert.deepEqual(packet.header.origin,[640000,4900000,0]);assert.deepEqual([...packet.distance],[5,80]);assert.deepEqual([...packet.rgbMask!],[1,0]);
});
test('wire parser rejects unsupported protocol and mismatched array count',()=>{
  assert.throws(()=>decodeFrame(makePacket(h=>{h.protocol=2;})),/Unsupported/);
  assert.throws(()=>decodeFrame(makePacket(h=>{h.count=3;})),/positions/);
});
test('wire parser rejects truncated, overlapping, misaligned, unpaired buffers',()=>{
  assert.throws(()=>decodeFrame(makePacket().slice(0,-1)),/buffer/);
  assert.throws(()=>decodeFrame(makePacket(h=>{h.buffers[1].offset=20;})),/Overlapping/);
  assert.throws(()=>decodeFrame(makePacket(h=>{h.buffers[1].offset=25;})),/type/);
  assert.throws(()=>decodeFrame(makePacket(h=>{h.buffers.pop();})),/paired/);
  assert.throws(()=>decodeFrame(new ArrayBuffer(2)),/incomplete/);
});
test('wire parser rejects malformed corners before renderer changes',()=>{
  assert.throws(()=>decodeFrame(makePacket(h=>{h.boxes=[{id:'bad',class_id:'car',group:'car',corners:[[0,0,0]]}];})),/bounding-box/);
});
