import test from 'node:test';
import assert from 'node:assert/strict';
import { Quaternion, Vector3 } from 'three';
import { absolute, parseCamera, rotationAngles, rotationQuaternion } from './camera';

test('desktop RzRyRx angles roundtrip arbitrary roll, pitch, yaw', () => {
  for (const angles of [[23,-37,132],[-120,65,-75],[90,0,0],[12,90,0],[0,-90,0]] as [number,number,number][]) {
    const q = rotationQuaternion(angles);
    const rebuilt = rotationQuaternion(rotationAngles(q));
    assert.ok(q.angleTo(rebuilt) < 1e-7, `${angles}: quaternion changed`);
  }
});
test('desktop convention uses +Z backward and Rz yaw', () => {
  const q = rotationQuaternion([90,0,90]);
  const forward = new Vector3(0,0,-1).applyQuaternion(q);
  assert.ok(forward.distanceTo(new Vector3(-1,0,0)) < 1e-10);
  assert.ok(rotationQuaternion([0,0,0]).angleTo(new Quaternion()) < 1e-10);
});
test('full camera JSON preserves off-axis pivot, projection and origin precision', () => {
  const view = {position:[640000.125,4900000.25,28],rotation:[23,-37,132],pivot:[640010,4900010,2],fov:48,near:0.03,far:8000};
  assert.deepEqual(parseCamera(JSON.stringify(view)),view);
  assert.deepEqual(absolute(new Vector3(.125,.25,28),new Vector3(640000,4900000,0)),view.position);
});
test('camera parser rejects incomplete, nonfinite, unknown and impossible projection', () => {
  const good={position:[0,0,10],rotation:[0,0,0],pivot:[0,0,0],fov:60,near:.1,far:500};
  for (const bad of [null,{}, {...good,fov:180},{...good,near:0},{...good,far:.05},{...good,rotation:[0,0]},{...good,position:['1',0,0]},{...good,extra:1}]) assert.throws(()=>parseCamera(JSON.stringify(bad)));
});
