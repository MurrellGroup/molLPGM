import test from 'node:test';
import assert from 'node:assert/strict';
import {selectFigureFrames,filterTrajectories} from './figure-utils.mjs';
test('frames come directly from recorded physical times', () => {
  assert.deepEqual(selectFigureFrames('0,.3,1',[0,.25,.7,1]).map(x=>x.frame),[0,1,3]);
  assert.throws(()=>selectFigureFrames('.2,.3',[0,.25,.7,1]));
});
test('joint filter requires both paired samples to pass', () => {
  const trajectories=[{id:'a',paired_id:'b'},{id:'b',paired_id:'a'}];
  const quality={samples:{a:{valid:true,connected:true},b:{valid:true,connected:false}}};
  assert.equal(filterTrajectories(trajectories,quality,true,'valid_connected',true).length,0);
  assert.equal(filterTrajectories(trajectories,quality,true,'valid',true).length,2);
  quality.samples.b.connected=true;
  assert.equal(filterTrajectories(trajectories,quality,true,'valid_connected',true).length,2);
  assert.equal(filterTrajectories(trajectories,null,true,'valid_connected',true).length,0);
});
