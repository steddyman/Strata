import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { clockPinnedLow, LOW_CLOCK_LEVELS } from '../server/device-probes.mjs';

test('amdgpu levels that hold the lowest clocks are flagged, the others are not', () => {
  for (const level of ['low', 'profile_min_sclk', 'profile_min_mclk', 'low\n']) assert.equal(clockPinnedLow(level), true, level);
  for (const level of ['auto', 'high', 'manual', 'profile_peak', 'profile_standard', '', null, undefined]) assert.equal(clockPinnedLow(level), false, String(level));
});

test('the Model page notice uses the same list as the probe', () => {
  const ui = readFileSync(new URL('../src/components/GpuClockNotice.tsx', import.meta.url), 'utf8');
  assert.match(ui, new RegExp(`const LOW = ${JSON.stringify(LOW_CLOCK_LEVELS).replaceAll('"', "'").replace(/[[\]]/g, '\\$&').replaceAll(',', ', ')};`));
  assert.match(readFileSync(new URL('../src/screens/ModelRun.tsx', import.meta.url), 'utf8'), /<GpuClockNotice gpus=\{hardware\?\.gpus\}\/>/);
});
