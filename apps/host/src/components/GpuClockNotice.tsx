import { tr } from '../lib/i18n';
import { Icon } from './ui';
import { CopyBtn } from './CopyBtn';
import type { GpuInfo } from '../lib/types';

/** amdgpu power levels that hold the card at its lowest clocks (server/device-probes.mjs LOW_CLOCK_LEVELS). */
const LOW = ['low', 'profile_min_sclk', 'profile_min_mclk'];

export function lowClockGpus(gpus: GpuInfo[] | undefined): GpuInfo[] {
    return (gpus ?? []).filter(g => g.vendor === 'amd' && LOW.includes(String(g.clockLevel ?? '').trim()));
}

const NOW = 'echo auto | sudo tee /sys/class/drm/card*/device/power_dpm_force_performance_level';
const FIND = 'grep -rl power_dpm_force_performance_level /etc/udev/rules.d /lib/udev/rules.d';

/** Linux AMD cards pinned to their lowest clock (seen on Mac Pro 2019 setups): decode runs ~30 % slower. */
export function GpuClockNotice({ gpus }: { gpus: GpuInfo[] | undefined }) {
    const low = lowClockGpus(gpus);
    if (!low.length)
        return null;
    const names = low.map(g => `${g.name} (${g.clockLevel})`).join(', ');
    return <div className="models-notice models-notice--warn gpu-clock-notice" role="status"><Icon name="warning" size={18}/><div>
      <strong>{tr('Your AMD GPU clocks are held at their lowest setting.')}</strong>
      <p className="small">{tr('{0}. Models decode about 30 % slower like this (on a Radeon Pro Vega II Duo: 999 MHz instead of up to 1,730 MHz). Set the power level to auto, then restart Strata.', [names])}</p>
      <details><summary className="small">{tr('How to fix it')}</summary>
        <p className="small">{tr('Until the next reboot:')}</p>
        <div className="row"><code className="xs" style={{ overflowWrap: 'anywhere' }}>{NOW}</code><CopyBtn text={NOW} label="Command" title="Copy command"/></div>
        <p className="small">{tr('To keep it: find the udev rule that sets low, and change "low" to "auto" in it (or remove the rule).')}</p>
        <div className="row"><code className="xs" style={{ overflowWrap: 'anywhere' }}>{FIND}</code><CopyBtn text={FIND} label="Command" title="Copy command"/></div>
        <p className="xs muted">{tr('Details in docs/MAC_PRO_2019_LINUX.md.')}</p>
      </details>
    </div></div>;
}
