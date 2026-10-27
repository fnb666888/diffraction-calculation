"""One off-axis exposure of the existing wolf point field, then independent replay."""
from pathlib import Path
import json
import numpy as np
from PIL import Image
from .core import Grid, fft2c, ifft2c, propagate

ROOT = Path(__file__).resolve().parents[1]


def recover_single(counts, calibration):
    """Recover U_H from one saved intensity frame and known reference only."""
    c = calibration
    g = Grid(c['n'], c['dx_m'], c['wavelength_m'])
    intensity = counts.astype(float) * (c['camera_intensity_peak'] / 65535)
    fx, fy = np.meshgrid(g.frequency, g.frequency)
    mask = (fx + c['carrier_per_m'])**2 + fy**2 <= c['bandwidth_per_m']**2
    selected = ifft2c(fft2c(intensity) * mask)
    x, _ = g.xy
    reference = c['reference_amplitude'] * np.exp(2j*np.pi*c['carrier_per_m']*x)
    return selected * reference / c['reference_amplitude']**2


def run():
    parent = ROOT/'outputs/pointcloud_direct/wolf'
    out = parent/'single-offaxis'
    out.mkdir(exist_ok=True)
    c = json.loads((parent/'calibration.json').read_text(encoding='utf-8'))
    # The reference field enters the forward camera simulation only.
    with np.load(parent/'input-reference.npz') as packet:
        forward = packet['reference_sensor_field'].astype(np.complex128)
    g = Grid(c['n'], c['dx_m'], c['wavelength_m'])
    x, _ = g.xy
    reference = c['reference_amplitude'] * np.exp(2j*np.pi*c['carrier_per_m']*x)
    intensity = np.abs(forward + reference)**2
    peak = float(intensity.max())
    counts = np.rint(intensity/peak*65535).astype(np.uint16)
    c.update(label='单幅离轴数字全息', digital_recording='single_offaxis',
             camera_intensity_peak=peak, camera_frame_count=1,
             replay='FFT of one 16-bit off-axis frame; circular -fc sideband selection; IFFT and reference demodulation; negative-distance Fresnel propagation')
    np.savez_compressed(out/'recordings.npz', camera_counts=counts)
    (out/'calibration.json').write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding='utf-8')
    Image.fromarray(np.flipud(counts)).save(out/'camera-single-16bit.png')
    # Independent inverse: reopen the saved exposure and calibration, no truth input.
    c_replay = json.loads((out/'calibration.json').read_text(encoding='utf-8'))
    with np.load(out/'recordings.npz') as packet:
        recovered = recover_single(packet['camera_counts'], c_replay)
    fields = np.array([propagate(recovered, g, -z, method='fresnel', backend='scipy')
                       for z in c_replay['focus_distances_m']])
    np.savez_compressed(out/'reconstructed.npz', recovered_hologram_field=recovered.astype(np.complex64),
                        focus_fields=fields.astype(np.complex64),
                        focus_distances_m=np.array(c_replay['focus_distances_m']))
    # Forward truth is used only after replay for a numerical accuracy assessment.
    error = float(np.linalg.norm(recovered-forward)/np.linalg.norm(forward))
    metrics = {'saved_record_replay_complex_relative_error': error, 'camera_frame_count': 1,
               'sideband_center_per_m': [-c['carrier_per_m'], 0],
               'scope': 'complex-field relative L2 error at hologram plane; not geometry recovery'}
    (out/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    run()
