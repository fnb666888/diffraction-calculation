"""Planar WRP experiment on the unchanged real-photo holography inputs.

The near-plane local accumulation uses a truncated *discrete* Fresnel impulse
response. For a single planar dense object its square window is separable, so
two local 1-D convolutions implement exactly the same 2-D point accumulation.
The second stage is an ordinary full-plane Fresnel FFT propagation. This is a
planar specialization, not a claim of a general RGB-D/3-D WRP implementation.
"""
from pathlib import Path
import argparse
import json
import platform
import time
import numpy as np
from scipy import fft, ndimage
from .core import Grid,transfer_function,propagate
from .holography import record,reconstruct_off_axis
from .metrics import relative_error

ROOT=Path(__file__).resolve().parents[1]


def prepare_local_kernel(grid,distance,radius):
    """Discrete kernel matches the same grid, normalization and periodic BC."""
    f=fft.fftfreq(grid.n,grid.dx)
    h=fft.fftshift(fft.ifft(np.exp(-1j*np.pi*grid.wavelength*distance*f*f)))
    c=grid.n//2
    return h[c-radius:c+radius+1].copy()


def local_accumulate(field,kernel):
    """Separable local point contributions, all additions in complex amplitude."""
    return ndimage.convolve1d(ndimage.convolve1d(field,kernel,axis=0,mode='wrap'),
                              kernel,axis=1,mode='wrap')


def cached_propagate(field,transfer):
    return fft.ifft2(fft.fft2(field,workers=1)*transfer,workers=1)


def run(directory=None,repeats=9):
    source=ROOT/'outputs/phase_comparison/red_panda'
    out=Path(directory or ROOT/'outputs/wrp_experiment/red_panda').resolve()
    out.mkdir(parents=True,exist_ok=True)
    with np.load(source/'holography.npz') as d:
        fields={k:d[k+'_target_field'] for k in ['zero','sine']}
        reference=d['reference_wave'];roi=d['photo_roi_rc'];x=d['object_x_m']
        original={k:d[k+'_object_wave_field'] for k in fields}
    g=Grid(len(x),2.5e-6,632.8e-9);z=.02;near=.0001
    h=transfer_function(g,z);h1=transfer_function(g,near);h2=transfer_function(g,z-near)
    kernels={r:prepare_local_kernel(g,near,r) for r in [6,12,24]}
    methods={'direct_fft':lambda u:cached_propagate(u,h),
             'full_wrp_split':lambda u:cached_propagate(cached_propagate(u,h1),h2)}
    for r,kernel in kernels.items():
        methods[f'local_wrp_r{r}']=lambda u,k=kernel:cached_propagate(local_accumulate(u,k),h2)
    setup=[]
    for _ in range(repeats):
        t=time.perf_counter();transfer_function(g,z);setup.append(time.perf_counter()-t)
    result={'configuration':{'n':g.n,'dx_m':g.dx,'wavelength_m':g.wavelength,
          'total_distance_m':z,'object_to_wrp_m':near,'wrp_to_hologram_m':z-near,
          'local_halfwidth_pixels':list(kernels),'repeats':repeats,
          'precision':'complex128','fft_workers':1,'cpu':platform.processor(),
          'timing_scope':'forward object-wave generation only, common cached transfer functions/kernels; file I/O, plotting, optical pupil and reference recording excluded',
          'kernel_setup_median_ms':float(np.median(setup)*1000),
          'input':'unchanged bandlimited target fields from phase_comparison/red_panda/holography.npz',
          'scope':'single-depth dense real photo, planar specialization of local WRP; no inferred real depth'},'cases':{}}
    arrays={'object_x_m':x,'photo_roi_rc':roi,'input_phase_zero':np.angle(fields['zero']),
            'input_phase_sine':np.angle(fields['sine'])}
    for key,u in fields.items():
        outputs={name:fn(u) for name,fn in methods.items()}
        timings={name:[] for name in methods}
        # Rotate order to reduce systematic cache/thermal ordering effects.
        names=list(methods)
        for repeat in range(repeats):
            for name in names[repeat%len(names):]+names[:repeat%len(names)]:
                t=time.perf_counter();methods[name](u);timings[name].append(time.perf_counter()-t)
        baseline=outputs['direct_fft'];baseline_hologram=record(baseline,reference)
        case={}
        for name,wave in outputs.items():
            holo=record(wave,reference)
            recovered_wave=reconstruct_off_axis(holo,reference,g,44000.,1.5)
            recovered=cached_propagate(recovered_wave,transfer_function(g,-z))
            t=np.asarray(timings[name])*1000
            case[name]={'median_ms':float(np.median(t)),'min_ms':float(t.min()),'max_ms':float(t.max()),
                        'samples_ms':t.tolist(),
                        'speed_ratio_vs_direct':float(np.median(timings['direct_fft'])/np.median(timings[name])),
                        'object_wave_relative_error':relative_error(wave,baseline),
                        'hologram_intensity_relative_error':relative_error(holo,baseline_hologram),
                        'reconstructed_complex_relative_error':relative_error(recovered,u),
                        'reconstructed_intensity_relative_error':relative_error(np.abs(recovered)**2,np.abs(u)**2)}
            arrays[f'{key}_{name}_wave']=wave
            arrays[f'{key}_{name}_recovered']=recovered
        case['direct_fft']['relative_difference_from_saved_lightpipes_wave']=relative_error(baseline,original[key])
        result['cases'][key]=case
        print(json.dumps({key:case},ensure_ascii=True),flush=True)
    result['summary']={}
    for name in methods:
        rows=[result['cases'][key][name] for key in fields]
        result['summary'][name]={
            'pooled_median_ms':float(np.median(sum([row['samples_ms'] for row in rows],[]))),
            'max_complex_relative_error':max(row['object_wave_relative_error'] for row in rows),
            'max_intensity_relative_error':max(row['reconstructed_intensity_relative_error'] for row in rows)}
    np.savez_compressed(out/'data.npz',**arrays)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--repeats',type=int,default=9)
    args=p.parse_args();run(args.output,args.repeats)


if __name__=='__main__':main()
