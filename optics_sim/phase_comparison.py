"""Reproducible zero/2-D-sinusoidal input-phase comparison on one real photo.

All reconstructions receive only their permitted data. CDI uses the same
complex support prior, seeds and iteration budget for both phase conditions.
Truth is used only after seed selection for ambiguity alignment/evaluation.
"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from .core import Grid, fft2c, ifft2c, propagate
from .holography import bandlimit, record, reconstruct_off_axis
from .abbe import invert
from .metrics import align_cdi, relative_error, phase_rmse, similarity
from scipy.optimize import minimize

ROOT=Path(__file__).resolve().parents[1]

def phase_on_photo(amplitude, peak=1.2, cycles=8):
    """Phase law on the shared photo rectangle."""
    rows,cols=np.nonzero(np.abs(amplitude)>0)
    r0,r1,c0,c1=rows.min(),rows.max()+1,cols.min(),cols.max()+1
    h,w=r1-r0,c1-c0
    xx=(np.arange(w)+.5)/w-.5; yy=(np.arange(h)+.5)/h-.5
    phase=np.zeros(amplitude.shape,float)
    phase[r0:r1,c0:c1]=peak*np.sin(2*np.pi*cycles*yy[:,None])*np.sin(2*np.pi*cycles*xx[None,:])
    return phase, [int(r0),int(r1),int(c0),int(c1)]

def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def align_for_evaluation(candidate,truth):
    """Remove spatial-origin ambiguity, including a subpixel origin offset.

    This truth-dependent operation is strictly after reconstruction/selection.
    A periodic Fourier shift preserves the recovered diffraction modulus.
    """
    aligned,error,ambiguity=align_cdi(candidate,truth)
    f=np.fft.fftshift(np.fft.fftfreq(truth.shape[0]));fx,fy=np.meshgrid(f,f)
    spectrum=fft2c(aligned)
    def shifted(delta):
        u=ifft2c(spectrum*np.exp(-2j*np.pi*(fy*delta[0]+fx*delta[1])))
        return u*np.exp(-1j*np.angle(np.vdot(truth,u)))
    result=minimize(lambda d:relative_error(shifted(d),truth),[0.,0.],method='Powell',
        bounds=[(-.75,.75),(-.75,.75)],options={'xtol':1e-7,'ftol':1e-10,'maxiter':50})
    if result.fun<error:
        aligned=shifted(result.x);error=float(result.fun)
        ambiguity['subpixel_shift_yx']=result.x.tolist()
    else:ambiguity['subpixel_shift_yx']=[0.,0.]
    return aligned,error,ambiguity

def update_saved_evaluation(output):
    out=Path(output)
    with np.load(out/'cdi.npz') as data:arrays={k:data[k] for k in data.files}
    metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'))
    for key in ['zero','sine']:
        if key+'_recovered_raw_field' not in arrays:continue
        truth=arrays[key+'_truth_field'];raw=arrays[key+'_recovered_raw_field']
        aligned,error,ambiguity=align_for_evaluation(raw,truth)
        arrays[key+'_recovered_aligned_field']=aligned
        metrics['cdi'][key].update(aligned_complex_relative_error=error,
          aligned_phase_rmse_rad=phase_rmse(aligned,truth),
          aligned_intensity_relative_error=relative_error(np.abs(aligned)**2,np.abs(truth)**2),
          evaluation_alignment=ambiguity)
    np.savez_compressed(out/'cdi.npz',**arrays);write_json(out/'metrics.json',metrics)
    print(json.dumps(metrics['cdi'],ensure_ascii=False),flush=True)

def run(output,iterations=3000,seeds=(7,17,27),peak=1.2,cycles=8,support_margin=0):
    if support_margin != 0:raise ValueError('Unified CDI uses the exact known photo rectangle; support margin must be zero.')
    out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    rp=ROOT/'outputs/image_simulations/red_panda'
    hdir=rp/'02_照片离轴全息/06_数据'
    adir=rp/'03_照片阿贝成像_圆形低通/06_数据'
    picture=hdir.parent/'01_输入图/原始输入图片.jpg'
    shutil.copy2(picture,out/'原始输入图片.jpg')
    config={'phase_peak_rad':peak,'phase_cycles_xy':[cycles,cycles],
      'phase_law':'phi=peak*sin(2*pi*cycles*x/W)*sin(2*pi*cycles*y/H), coordinates centered on photo pixel footprint',
      'input':'same real-photo grayscale amplitude, changed phase only before optical pupils',
      'phase_scope':'prescribed numerical phase; not the real optical phase of the photographed animal',
      'cdi_iterations_per_seed':iterations,'cdi_seeds':list(seeds),'cdi_positive_real':False,
      'cdi_support_margin_pixels':support_margin,'cdi_support_source':'known rectangular photo dimensions, not image pixel values',
      'source_data':[str(hdir/'data.npz'),str(adir/'data.npz')]}
    write_json(out/'配置.json',config)
    metrics={}
    with np.load(hdir/'data.npz') as d:
        raw=d['raw_object_field'];phi,roi=phase_on_photo(raw,peak,cycles)
        p=json.loads((hdir/'配置.json').read_text(encoding='utf-8'))['config']
        g=Grid(p['n'],p['dx'],p['wavelength']);ref=d['reference_wave']
        arrays={'object_x_m':g.x,'frequency_per_m':g.frequency,'input_amplitude':np.abs(raw),
            'prescribed_phase_rad':phi,'photo_roi_rc':np.array(roi),'reference_wave':ref}
        hm={}
        for key,phase in [('zero',np.zeros_like(phi)),('sine',phi)]:
            source=raw*np.exp(1j*phase)
            target,pupil=bandlimit(source,g,p['bandwidth'])
            wave=propagate(target,g,p['z'],backend=p['backend'])
            intensity=record(wave,ref)
            recovered_wave=reconstruct_off_axis(intensity,ref,g,p['bandwidth'],p['reference_amplitude'])
            recovered=propagate(recovered_wave,g,-p['z'],backend=p['backend'])
            arrays.update({key+'_raw_field':source,key+'_target_field':target,
                key+'_object_wave_field':wave,key+'_hologram_intensity':intensity,
                key+'_hologram_spectrum':np.abs(fft2c(intensity))**2,key+'_recovered_field':recovered})
            hm[key]={'complex_relative_error':relative_error(recovered,target),
                'phase_rmse_rad':phase_rmse(recovered,target),
                'bandlimit_retained_power':float(np.sum(np.abs(target)**2)/np.sum(np.abs(source)**2))}
        arrays['pupil']=pupil
        np.savez_compressed(out/'holography.npz',**arrays)
        metrics['holography']=hm
    print('holography complete',flush=True)
    with np.load(adir/'data.npz') as d:
        raw=d['object_field'];phi,roi=phase_on_photo(raw,peak,cycles)
        p=json.loads((adir/'配置.json').read_text(encoding='utf-8'))['config']
        pupil=d['pupil'];lowpass=d['filter_mask']
        highfile=adir.parent/'滤波对比/06_数据/highpass/data.npz'
        with np.load(highfile) as v:highpass=v['filter_mask']
        mag=p['f2']/p['f1']
        arrays={'object_x_m':d['object_x_m'],'prescribed_phase_rad':phi,'input_amplitude':np.abs(raw),
          'photo_roi_rc':np.array(roi),'pupil':pupil,'lowpass_mask':lowpass,'highpass_mask':highpass,
          'frequency_per_m':d['frequency_per_m']}
        am={}
        for key,phase in [('zero',np.zeros_like(phi)),('sine',phi)]:
            source=raw*np.exp(1j*phase);arrays[key+'_input_field']=source
            for label,mask in [('none',np.ones_like(pupil)),('lowpass',lowpass),('highpass',highpass)]:
                restored=ifft2c(fft2c(source)*pupil*mask)/mag
                # Equivalent to the original two-lens double forward transform.
                physical=-invert(restored)
                arrays[key+'_'+label+'_field']=physical
                arrays[key+'_'+label+'_oriented_field']=restored
                am[key+'_'+label]={'transmitted_power_fraction':float(np.sum(np.abs(physical)**2)/np.sum(np.abs(source)**2))}
        for label in ['none','lowpass','highpass']:
            a=np.abs(arrays['zero_'+label+'_oriented_field'])**2
            b=np.abs(arrays['sine_'+label+'_oriented_field'])**2
            am[label+'_phase_effect']={'intensity_relative_difference':relative_error(b,a),'intensity_similarity':similarity(b,a)}
        np.savez_compressed(out/'abbe.npz',**arrays);metrics['abbe']=am
    print('abbe complete',flush=True)
    write_json(out/'metrics.json',metrics)
    from .cdi_shared_input import run as run_shared_cdi
    metrics=run_shared_cdi(out,iterations,tuple(seeds),backend='cpu')
    (out/'README.md').write_text('# 初始相位对比\n\n相同照片灰度振幅，对比零相位和1.2 rad二维正弦相位，照片宽、高方向各8周期。相位为数值模型，不是照片物体真实相位。\n\n全息保留原有限带宽和参考光设置。阿贝保留原孔径及低通、圆形高通滤波器，保存物理倒像和为比较而转正的图像。CDI两组均用相同复振幅支集约束、迭代预算和随机初值，按衍射光强一致性选取结果，真值仅用于事后评价。\n\n原始复数数组保存在npz中，绘图增强仅用于显示。\n',encoding='utf-8')
    print(json.dumps(metrics,ensure_ascii=False),flush=True)
    return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'outputs/phase_comparison/red_panda')
    p.add_argument('--iterations',type=int,default=3000);p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--support-margin',type=int,default=0,help='Extra pixels outside the known photo rectangle')
    p.add_argument('--align-only',action='store_true',help='Update evaluation alignment of saved reconstructions only')
    args=p.parse_args()
    if args.align_only:update_saved_evaluation(args.output)
    else:run(args.output,args.iterations,tuple(args.seeds),support_margin=args.support_margin)

if __name__=='__main__':main()
