"""CDI of a captured photograph: synthetic intensity measurement, ER/HIO recovery.

The inverse routine receives only measured Fourier intensity and a rectangular
support prior. The original image is used for forward modelling and evaluation.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import json
import shutil
import time
import numpy as np
from PIL import Image
from .core import Grid, fft2c
from .objects import image_object
from .diffraction import recover_cdi
from .metrics import align_cdi, relative_error, similarity


@dataclass
class CDIImageConfig:
    n: int = 256
    image_samples: int = 80
    dx: float = 5e-6
    wavelength: float = 632.8e-9
    focal_length: float = 1.
    support_margin_pixels: int = 3
    iterations: int = 2000
    beta: float = .9
    seeds: tuple = (7, 17, 27)


def run(image_path, output, config=None):
    """Run a complete image CDI example and save measured data and raw iterates."""
    c=config or CDIImageConfig()
    g=Grid(c.n,c.dx,c.wavelength)
    if c.iterations<1 or not c.seeds or not 0<c.beta<=1:
        raise ValueError('CDI requires iterations > 0, at least one seed and 0 < beta <= 1.')
    if c.support_margin_pixels<0 or c.focal_length<=0:
        raise ValueError('Support margin must be nonnegative and focal length positive.')
    source=Path(image_path).resolve()
    output=Path(output).resolve()
    (output/'01_输入图').mkdir(parents=True,exist_ok=True)
    (output/'06_数据').mkdir(parents=True,exist_ok=True)
    captured=output/'01_输入图'/('原始输入图片'+source.suffix.lower())
    if source!=captured: shutil.copy2(source,captured)
    with Image.open(captured) as pic:
        pic=pic.convert('L')
        pic.thumbnail((c.image_samples,c.image_samples),Image.Resampling.LANCZOS)
        w,h=pic.size
    if max(w,h)+2*c.support_margin_pixels>=c.n:
        raise ValueError('Image plus support margin must fit within the zero-padded grid.')
    truth=image_object(g,captured,c.image_samples,'amplitude')
    intensity=np.abs(fft2c(truth))**2
    # Support uses only specified rectangular extent, never pixel values of truth.
    y0,x0=c.n//2-h//2,c.n//2-w//2
    m=c.support_margin_pixels
    support=np.zeros((c.n,c.n),dtype=bool)
    support[y0-m:y0+h+m,x0-m:x0+w+m]=True
    inverse_start=time.perf_counter()
    trials=[]
    arrays={'truth_field':truth,'measurement_intensity':intensity,'support':support,
            'object_x_m':g.x,'frequency_per_m':g.frequency,
            'detector_x_m':g.wavelength*c.focal_length*g.frequency}
    for seed in c.seeds:
        candidate,history=recover_cdi(intensity,support,iterations=c.iterations,
            algorithm='hio',beta=c.beta,seed=int(seed),positive_real=True)
        residual=relative_error(np.abs(fft2c(candidate)),np.sqrt(intensity))
        trials.append((residual,int(seed),candidate,history))
        arrays[f'hio_seed_{seed}_field']=candidate
        arrays[f'hio_seed_{seed}_residual']=history
        print(f'HIO seed={seed}: Fourier modulus residual={residual:.6g}',flush=True)
    # Select by measured-data consistency; no truth-dependent selection.
    residual,seed,best,history=min(trials,key=lambda entry:entry[0])
    er,er_history=recover_cdi(intensity,support,iterations=c.iterations,
        algorithm='er',seed=int(c.seeds[0]),positive_real=True)
    elapsed=time.perf_counter()-inverse_start
    aligned,error,ambiguity=align_cdi(best,truth)
    er_aligned,er_error,er_ambiguity=align_cdi(er,truth)
    arrays.update(recovered_raw_field=best,recovered_aligned_field=aligned,
        er_raw_field=er,er_aligned_field=er_aligned,
        hio_residual=history,er_residual=er_history)
    metrics={
        'image_samples_yx':[h,w], 'support_samples_yx':[h+2*m,w+2*m],
        'object_size_mm_yx':[h*g.dx*1e3,w*g.dx*1e3],
        'support_area_oversampling_ratio':float(c.n**2/support.sum()),
        'selected_seed':seed,
        'selection_rule':'minimum Fourier-modulus residual across seeds, no truth input',
        'hio_fourier_modulus_relative_error':residual,
        'er_fourier_modulus_relative_error':relative_error(np.abs(fft2c(er)),np.sqrt(intensity)),
        'hio_aligned_complex_relative_error':error,
        'er_aligned_complex_relative_error':er_error,
        'hio_aligned_intensity_relative_error':relative_error(np.abs(aligned)**2,np.abs(truth)**2),
        'hio_aligned_intensity_cosine_similarity':similarity(np.abs(aligned)**2,np.abs(truth)**2),
        'hio_evaluation_alignment':ambiguity,'er_evaluation_alignment':er_ambiguity,
        'hio_trials':[{'seed':t[1],'fourier_modulus_residual':t[0]} for t in trials],
        'inverse_elapsed_seconds':elapsed,
        'measurement':'noise-free synthetic Fourier intensity from a real photograph',
        'prior':f'rectangular extent with {m}-pixel margin, real nonnegative object',
        'source':str(captured),
        'display':'linear grayscale intensity; no gain or gamma enhancement',
        'phase_scope':'photo gray is amplitude; phase zero is a model assumption',
        'ambiguity':'translation, conjugate inversion and global phase aligned only for evaluation',
        'focal_plane_scale':'orthonormal DFT intensity in arbitrary units, X=lambda*f*frequency',
    }
    np.savez_compressed(output/'06_数据/data.npz',**arrays)
    (output/'06_数据/metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    config_dict={'image_path':str(captured),**asdict(c)}
    (output/'06_数据/配置.json').write_text(json.dumps(config_dict,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'README.md').write_text(
        '# 真实照片的CDI反演\n\n'
        '照片灰度作为实非负振幅，初相位设为0。正向模型只生成傅里叶光强。'
        '反演只接收该光强、外扩矩形支集及实非负先验，不接收原图像素或原始相位。\n\n'
        'HIO按各随机初值的测量模残差选择结果，最后50次迭代为ER精修。'
        '真值只在选择结束后用于平移、共轭反演和全局相位对齐及评价。'
        '结果属于理想无噪声数值例，不能据此声称任意复物体都能唯一恢复。\n\n'
        'data.npz同时保存测量光强、支集、全部随机初值结果、未对齐恢复、'
        '评价对齐恢复及迭代曲线。图像采用线性灰度光强，不增强弱条纹。\n',encoding='utf-8')
    from .cdi_image_report import export_report
    export_report(output)
    print(json.dumps({k:metrics[k] for k in ['selected_seed','hio_fourier_modulus_relative_error',
        'hio_aligned_complex_relative_error','er_aligned_complex_relative_error','inverse_elapsed_seconds']},ensure_ascii=False),flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser(description='Real-photo coherent diffraction imaging')
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/cdi_image/photo'))
    p.add_argument('--samples',type=int,default=80)
    p.add_argument('--n',type=int,default=256)
    p.add_argument('--iterations',type=int,default=2000)
    args=p.parse_args()
    run(args.image,args.output,CDIImageConfig(n=args.n,image_samples=args.samples,iterations=args.iterations))


if __name__=='__main__':main()
