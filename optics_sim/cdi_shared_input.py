"""CDI on the exact 1024-grid input fields used by holography and Abbe.

No resizing, known-object initialization or input-phase prior is used. The
inverse receives Fourier intensity plus the known photo rectangle only.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import hashlib
import json
import threading
import time
import numpy as np
from scipy import fft
from .core import Grid
from .metrics import relative_error,phase_rmse

ROOT=Path(__file__).resolve().parents[1]

def _forward(u,workers):
    return fft.fftshift(fft.fft2(fft.ifftshift(u),norm='ortho',workers=workers))

def _inverse(u,workers):
    return fft.fftshift(fft.ifft2(fft.ifftshift(u),norm='ortho',workers=workers))

def recover_trial(intensity,roi,iterations,seed,checkpoint,workers=2,callback=None,tolerance=1e-8):
    amplitude=np.sqrt(intensity);denom=float(np.linalg.norm(amplitude))
    signature=hashlib.sha256(intensity.tobytes()+np.asarray(roi).tobytes()).hexdigest()
    r0,r1,c0,c1=map(int,roi);box=np.s_[r0:r1,c0:c1]
    history=[];best_residual=np.inf;start=0;polish_remaining=-1
    if checkpoint.exists():
        with np.load(checkpoint) as data:
            if str(data['signature'])==signature and int(data['budget'])==iterations:
                if bool(data['complete']):
                    return data['best'],data['history'],float(data['best_residual']),float(data['elapsed'])
                u=data['current'];best=data['best'];history=data['history'].tolist()
                start=int(data['step']);best_residual=float(data['best_residual'])
                polish_remaining=int(data['polish_remaining']);previous_elapsed=float(data['elapsed'])
            else:previous_elapsed=0.
    else:previous_elapsed=0.
    if start==0:
        rng=np.random.default_rng(seed)
        u=_inverse(amplitude*np.exp(1j*rng.uniform(-np.pi,np.pi,amplitude.shape)),workers)
        best=np.zeros_like(u)
    candidate=np.zeros_like(u);began=time.perf_counter()
    def save(step,complete):
        temp=checkpoint.with_suffix('.tmp.npz')
        np.savez(temp,signature=signature,budget=iterations,current=u,best=best,history=np.asarray(history),
          best_residual=best_residual,step=step,polish_remaining=polish_remaining,complete=complete,
          elapsed=previous_elapsed+time.perf_counter()-began)
        temp.replace(checkpoint)
    for step in range(start,iterations):
        spectrum=_forward(u,workers)
        magnitude=np.abs(spectrum)
        # F/|F| equals exp(i arg F); at F=0 use the same arg(0)=0 convention.
        unit=np.ones_like(spectrum);np.divide(spectrum,magnitude,out=unit,where=magnitude>0)
        projected=_inverse(amplitude*unit,workers)
        er=step>=iterations-50 or polish_remaining>=0
        if er:u.fill(0)
        else:u-=.9*projected
        u[box]=projected[box]
        candidate.fill(0);candidate[box]=projected[box]
        residual=float(np.linalg.norm(np.abs(_forward(candidate,workers))-amplitude)/denom)
        history.append(residual)
        if residual<best_residual:best_residual=residual;best=candidate.copy()
        if callback is not None and ((step+1)%50==0 or step+1==iterations):
            callback(step+1,best_residual,previous_elapsed+time.perf_counter()-began)
        if polish_remaining>0:polish_remaining-=1
        elif polish_remaining<0 and best_residual<=tolerance:polish_remaining=50
        complete=polish_remaining==0 or step+1==iterations
        if (step+1)%250==0 or complete:save(step+1,complete)
        if complete:break
    return best,np.asarray(history),best_residual,previous_elapsed+time.perf_counter()-began

def run(directory=None,iterations=3000,seeds=(7,17,27),parallel_trials=3,fft_workers=2,backend='cpu'):
    from .phase_comparison import align_for_evaluation
    out=Path(directory or ROOT/'outputs/phase_comparison/red_panda').resolve()
    work=out/'shared_cdi_checkpoints';work.mkdir(parents=True,exist_ok=True)
    with np.load(out/'holography.npz') as data:
        raw={key:data[key+'_raw_field'] for key in ['zero','sine']}
        x=data['object_x_m'];roi=data['photo_roi_rc'];phi=data['prescribed_phase_rad'];amp=data['input_amplitude']
    with np.load(out/'abbe.npz') as data:
        comparisons={key:float(np.max(np.abs(raw[key]-data[key+'_input_field']))) for key in raw}
        coordinate_difference=float(np.max(np.abs(x-data['object_x_m'])))
    if any(v!=0 for v in comparisons.values()) or coordinate_difference!=0:
        raise ValueError('Existing holography and Abbe inputs differ; unify them before CDI.')
    n=len(x);dx=float(x[1]-x[0]);g=Grid(n,dx,632.8e-9)
    r0,r1,c0,c1=map(int,roi);support=np.zeros((n,n),bool);support[r0:r1,c0:c1]=True
    intensities={key:np.abs(_forward(u,fft_workers))**2 for key,u in raw.items()}
    consistency={'photo_samples_yx':[r1-r0,c1-c0],'grid_yx':[n,n],'dx_m':dx,
      'photo_size_m_yx':[(r1-r0)*dx,(c1-c0)*dx],
      'max_complex_input_difference_cdi_holography':{key:0. for key in raw},
      'max_complex_input_difference_cdi_abbe':comparisons,'max_coordinate_difference_m':coordinate_difference,
      'prescribed_phase_max_difference_rad':0.,'source':'holography.npz zero_raw_field/sine_raw_field, before optical pupil'}
    (out/'shared_input.json').write_text(json.dumps(consistency,ensure_ascii=False,indent=2),encoding='utf-8')
    configuration=json.loads((out/'配置.json').read_text(encoding='utf-8'))
    configuration.update(cdi_iterations_per_seed=iterations,cdi_seeds=list(seeds),cdi_positive_real=False,
      cdi_support_margin_pixels=0,cdi_input_source='exact existing holography/Abbe raw complex fields',
      cdi_grid_n=n,cdi_dx_m=dx,cdi_photo_samples_yx=[r1-r0,c1-c0],
      cdi_early_stop_modulus_tolerance=1e-8,cdi_final_er_steps=50)
    (out/'配置.json').write_text(json.dumps(configuration,ensure_ascii=False,indent=2),encoding='utf-8')
    progress={};lock=threading.Lock();last_print=[0.]
    def report(key,seed,step,residual,elapsed):
        with lock:
            progress[f'{key}_{seed}']={'step':step,'best_modulus_residual':residual,'seconds':elapsed}
            (work/'progress.json').write_text(json.dumps(progress,indent=2),encoding='utf-8')
            if time.monotonic()-last_print[0]>=15:
                print(f'{key} seed={seed}: step={step}, residual={residual:.6g}, elapsed={elapsed:.1f}s',flush=True)
                last_print[0]=time.monotonic()
    print(json.dumps(consistency,ensure_ascii=True),flush=True)
    if backend=='auto':
        import importlib.util
        backend='cuda' if importlib.util.find_spec('cupy') else 'cpu'
    results={key:[] for key in raw}
    if backend=='cuda':
        from .cdi_cuda import recover_trials
        results=recover_trials(intensities,roi,iterations,seeds,work,report)
    else:
        with ThreadPoolExecutor(max_workers=parallel_trials) as pool:
            tasks={}
            for key in ['zero','sine']:
                for seed in seeds:
                    callback=lambda step,res,elapsed,k=key,s=seed:report(k,s,step,res,elapsed)
                    job=pool.submit(recover_trial,intensities[key],roi,iterations,seed,work/f'{key}-{seed}.npz',fft_workers,callback)
                    tasks[job]=(key,seed)
            for job in as_completed(tasks):
                key,seed=tasks[job];candidate,history,residual,elapsed=job.result()
                results[key].append((residual,seed,candidate,history,elapsed))
                print(f'completed {key} seed={seed}: iterations={len(history)}, residual={residual:.6g}',flush=True)
    arrays={'object_x_m':x,'frequency_per_m':g.frequency,'detector_x_m':g.wavelength*g.frequency,
      'support':support,'prescribed_phase_rad':phi,'input_amplitude':amp,'photo_roi_rc':roi}
    metrics=json.loads((out/'metrics.json').read_text(encoding='utf-8'));cm={}
    for key in ['zero','sine']:
        trials=results[key];residual,seed,best,history,elapsed=min(trials,key=lambda trial:trial[0])
        aligned,error,ambiguity=align_for_evaluation(best,raw[key])
        arrays.update({key+'_truth_field':raw[key],key+'_diffraction_intensity':intensities[key],
          key+'_recovered_raw_field':best,key+'_recovered_aligned_field':aligned,key+'_selected_history':history})
        for res,s,u,h,e in trials:arrays[f'{key}_seed_{s}_field']=u;arrays[f'{key}_seed_{s}_history']=h
        cm[key]={'selected_seed':seed,'fourier_modulus_relative_error':residual,
          'aligned_complex_relative_error':error,'aligned_phase_rmse_rad':phase_rmse(aligned,raw[key]),
          'aligned_intensity_relative_error':relative_error(np.abs(aligned)**2,np.abs(raw[key])**2),
          'evaluation_alignment':ambiguity,'iterations_used':len(history),
          'trials':[{'seed':s,'fourier_modulus_relative_error':res,'iterations':len(h),'elapsed_seconds':e} for res,s,u,h,e in trials]}
    cm['prior']='known 512x341 photo rectangle; unrestricted complex values, no amplitude truth or input phase in inversion'
    cm['selection']='minimum Fourier modulus residual, truth used only for subsequent evaluation/alignment'
    cm['shared_input']=consistency;cm['backend']=backend;cm['precision']='complex128';metrics['cdi']=cm
    np.savez_compressed(out/'cdi.npz',**arrays)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(cm,ensure_ascii=True),flush=True)
    return metrics

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=ROOT/'outputs/phase_comparison/red_panda')
    p.add_argument('--iterations',type=int,default=3000);p.add_argument('--seeds',type=int,nargs='+',default=[7,17,27])
    p.add_argument('--parallel-trials',type=int,default=3);p.add_argument('--fft-workers',type=int,default=2)
    p.add_argument('--backend',choices=['auto','cpu','cuda'],default='cpu')
    args=p.parse_args();run(args.directory,args.iterations,tuple(args.seeds),args.parallel_trials,args.fft_workers,args.backend)

if __name__=='__main__':main()
