"""Batched double-precision HIO/ER on CUDA, with CPU-compatible checkpoints."""
import hashlib
import time
import numpy as np

def recover_trials(intensities,roi,iterations,seeds,work,callback,tolerance=1e-8):
    import cupy as cp
    jobs=[(key,seed) for key in ['zero','sine'] for seed in seeds]
    count=len(jobs);n=next(iter(intensities.values())).shape[0]
    r0,r1,c0,c1=map(int,roi);support=np.zeros((n,n),bool);support[r0:r1,c0:c1]=True
    amplitudes=[];currents=[];bests=[];histories=[];best_res=[];polishes=[];previous=[];signatures=[]
    for key,seed in jobs:
        intensity=intensities[key];amplitude=np.sqrt(intensity);amplitudes.append(amplitude)
        signature=hashlib.sha256(intensity.tobytes()+np.asarray(roi).tobytes()).hexdigest();signatures.append(signature)
        checkpoint=work/f'{key}-{seed}.npz';loaded=False
        if checkpoint.exists():
            with np.load(checkpoint) as d:
                if str(d['signature'])==signature and int(d['budget'])==iterations:
                    currents.append(d['current']);bests.append(d['best']);histories.append(d['history'].tolist())
                    best_res.append(float(d['best_residual']));polishes.append(int(d['polish_remaining']))
                    previous.append(float(d['elapsed']));loaded=True
        if not loaded:
            rng=np.random.default_rng(seed)
            currents.append(np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(amplitude*np.exp(1j*rng.uniform(-np.pi,np.pi,(n,n)))),norm='ortho')))
            bests.append(np.zeros((n,n),complex));histories.append([]);best_res.append(np.inf);polishes.append(-1);previous.append(0.)
    u=cp.asarray(np.fft.ifftshift(np.stack(currents),axes=(-2,-1)))
    best=cp.asarray(np.fft.ifftshift(np.stack(bests),axes=(-2,-1)))
    amp=cp.asarray(np.fft.ifftshift(np.stack(amplitudes),axes=(-2,-1)))
    den=cp.sqrt(cp.sum(amp*amp,axis=(-2,-1)));mask=cp.asarray(np.fft.ifftshift(support))[None,:,:]
    del currents,bests,amplitudes
    steps=np.asarray([len(h) for h in histories]);best_res=np.asarray(best_res);polishes=np.asarray(polishes)
    active=(steps<iterations)&(polishes!=0);began=time.perf_counter();saved=np.zeros(count,bool)
    def checkpoint(i,complete):
        key,seed=jobs[i];file=work/f'{key}-{seed}.npz';temp=file.with_suffix('.tmp.npz')
        np.savez(temp,signature=signatures[i],budget=iterations,
          current=cp.asnumpy(cp.fft.fftshift(u[i])),best=cp.asnumpy(cp.fft.fftshift(best[i])),
          history=np.asarray(histories[i]),best_residual=best_res[i],step=steps[i],
          polish_remaining=polishes[i],complete=complete,elapsed=previous[i]+time.perf_counter()-began)
        temp.replace(file)
    print('CUDA device: '+cp.cuda.runtime.getDeviceProperties(0)['name'].decode()+', complex128',flush=True)
    while active.any():
        spectrum=cp.fft.fft2(u,axes=(-2,-1),norm='ortho');magnitude=cp.abs(spectrum)
        denominator=cp.where(magnitude>0,magnitude,1.)
        unit=cp.where(magnitude>0,spectrum/denominator,1.+0j)
        projected=cp.fft.ifft2(amp*unit,axes=(-2,-1),norm='ortho')
        er=cp.asarray(((steps>=iterations-50)|(polishes>=0))[:,None,None])
        updated=cp.where(mask,projected,cp.where(er,0,u-.9*projected))
        u=cp.where(cp.asarray(active[:,None,None]),updated,u)
        candidate=cp.where(mask,projected,0)
        delta=cp.abs(cp.fft.fft2(candidate,axes=(-2,-1),norm='ortho'))-amp
        residuals=cp.asnumpy(cp.sqrt(cp.sum(delta*delta,axis=(-2,-1)))/den)
        for i,(key,seed) in enumerate(jobs):
            if not active[i]:continue
            residual=float(residuals[i]);histories[i].append(residual);steps[i]+=1
            if residual<best_res[i]:best_res[i]=residual;best[i]=candidate[i]
            if polishes[i]>0:polishes[i]-=1
            elif polishes[i]<0 and best_res[i]<=tolerance:polishes[i]=50
            complete=steps[i]>=iterations or polishes[i]==0
            if steps[i]%50==0 or complete:callback(key,seed,int(steps[i]),float(best_res[i]),previous[i]+time.perf_counter()-began)
            if steps[i]%250==0 or complete:checkpoint(i,complete)
            if complete:
                active[i]=False;saved[i]=True
                print(f'completed {key} seed={seed}: iterations={steps[i]}, residual={best_res[i]:.6g}',flush=True)
    results={'zero':[],'sine':[]}
    for i,(key,seed) in enumerate(jobs):
        results[key].append((float(best_res[i]),seed,cp.asnumpy(cp.fft.fftshift(best[i])),np.asarray(histories[i]),previous[i]+time.perf_counter()-began))
    return results
