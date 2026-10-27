"""Scientific evidence figures for the method explanation slides."""
from pathlib import Path
import json
import numpy as np
from scipy import fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from .core import Grid,propagate,fft2c
from .holography import reconstruct_off_axis
from .phase_comparison_figures import image

ROOT=Path(__file__).resolve().parents[1]


def run():
    data=ROOT/'outputs/phase_comparison/red_panda'
    out=ROOT/'outputs/method_explanation';out.mkdir(parents=True,exist_ok=True)
    manifest={}
    def save(fig,name,sources,extra=None):
        file=out/(name+'.png');fig.savefig(file,dpi=180,bbox_inches='tight',pad_inches=.12,facecolor='white');plt.close(fig)
        manifest[name]={'image':str(file),'sources':[str(s) for s in sources],**(extra or {})}
    with np.load(data/'cdi.npz') as d:
        fig,ax=plt.subplots(figsize=(11,4.2),layout='constrained')
        for key,label,color in [('zero','零相位','#174994'),('sine','正弦相位','#AE1324')]:
            h=d[key+'_selected_history'];ax.semilogy(np.arange(1,len(h)+1),h,label=label,color=color,lw=1.6)
        ax.axvspan(2950,3000,color='#AAAAAA',alpha=.25,label='末50步ER')
        ax.set(xlabel='迭代次数',ylabel='相对傅里叶模残差',xlim=(1,3000))
        ax.legend(loc='upper right');ax.grid(alpha=.18)
        save(fig,'cdi-convergence',[data/'cdi.npz'],{'definition':'|| |F(P_S u)|-sqrt(I_F)||2 / ||sqrt(I_F)||2'})
    with np.load(data/'holography.npz') as d:
        x=d['object_x_m'];roi=d['photo_roi_rc'];g=Grid(len(x),2.5e-6,632.8e-9)
        wave=reconstruct_off_axis(d['sine_hologram_intensity'],d['reference_wave'],g,44000.,1.5)
        distances=[.01,.02,.03]
        fields=[propagate(wave,g,-z,backend='scipy') for z in distances]
        fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
        ref=max(float(np.abs(u).max()**2) for u in fields)
        for ax,z,u in zip(axes,distances,fields):
            image(ax,np.abs(u)**2,x,roi,f'重建距离 {z*1000:.0f} mm',ref)
        save(fig,'distance-comparison',[data/'holography.npz'],{'reconstruction_distances_m':distances,'intensity_reference':ref,'gamma':.30,'autofocus':False})
        f=d['frequency_per_m']/1000;half=(f[1]-f[0])/2
        demod=d['sine_hologram_intensity']*d['reference_wave']/1.5**2
        spectrum=np.abs(fft2c(demod))**2
        selected=spectrum*d['pupil']
        keep=np.flatnonzero(np.abs(f)<=60);sl=np.s_[keep[0]:keep[-1]+1,keep[0]:keep[-1]+1]
        fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
        ref=float(spectrum.max())
        ext=[f[keep[0]]-half,f[keep[-1]]+half,f[keep[-1]]+half,f[keep[0]]-half]
        for ax,a,title in zip(axes,[spectrum,selected],['解调后的中心频区','圆形低通保留物光']):
            im=ax.imshow(a[sl]/ref,extent=ext,cmap='gray',norm=LogNorm(1e-8,1),interpolation='nearest')
            ax.set(title=title,xlabel=r'$f_x$ / mm$^{-1}$',ylabel=r'$f_y$ / mm$^{-1}$')
            plt.colorbar(im,ax=ax,fraction=.04,pad=.025,ticks=[1e-8,1e-4,1])
        save(fig,'holography-filter',[data/'holography.npz'],{'spectrum_reference':ref,'display':'log spectrum power','retained_bandwidth_per_m':44000})
    wrp=ROOT/'outputs/wrp_experiment/red_panda'
    with np.load(wrp/'data.npz') as d:
        x=d['object_x_m'];roi=d['photo_roi_rc']
        keys=['direct_fft','local_wrp_r6','local_wrp_r24'];titles=['单步 FFT 基准','WRP 13×13 窗口','WRP 49×49 窗口']
        fields=[d['sine_'+k+'_recovered'] for k in keys]
        ref=max(float(np.abs(u).max()**2) for u in fields)
        fig,axes=plt.subplots(2,3,figsize=(16,7.1),layout='constrained')
        for j,(u,title) in enumerate(zip(fields,titles)):
            image(axes[0,j],np.abs(u)**2,x,roi,title,ref)
            image(axes[1,j],np.angle(u),x,roi,'重建相位',phase=True)
        save(fig,'wrp-reconstruction',[wrp/'data.npz'],{'phase_condition':'sine','intensity_reference':ref,'gamma':.30,'phase_mask':None})
    (out/'figure-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v['image'] for k,v in manifest.items()},ensure_ascii=True))


if __name__=='__main__':run()
