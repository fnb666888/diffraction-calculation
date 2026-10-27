"""Scientific comparison plots; display gamma never changes saved fields."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]
GAMMA=.30
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
 'axes.unicode_minus':False,'font.size':14,'axes.titlesize':18,'axes.labelsize':12,
 'xtick.labelsize':10,'ytick.labelsize':10,'savefig.dpi':175})

def crop(a,roi):
    r0,r1,c0,c1=roi;return a[r0:r1,c0:c1]

def image(ax,a,x,roi,title,ref=None,phase=False):
    r0,r1,c0,c1=roi;half=(x[1]-x[0])/2
    value=crop(a,roi);ext=np.array([x[c0]-half,x[c1-1]+half,x[r1-1]+half,x[r0]-half])*1e6
    if phase:
        cmap=plt.get_cmap('RdBu_r')
        im=ax.imshow(value,origin='upper',extent=ext,cmap=cmap,vmin=-np.pi,vmax=np.pi,interpolation='nearest')
        cb=plt.colorbar(im,ax=ax,fraction=.04,pad=.025,ticks=[-np.pi,0,np.pi])
        cb.ax.set_yticklabels(['−π','0','π']);cb.set_label('相位 / rad')
    else:
        im=ax.imshow(value/ref,origin='upper',extent=ext,cmap='gray',
           norm=PowerNorm(GAMMA,vmin=0,vmax=1),interpolation='nearest')
        cb=plt.colorbar(im,ax=ax,fraction=.04,pad=.025,ticks=[0,.01,.1,1])
        cb.ax.set_yticklabels(['0','0.01','0.1','1']);cb.set_label('归一化光强')
    ax.set(title=title,xlabel='x / μm',ylabel='y / μm')
    ax.locator_params(axis='both',nbins=4)
    return im

def rowlabels(fig,axes,offset=.24):
    # Text on the left of each row, without modifying scientific image geometry.
    for i,label in enumerate(['零相位','正弦相位']):
        axes[i,0].text(-offset,.5,label,transform=axes[i,0].transAxes,
          rotation=90,ha='center',va='center',fontsize=19,color='#174994',fontweight='bold')

def render(directory):
    directory=Path(directory);out=directory/'figures';out.mkdir(parents=True,exist_ok=True)
    manifest={}
    def save(fig,key,source,extra=None):
        file=out/(key+'.png');fig.savefig(file,bbox_inches='tight',pad_inches=.15,facecolor='white');plt.close(fig)
        manifest[key]={'image':str(file.resolve()),'sources':[str(Path(source).resolve())],
          'gamma':GAMMA,**(extra or {})};print(key,flush=True)
    with np.load(directory/'holography.npz') as d:
        x=d['object_x_m'];roi=d['photo_roi_rc'];a=d['input_amplitude']
        fig,axes=plt.subplots(1,3,figsize=(16,4.5),layout='constrained')
        image(axes[0],a*a,x,roi,'共同输入光强',float(a.max()**2))
        image(axes[1],np.zeros_like(a),x,roi,'零相位',phase=True)
        image(axes[2],d['prescribed_phase_rad'],x,roi,'二维正弦相位',phase=True)
        save(fig,'input-phase',directory/'holography.npz')
        fig,axes=plt.subplots(2,2,figsize=(13,7.1),layout='constrained')
        ref=max(float(d[key+'_hologram_intensity'].max()) for key in ['zero','sine'])
        sref=max(float(d[key+'_hologram_spectrum'].max()) for key in ['zero','sine'])
        f=d['frequency_per_m']/1000;step=(f[1]-f[0])/2
        for i,key in enumerate(['zero','sine']):
            image(axes[i,0],d[key+'_hologram_intensity'],x,roi,'全息干涉光强' if i==0 else '',ref)
            p=d[key+'_hologram_spectrum']/sref
            im=axes[i,1].imshow(p,origin='upper',extent=[f[0]-step,f[-1]+step,f[-1]+step,f[0]-step],
                cmap='gray',norm=LogNorm(vmin=1e-8,vmax=1),interpolation='nearest')
            axes[i,1].add_patch(Circle((-144.53125,0),44,fill=False,edgecolor='#D23B31',linewidth=1.8))
            axes[i,1].set(title='全息图二维频谱' if i==0 else '',xlabel=r'$f_x$ / mm$^{-1}$',ylabel=r'$f_y$ / mm$^{-1}$')
            axes[i,1].locator_params(axis='both',nbins=4)
            cb=plt.colorbar(im,ax=axes[i,1],fraction=.04,pad=.025,ticks=[1e-8,1e-4,1]);cb.set_label('归一化频谱功率（对数）')
        rowlabels(fig,axes);save(fig,'holography-recording',directory/'holography.npz',{'intensity_reference':ref,'spectrum_reference':sref})
        fig,axes=plt.subplots(2,3,figsize=(16,7.1),layout='constrained')
        ref=max(float(np.abs(d[k+'_recovered_field']).max()**2) for k in ['zero','sine'])
        for i,key in enumerate(['zero','sine']):
            truth=d[key+'_target_field'];rec=d[key+'_recovered_field']
            image(axes[i,0],np.abs(rec)**2,x,roi,'重建光强' if i==0 else '',ref)
            image(axes[i,1],np.angle(truth),x,roi,'目标相位（带限后）' if i==0 else '',phase=True)
            image(axes[i,2],np.angle(rec),x,roi,'重建相位' if i==0 else '',phase=True)
        rowlabels(fig,axes);save(fig,'holography-reconstruction',directory/'holography.npz',{'intensity_reference':ref})
    with np.load(directory/'abbe.npz') as d:
        x=d['object_x_m'];roi=d['photo_roi_rc'];labels=['none','lowpass','highpass']
        ref=max(float(np.abs(d[k+'_'+f+'_oriented_field']).max()**2) for k in ['zero','sine'] for f in labels)
        fig,axes=plt.subplots(2,3,figsize=(16,7.1),layout='constrained')
        for i,key in enumerate(['zero','sine']):
            for j,(label,title) in enumerate(zip(labels,['无附加滤波','圆形低通','圆形高通'])):
                image(axes[i,j],np.abs(d[key+'_'+label+'_oriented_field'])**2,x,roi,title if i==0 else '',ref)
        rowlabels(fig,axes);save(fig,'abbe-comparison',directory/'abbe.npz',{'intensity_reference':ref,
          'orientation':'physical inverted image rotated 180 degrees to align with input for comparison'})
    if (directory/'cdi.npz').exists():
        with np.load(directory/'cdi.npz') as d:
            if 'sine_recovered_aligned_field' in d:
                x=d['object_x_m'];roi=d['photo_roi_rc'];ref=max(float(np.abs(d[k+'_recovered_aligned_field']).max()**2) for k in ['zero','sine'])
                iref=max(float(d[k+'_diffraction_intensity'].max()) for k in ['zero','sine'])
                fig,axes=plt.subplots(2,3,figsize=(16,7.1),layout='constrained')
                f=d['frequency_per_m']/1000;step=(f[1]-f[0])/2
                phase_stats={}
                display_frequency_halfwidth_mm=30
                center=np.flatnonzero(np.abs(f)<=display_frequency_halfwidth_mm)
                band=np.s_[center[0]:center[-1]+1,center[0]:center[-1]+1]
                edge=[f[center[0]]-step,f[center[-1]]+step,f[center[-1]]+step,f[center[0]]-step]
                for i,key in enumerate(['zero','sine']):
                    im=axes[i,0].imshow(d[key+'_diffraction_intensity'][band]/iref,origin='upper',
                      extent=edge,cmap='gray',norm=PowerNorm(GAMMA,vmin=0,vmax=1),interpolation='nearest')
                    axes[i,0].set(title='衍射光强（中心频区）' if i==0 else '',xlabel=r'$f_x$ / mm$^{-1}$',ylabel=r'$f_y$ / mm$^{-1}$')
                    axes[i,0].locator_params(axis='both',nbins=4)
                    cb=plt.colorbar(im,ax=axes[i,0],fraction=.04,pad=.025,ticks=[0,.01,.1,1]);cb.ax.set_yticklabels(['0','0.01','0.1','1']);cb.set_label('归一化光强')
                    rec=d[key+'_recovered_aligned_field'];truth=d[key+'_truth_field']
                    phase_stats[key]=float(np.sqrt(np.mean(crop(np.angle(rec*np.conj(truth)),roi)**2)))
                    image(axes[i,1],np.abs(rec)**2,x,roi,'CDI恢复光强' if i==0 else '',ref)
                    image(axes[i,2],np.angle(rec),x,roi,'CDI恢复相位' if i==0 else '',phase=True)
                rowlabels(fig,axes,offset=.40);save(fig,'cdi-comparison',directory/'cdi.npz',{'intensity_reference':ref,
                    'diffraction_reference':iref,'phase_mask':None,'phase_rmse_full_photo_rad':phase_stats,
                    'display_frequency_halfwidth_per_mm':display_frequency_halfwidth_mm,
                    'inversion_uses_full_diffraction_grid':True,
                    'alignment':'translation/conjugate/global-phase ambiguities removed only for evaluation/display'})
    (out/'figure-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    return manifest

def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=ROOT/'outputs/phase_comparison/red_panda')
    args=p.parse_args();render(args.directory)

if __name__=='__main__':main()
