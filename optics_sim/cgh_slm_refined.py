"""Refined sampling and pupil for the existing analytic double-phase CGH.

The two original grayscale targets, zero object phase and 40/60 mm planes stay
fixed. This module changes numerical sampling and Fourier-pupil parameters,
without GS or post-processing the reconstructed data. Optional linear complex
field compensation constrains both target silhouettes before phase encoding.
"""
from pathlib import Path
import argparse
import gc
import json
import numpy as np
from PIL import Image
from scipy import fft
from scipy.sparse.linalg import LinearOperator,cg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,LogNorm
from matplotlib.patches import Circle
from .core import Grid
from .metrics import relative_error
from . import phase_comparison_figures

ROOT=Path(__file__).resolve().parents[1]
PROFILES={
 'legacy':dict(macro_n=512,macro_pitch_m=8e-6,subpixels=2,pupil_pass_per_m=18000.,pupil_stop_per_m=18000.),
 'dense36':dict(macro_n=512,macro_pitch_m=8e-6,subpixels=4,pupil_pass_per_m=36000.,pupil_stop_per_m=36000.),
 'dense48':dict(macro_n=512,macro_pitch_m=8e-6,subpixels=4,pupil_pass_per_m=48000.,pupil_stop_per_m=48000.),
 'soft48':dict(macro_n=512,macro_pitch_m=8e-6,subpixels=4,pupil_pass_per_m=30000.,pupil_stop_per_m=48000.),
 'padded48':dict(macro_n=1024,macro_pitch_m=8e-6,subpixels=4,pupil_pass_per_m=30000.,pupil_stop_per_m=48000.),
 'padded60':dict(macro_n=1024,macro_pitch_m=8e-6,subpixels=4,pupil_pass_per_m=40000.,pupil_stop_per_m=60000.),
 'padded60wide':dict(macro_n=1024,macro_pitch_m=8e-6,subpixels=4,pupil_pass_per_m=48000.,pupil_stop_per_m=60000.),
 'fine48':dict(macro_n=1024,macro_pitch_m=4e-6,subpixels=2,pupil_pass_per_m=30000.,pupil_stop_per_m=48000.),
 'fine48x4':dict(macro_n=1024,macro_pitch_m=4e-6,subpixels=4,pupil_pass_per_m=30000.,pupil_stop_per_m=48000.),
 'fine60padded':dict(macro_n=2048,macro_pitch_m=4e-6,subpixels=2,pupil_pass_per_m=40000.,pupil_stop_per_m=60000.),
 'fine60x4':dict(macro_n=1024,macro_pitch_m=4e-6,subpixels=4,pupil_pass_per_m=40000.,pupil_stop_per_m=60000.),
 'fine48comp':dict(macro_n=1024,macro_pitch_m=4e-6,subpixels=4,pupil_pass_per_m=30000.,pupil_stop_per_m=48000.,compensation_iterations=30,compensation_regularization=.02),
 'fine48compstrong':dict(macro_n=1024,macro_pitch_m=4e-6,subpixels=4,pupil_pass_per_m=30000.,pupil_stop_per_m=48000.,compensation_iterations=40,compensation_regularization=.005),
}


def propagate(a,g,z):
    f=fft.fftfreq(g.n,g.dx).astype(np.float32)
    kernel=np.exp(-1j*np.pi*g.wavelength*z*(f[:,None]**2+f[None,:]**2)).astype(np.complex64)
    s=fft.fft2(a,workers=2,overwrite_x=False);s*=kernel
    return fft.ifft2(s,workers=2,overwrite_x=True)


def pupil(g,passed,stopped):
    f=fft.fftfreq(g.n,g.dx).astype(np.float32)
    r=np.hypot(f[:,None],f[None,:])
    if stopped<=passed:return (r<=stopped).astype(np.float32)
    v=np.clip((r-passed)/(stopped-passed),0,1)
    return ((1+np.cos(np.pi*v))/2).astype(np.float32)


def select(a,p):
    s=fft.fft2(a,workers=2);s*=p
    return fft.ifft2(s,workers=2,overwrite_x=True)


def expanded(a,q):return np.repeat(np.repeat(a,q,axis=0),q,axis=1)


def compensate(initial,raw,masks,g,distances,config):
    """Regularized complex least squares in the hologram design stage.

    min sum_j ||M_j(P_zj Q H - Uj)||² + reg||H-H_initial||².
    CG acts on this Hermitian positive definite normal operator. The original
    zero-phase amplitudes are design targets, never replay inputs.
    """
    iterations=config.get('compensation_iterations',0)
    if not iterations:return initial,{'iterations':0,'regularization':None}
    reg=config['compensation_regularization']
    Q=pupil(g,config['pupil_pass_per_m'],config['pupil_stop_per_m'])
    def forward(v,d):return propagate(select(v,Q),g,d)
    def adjoint(v,d):return select(propagate(v,g,-d),Q)
    def normal(flat):
        v=flat.reshape(initial.shape);answer=reg*v
        for m,d in zip(masks,distances):answer=answer+adjoint(m*forward(v,d),d)
        return answer.ravel()
    rhs=reg*initial
    for target,d in zip(raw,distances):rhs=rhs+adjoint(target,d)
    count=[0]
    def callback(v):count[0]+=1
    operator=LinearOperator((initial.size,initial.size),matvec=normal,dtype=np.complex64)
    answer,status=cg(operator,rhs.ravel(),x0=initial.ravel(),rtol=1e-5,maxiter=iterations,callback=callback)
    h=answer.reshape(initial.shape)
    errors=[]
    for m,target,d in zip(masks,raw,distances):errors.append(relative_error(forward(h,d)[m],target[m]))
    return h,{'iterations':count[0],'maximum_iterations':iterations,'cg_status':int(status),'regularization':reg,
      'target_complex_foreground_errors_on_design_grid':errors,
      'objective':'sum squared foreground complex-field residuals at both object planes + regularization to initial backward-sum hologram',
      'solver':'conjugate gradient on regularized Hermitian normal equations; not Gerchberg-Saxton'}


def load_scene(g):
    data=ROOT/'data/cgh_slm/two_planes'
    rgb=np.asarray(Image.open(data/'input-color.png').convert('RGB'))
    depth=np.asarray(Image.open(data/'input-depth-levels.png').convert('L'))
    intensity=(rgb.astype(np.float32)@np.array([.2126,.7152,.0722],np.float32))/255
    source_scale=round(8e-6/g.dx)
    if not np.isclose(source_scale*g.dx,8e-6):raise ValueError('Input physical pitch must be preserved')
    working_depth=expanded(depth,source_scale);intensity=expanded(intensity,source_scale)
    r0=(g.n-working_depth.shape[0])//2;c0=(g.n-working_depth.shape[1])//2
    fields=[];masks=[]
    for value in sorted(x for x in np.unique(depth) if x>0):
        a=np.zeros((g.n,g.n),np.complex64);m=np.zeros(a.shape,bool)
        a[r0:r0+working_depth.shape[0],c0:c0+working_depth.shape[1]]=np.sqrt(intensity)*(working_depth==value)
        m[r0:r0+working_depth.shape[0],c0:c0+working_depth.shape[1]]=working_depth==value
        fields.append(a);masks.append(m)
    return fields,masks,rgb,depth


def replay_slm(command_path,calibration_path):
    """Replay only the exported SLM command plus known optical calibration."""
    c=json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    command=np.asarray(Image.open(command_path),np.float32)
    g=Grid(c['slm_n'],c['slm_pitch_m'],c['wavelength_m'])
    if command.shape!=(g.n,g.n):raise ValueError('SLM command shape mismatch')
    phase=command*(2*np.pi/c['phase_levels'])
    exit_wave=np.exp(1j*phase).astype(np.complex64)*c['incident_amplitude']
    p=pupil(g,c['pupil_pass_per_m'],c['pupil_stop_per_m'])
    chosen=select(exit_wave,p)
    del exit_wave,command
    return chosen,p,c


def compute(key,keep=False,out=None):
    cfg=PROFILES[key].copy();q=cfg['subpixels'];g=Grid(cfg['macro_n'],cfg['macro_pitch_m'],632.8e-9)
    fine=Grid(g.n*q,g.dx/q,g.wavelength);distances=[.040,.060]
    raw,masks,rgb,depth=load_scene(g)
    input_pupil=pupil(g,cfg['pupil_stop_per_m'],cfg['pupil_stop_per_m'])
    layers=[select(a,input_pupil) for a in raw]
    h=sum(propagate(a,g,-d) for a,d in zip(layers,distances))
    h,compensation=compensate(h,raw,masks,g,distances,cfg)
    scale=float(np.abs(h).max());h/=scale
    phi=np.angle(h);delta=np.arccos(np.clip(np.abs(h),0,1))
    plus=expanded(phi+delta,q);minus=expanded(phi-delta,q)
    yy,xx=np.ogrid[:fine.n,:fine.n];even=(yy+xx)%2==0
    phase=np.where(even,plus,minus);del plus,minus,even
    command=(np.rint(np.mod(phase,2*np.pi)*256/(2*np.pi)).astype(np.int16)%256).astype(np.uint8)
    del phase
    c={**cfg,'wavelength_m':g.wavelength,'slm_n':fine.n,'slm_pitch_m':fine.dx,
       'phase_levels':256,'incident_amplitude':1.,'plane_distances_m':distances,
       'complex_field_normalizer':scale,'macro_cell_slm_pixels':[q,q],
       'input_cutoff_per_m':cfg['pupil_stop_per_m'],'input_pixel_pitch_m':8e-6,
       'pupil_profile':'hard circle' if cfg['pupil_pass_per_m']==cfg['pupil_stop_per_m'] else 'amplitude raised-cosine edge',
       'generation':'two zero-phase target layers, backward Fresnel sum, optional regularized complex-field compensation',
       'compensation':compensation,
       'encoding':'analytic double phase, complementary checkerboard subpixels, 256 levels',
       'optical_replay':'uniform illumination, phase SLM, physical 4f low-pass pupil, positive-distance Fresnel propagation',
       'coordinates':'unit-magnification 4f relay, fixed inversion shown in upright coordinates',
       'distances_are_assigned_not_document_values':True,'precision':'complex64; physical gain restored from known normalization for target comparisons',
       'target_phase_rad':0.,'display_gamma':.30}
    if keep:
        out=Path(out);out.mkdir(parents=True,exist_ok=True)
        Image.fromarray(command).save(out/'slm-phase-8bit.png')
        (out/'calibration.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
        chosen,p,c=replay_slm(out/'slm-phase-8bit.png',out/'calibration.json')
    else:
        p=pupil(fine,cfg['pupil_pass_per_m'],cfg['pupil_stop_per_m'])
        chosen=select(np.exp(1j*(command.astype(np.float32)*(2*np.pi/256))).astype(np.complex64),p)
    ideal=select(expanded(h,q),p)
    encoding_error=relative_error(chosen,ideal)
    # Same physical object coordinates as legacy, independent of padding or q.
    scene_mask=np.logical_or.reduce(masks);rr,cc=np.where(scene_mask)
    roi=np.array([q*rr.min()-q*15,q*(rr.max()+1)+q*15,q*cc.min()-q*15,q*(cc.max()+1)+q*15],int)
    roi[[0,2]]=np.maximum(roi[[0,2]],0);roi[[1,3]]=np.minimum(roi[[1,3]],fine.n)
    roi_s=np.s_[roi[0]:roi[1],roi[2]:roi[3]]
    results=[];crops=[];ideal_crops=[];focus_crops=[];references=[];reference_crops=[]
    nonobject=~expanded(scene_mask,q)[roi_s]
    for j,d in enumerate(distances):
        u=propagate(chosen,fine,d)*scale;v=propagate(ideal,fine,d)*scale
        m=expanded(masks[j],q);target=np.abs(expanded(raw[j],q))**2
        i=np.abs(u)**2;vi=np.abs(v)**2
        r,co=np.where(m);margin=q*9
        focus_roi=(max(0,int(r.min())-margin),min(fine.n,int(r.max())+margin+1),max(0,int(co.min())-margin),min(fine.n,int(co.max())+margin+1))
        fs=np.s_[focus_roi[0]:focus_roi[1],focus_roi[2]:focus_roi[3]]
        results.append({'object':['cube','ball'][j],'distance_mm':d*1000,
          'foreground_intensity_error_to_original':relative_error(i[m],target[m]),
          'ideal_foreground_intensity_error_to_original':relative_error(vi[m],target[m]),
          'slm_to_ideal_foreground_intensity_error':relative_error(i[m],vi[m]),
          'nonobject_mean_intensity_relative_to_target_foreground_mean':float(i[roi_s][nonobject].mean()/target[m].mean()),
          'focus_roi_rc':focus_roi})
        crops.append(u[roi_s].copy());ideal_crops.append(v[roi_s].copy());focus_crops.append(i[fs].copy())
        reference_crops.append(target[fs].copy());references.append(target[roi_s].copy())
        del u,v,m,target,i,vi
    summary={'profile':key,'configuration':c,'encoding_field_relative_error':encoding_error,'planes':results,
       'metric_definition':'L2 intensity error inside original object silhouette, compared with original grayscale target at fixed physical size; known design gain, no fit or registration',
       'passed_power_fraction_of_incident':float(np.mean(np.abs(chosen)**2)),
       'method_changes':'denser sampling, soft pupil and optional regularized complex-field compensation in design; no target change, GS, output postprocessing or truth-dependent rescaling'}
    if keep:
        spectrum=np.abs(fft.fftshift(fft.fft2(np.exp(1j*(command.astype(np.float32)*(2*np.pi/256))).astype(np.complex64),workers=2)))**2
        spectrum/=spectrum.max()
        np.savez_compressed(out/'fields.npz',macro_x_m=g.x,x_m=fine.x,distances_m=np.asarray(distances),scene_roi_rc=roi,
          calculated_hologram=h,macro_layers=np.asarray(layers),optical_selected_wave=chosen,
          reconstructed_fields_roi=np.asarray(crops),ideal_fields_roi=np.asarray(ideal_crops),pupil_centered=fft.fftshift(p),
          scene_x_m=fine.x[roi[2]:roi[3]],scene_y_m=fine.x[roi[0]:roi[1]])
        (out/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
        render(out,c,command,spectrum,h,g,fine,roi,crops,ideal_crops,focus_crops,reference_crops,rgb,depth)
    del chosen,ideal,p
    return summary,focus_crops,reference_crops


def render(out,c,command,spectrum,h,g,fine,roi,crops,ideal_crops,focus_crops,reference_crops,rgb,depth):
    def save(fig,name):fig.savefig(out/(name+'.png'),dpi=185,bbox_inches='tight',pad_inches=.12,facecolor='white');plt.close(fig)
    def intensity(ax,value,title,ref,extent=None):
        im=ax.imshow(value/ref,cmap='gray',norm=PowerNorm(c['display_gamma'],vmin=0,vmax=1),extent=extent,interpolation='nearest')
        ax.set_title(title);cb=plt.colorbar(im,ax=ax,fraction=.04,pad=.025,ticks=[0,.01,.1,1]);cb.ax.set_yticklabels(['0','0.01','0.1','1']);cb.set_label('归一化光强')
        return im
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    axs[0].imshow(rgb);axs[0].set_title('双层物体输入图样');axs[0].axis('off')
    values=sorted(x for x in np.unique(depth) if x>0);dd=np.full(depth.shape,np.nan)
    for value,d in zip(values,[40,60]):dd[depth==value]=d
    v=axs[1].imshow(dd,cmap='viridis',vmin=40,vmax=60);axs[1].set_title('本次设置的两层距离');axs[1].axis('off');plt.colorbar(v,ax=axs[1],fraction=.04,pad=.03,ticks=[40,60],label='等效输出面到物体 / mm');save(fig,'input')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    # Crop to physical ±2 mm for readable wavefront detail; this is a display crop.
    ids=np.flatnonzero(np.abs(g.x)<=.002);hs=h[ids[0]:ids[-1]+1,ids[0]:ids[-1]+1];ext=[g.x[ids[0]]*1e6,g.x[ids[-1]]*1e6,g.x[ids[-1]]*1e6,g.x[ids[0]]*1e6]
    intensity(axs[0],np.abs(hs)**2,'全息面设计光强',float(np.abs(h).max()**2),ext)
    v=axs[1].imshow(np.angle(hs),cmap='RdBu_r',vmin=-np.pi,vmax=np.pi,extent=ext,interpolation='nearest');axs[1].set_title('全息面设计相位');cb=plt.colorbar(v,ax=axs[1],fraction=.04,pad=.025,ticks=[-np.pi,0,np.pi]);cb.ax.set_yticklabels(['−π','0','π']);cb.set_label('相位 / rad')
    for ax in axs:ax.set(xlabel='x / μm',ylabel='y / μm')
    save(fig,'calculated-hologram')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    v=axs[0].imshow(command,cmap='gray',vmin=0,vmax=255,interpolation='nearest');axs[0].set(title='加载至模拟SLM的相位图',xlabel='SLM列像素',ylabel='SLM行像素');plt.colorbar(v,ax=axs[0],fraction=.04,pad=.025,ticks=[0,128,255],label='相位命令 / 0–255')
    # Show all sampled frequencies, with separate full-pass and zero-pass edges.
    f=fine.frequency/1000;df=(f[1]-f[0])/2
    v=axs[1].imshow(spectrum,extent=[f[0]-df,f[-1]+df,f[-1]+df,f[0]-df],cmap='gray',norm=LogNorm(vmin=1e-9,vmax=1),interpolation='nearest')
    for radius,color,style in [(c['pupil_pass_per_m']/1000,'#174994','--'),(c['pupil_stop_per_m']/1000,'#D23B31','-')]:axs[1].add_patch(Circle((0,0),radius,fill=False,edgecolor=color,lw=1.6,linestyle=style))
    axs[1].set(title='SLM出射频谱与光瞳边缘',xlabel=r'$f_x$ / mm$^{-1}$',ylabel=r'$f_y$ / mm$^{-1}$');plt.colorbar(v,ax=axs[1],fraction=.04,pad=.025,label='归一化功率（对数）');save(fig,'slm-and-pupil')
    ext=[fine.x[roi[2]]*1e6,fine.x[roi[3]-1]*1e6,fine.x[roi[1]-1]*1e6,fine.x[roi[0]]*1e6]
    ref=max(float(np.abs(u).max()**2) for u in crops+ideal_crops)
    fig,axs=plt.subplots(1,2,figsize=(10,5.5),layout='constrained')
    for j,ax in enumerate(axs):intensity(ax,np.abs(crops[j])**2,f'SLM光学再现：{[40,60][j]} mm',ref,ext);ax.set(xlabel='x / μm',ylabel='y / μm')
    save(fig,'replay-two-planes')
    fig,axs=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for j in range(2):
        intensity(axs[0,j],np.abs(ideal_crops[j])**2,f'理想复调制参考：{[40,60][j]} mm',ref,ext)
        intensity(axs[1,j],np.abs(crops[j])**2,f'相位SLM再现：{[40,60][j]} mm',ref,ext)
    save(fig,'optical-replay')
    fig,axs=plt.subplots(2,2,figsize=(9,8),layout='constrained');ref=max(float(a.max()) for a in focus_crops+reference_crops)
    for j,label in enumerate(['魔方','足球']):
        intensity(axs[j,0],reference_crops[j],label+'灰度目标',ref)
        intensity(axs[j,1],focus_crops[j],label+'聚焦区域再现',ref)
        for ax in axs[j]:ax.set_xticks([]);ax.set_yticks([])
    save(fig,'target-comparison')


def sweep(keys=None):
    out=ROOT/'outputs/cgh_slm/repair_selection';out.mkdir(parents=True,exist_ok=True)
    all_results=[];previews=[];refs=None
    for key in keys or PROFILES:
        print('Computing '+key,flush=True)
        summary,crops,reference=compute(key)
        all_results.append(summary);previews.append(crops);refs=reference
        print(json.dumps({'profile':key,'errors':[p['foreground_intensity_error_to_original'] for p in summary['planes']],
          'encoding_error':summary['encoding_field_relative_error']},ensure_ascii=True),flush=True);gc.collect()
    (out/'selection.json').write_text(json.dumps(all_results,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,axs=plt.subplots(2,len(all_results),figsize=(4*len(all_results),8),layout='constrained')
    ref=max(float(a.max()) for pair in previews for a in pair)
    for k,(result,pair) in enumerate(zip(all_results,previews)):
        for j in range(2):
            axs[j,k].imshow(pair[j]/ref,cmap='gray',norm=PowerNorm(.30,vmin=0,vmax=1),interpolation='nearest')
            axs[j,k].set_title(result['profile']);axs[j,k].set_xlabel(f"误差 {100*result['planes'][j]['foreground_intensity_error_to_original']:.1f}%")
            axs[j,k].set_xticks([]);axs[j,k].set_yticks([])
    fig.savefig(out/'selection.png',dpi=150,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--sweep',action='store_true');parser.add_argument('--profiles',nargs='+',choices=PROFILES);parser.add_argument('--profile',choices=PROFILES,default='fine48comp');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.sweep:sweep(args.profiles)
    else:
        result,_,_=compute(args.profile,keep=True,out=args.output or ROOT/'outputs/cgh_slm/two_planes_refined')
        print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)
