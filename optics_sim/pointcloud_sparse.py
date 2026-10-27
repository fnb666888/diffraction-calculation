"""Complex L1-regularized single-view 3-D holographic inverse (FISTA).

The inverse uses only recordings and calibration; source geometry is confined
to a separate post-hoc assessment. No source shape mask, depth fit, truth gain,
or true phase enters inversion. Results are a regularized volume estimate.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy import fft,ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import fft2c
from .pointcloud_volume import recover_wave,reconstruct as focus_stack

ROOT=Path(__file__).resolve().parents[1]

def reconstruct(recording,calibration,output,iterations=250,regularization=.015,thin_prior=True):
    out=Path(output);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    u,c=recover_wave(recording,calibration);old=len(u);n=256;dx=c['n']*c['dx_m']/n
    s=fft2c(u);r=(old-n)//2;b=fft.ifftshift(s[r:r+n,r:r+n]*(n/old)).astype(np.complex64)
    del s,u
    freq=fft.fftfreq(n,dx);q=freq[:,None]**2+freq[None,:]**2
    pupil=q<=c['bandwidth_per_m']**2;b*=pupil
    depths=np.linspace(.0025,.0035,41)
    H=np.exp(-1j*np.pi*c['wavelength_m']*depths[:,None,None]*q[None]).astype(np.complex64)*pupil
    # Instrument field of view, same 640 μm square for all three data sets.
    xy=np.fft.ifftshift((np.arange(n)-n//2)*dx)
    support=(np.abs(xy[:,None])<=.00032)&(np.abs(xy[None,:])<=.00032)
    prior=None
    if thin_prior:
        # Fit a dominant x-depth plane to the independent focus score. This is
        # an explicit thin-object prior, not a recovered generic solid volume.
        _,_,_,scan_x,scan_depths,_=focus_stack(recording,calibration,out/'focus-scan')
        scan=np.load(out/'focus-scan/volume.npz');score=scan['focus_score']
        profile=ndimage.gaussian_filter1d(score.sum(axis=1),1.,axis=1)
        profile=(profile-profile.mean(axis=0))/(profile.std(axis=0)+1e-8)
        strength=score.max(axis=0).sum(axis=0);strength/=strength.max();strength[strength<.08]=0
        candidates=[];cols=np.arange(len(scan_x))
        for slope in np.arange(-1.5,1.51,.05):
            for offset in np.arange(.0027,.003301,.00001):
                t=(offset+slope*scan_x-scan_depths[0])/(scan_depths[1]-scan_depths[0]);lo=np.floor(t).astype(int);f=t-lo
                good=(lo>=0)&(lo+1<len(scan_depths));ii=lo.clip(0,len(scan_depths)-2)
                value=(profile[ii,cols]*(1-f)+profile[ii+1,cols]*f)*good
                candidates.append((float((value*strength).sum()),float(slope),float(offset)))
        merit,slope,offset=max(candidates)
        support=support[None]&(np.abs(depths[:,None,None]-(offset+slope*xy[None,None,:]))<=.00012)
        prior={'kind':'thin cloud around a dominant x-depth plane; vertical slope assumed negligible',
          'plane_estimated_from':'normalized local-contrast focus stack of the saved holographic data',
          'plane_depth_m':'offset + slope*x','slope':slope,'offset_m':offset,'half_thickness_m':.00012,
          'slope_search':[-1.5,1.5,.05],'offset_search_m':[.0027,.0033,.00001],
          'plane_merit':merit,'source_geometry_used':False}
        del score,profile,scan
    def forward(v):return (fft.fft2(v,axes=(-2,-1),norm='ortho',workers=4)*H).sum(axis=0)
    def adjoint(v):return fft.ifft2(v[None]*np.conj(H),axes=(-2,-1),norm='ortho',workers=4)*support
    back=adjoint(b);lam=float(np.abs(back).max()*regularization)
    L=len(depths);step=np.float32(1/L)
    x=np.zeros_like(back);y=x.copy();t=1.;history=[];norm=float(np.linalg.norm(b))
    for it in range(iterations):
        residual=forward(y)-b
        z=y-step*adjoint(residual);mag=np.abs(z)
        candidate=z*np.maximum(0,1-step*lam/np.maximum(mag,1e-30))
        tnext=(1+np.sqrt(1+4*t*t))/2
        y=candidate+np.float32((t-1)/tnext)*(candidate-x)
        x=candidate;t=tnext
        if (it+1)%25==0 or it==0:
            error=float(np.linalg.norm(forward(x)-b)/norm)
            history.append({'iteration':it+1,'complex_data_relative_error':error})
            print(Path(output).parent.name,it+1,error,flush=True)
    x=fft.fftshift(x,axes=(-2,-1));density=np.abs(x)
    # Select spatial local maxima of the estimated scattering magnitude for a
    # readable PLY. Save the whole complex volume for threshold-independent use.
    active=(density==ndimage.maximum_filter(density,size=(3,3,3),mode='constant'))&(density>=.12*float(density.max()))
    active[[0,-1]]=False
    zi,yi,xi=np.where(active);axis=(np.arange(n)-n//2)*dx
    points=np.column_stack([axis[xi],axis[yi],depths[zi]]);weights=density[zi,yi,xi]
    np.savez_compressed(out/'sparse-volume.npz',complex_scattering_volume=x.astype(np.complex64),
       x_m=axis,depth_m=depths,estimated_xyz_m=points,estimated_magnitude=weights)
    config={'inverse_inputs':'saved hologram/SLM commands and calibration only',
       'operator':'sum of 41 Fresnel-propagated complex object slices, common 120/mm pupil',
       'objective':'0.5 ||A o - b||_2^2 + lambda ||o||_1 for complex voxels',
       'algorithm':'FISTA with complex soft thresholding; step 1/Nz is a safe spectral norm bound',
       'lambda':lam,'lambda_fraction_of_max_abs_adjoint_data':regularization,'iterations':iterations,
       'grid_zyx':[41,n,n],'xy_pitch_m':dx,'depth_step_m':25e-6,
       'search_depth_m':[.0025,.0035],'instrument_fov_xy_m':[-.00032,.00032],
       'scatter_display':'local maxima, 12% of estimated volume peak; first/last depth excluded',
       'estimated_point_count':len(points),'data_relative_error':history[-1]['complex_data_relative_error'],
       'history':history,'elapsed_seconds':time.perf_counter()-start,
       'truth_shape_or_support_used':False,'multiple_views':False,'thin_object_prior':prior,
       'interpretation':'regularized complex scattering estimate; not unique geometry or a watertight mesh',
       'limitations':['finite aperture/bandwidth and single view produce axial ambiguity',
        'complex voxel coefficients approximate continuously positioned coherent sources',
        'Fresnel discrete periodic operator approximates the finite-window point-wave model',
        'sparsity and regularization can remove weak points and create biased amplitudes/depths']}
    (out/'reconstruction.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['ply','format ascii 1.0',f'element vertex {len(points)}','property float x','property float y','property float z','property float scattering_magnitude','end_header']
    lines.extend(' '.join(f'{v:.9g}' for v in a) for a in np.column_stack([points,weights]))
    (out/'estimated-points.ply').write_text('\n'.join(lines)+'\n',encoding='ascii')
    return points,weights,density,axis,depths,config

def render(out,points,weights,density,axis,depths):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12})
    for name,azim in [('view-a',-65),('view-b',-135)]:
        fig=plt.figure(figsize=(6,5.2),layout='constrained');ax=fig.add_subplot(projection='3d')
        p=ax.scatter(points[:,0]*1000,points[:,2]*1000,points[:,1]*1000,c=points[:,2]*1000,cmap='viridis',s=8,depthshade=False,vmin=2.5,vmax=3.5)
        ax.set(xlabel='x / mm',ylabel='距全息面 / mm',zlabel='y / mm',xlim=(-.32,.32),ylim=(2.5,3.5),zlim=(-.32,.32));ax.set_box_aspect((.64,1.,.64));ax.view_init(elev=18,azim=azim)
        fig.colorbar(p,ax=ax,shrink=.65,pad=.08,label='恢复深度 / mm')
        fig.savefig(out/(name+'.png'),dpi=180,facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    p=ax.scatter(points[:,0]*1000,points[:,1]*1000,c=points[:,2]*1000,cmap='viridis',s=8,vmin=2.5,vmax=3.5)
    ax.set(xlabel='x / mm',ylabel='y / mm',aspect='equal',xlim=(-.32,.32),ylim=(-.32,.32));fig.colorbar(p,ax=ax,label='恢复深度 / mm')
    fig.savefig(out/'depth-projection.png',dpi=180,facecolor='white');plt.close(fig)
    fig=plt.figure(figsize=(13,4.6),facecolor='white')
    for j,azim in enumerate([-135,-65]):
        ax=fig.add_axes([.01+j*.465,.035,.445,.96],projection='3d')
        p=ax.scatter(points[:,0]*1000,points[:,2]*1000,points[:,1]*1000,c=points[:,2]*1000,cmap='viridis',s=7,depthshade=False,vmin=2.5,vmax=3.5)
        ax.set(xlabel='x / mm',ylabel='距全息面 / mm',zlabel='y / mm',xlim=(-.32,.32),ylim=(2.5,3.5),zlim=(-.32,.32))
        ax.set_box_aspect((.64,1.,.64));ax.view_init(elev=18,azim=azim)
        ax.set_xticks([-.3,0,.3]);ax.set_yticks([2.5,3,3.5]);ax.set_zticks([-.3,0,.3])
    cb=fig.add_axes([.95,.17,.012,.67]);fig.colorbar(p,cax=cb,label='恢复深度 / mm')
    fig.savefig(out/'presentation-views.png',dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.04);plt.close(fig)

def run(name,iterations=250):
    folder=ROOT/f'outputs/pointcloud_direct/{name}';out=folder/'thin-prior-reconstruction'
    points,w,density,axis,depths,c=reconstruct(folder/'recordings.npz',folder/'calibration.json',out,iterations)
    render(out,points,w,density,axis,depths)
    # Post-hoc assessment only. Do not use it to modify the estimate.
    source=np.load(folder/'input-reference.npz')['points_xyz_m']
    distances=cKDTree(source).query(points)[0]
    back=cKDTree(points).query(source)[0]
    metrics={'evaluation_only':True,'estimate_to_source_median_um':float(np.median(distances)*1e6),
       'source_to_estimate_median_um':float(np.median(back)*1e6),
       'estimate_within_30um_fraction':float(np.mean(distances<30e-6)),
       'source_within_30um_fraction':float(np.mean(back<30e-6))}
    (out/'evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(name,json.dumps({'points':len(points),**metrics}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--case',default='all');p.add_argument('--iterations',type=int,default=250);a=p.parse_args()
    for name in (['cat','horse','wolf'] if a.case=='all' else [a.case]):run(name,a.iterations)
