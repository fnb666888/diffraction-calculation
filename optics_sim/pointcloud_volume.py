"""Depth-scanned holographic volume and 3-D focus-peak localization.

Inverse inputs are saved hologram/SLM commands and calibration only.  Source
geometry is used only in a separate optional evaluation after reconstruction.
The recovered points are estimates, not a unique closed-surface reconstruction.
"""
from pathlib import Path
import json
import numpy as np
from scipy import fft,ndimage
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import Grid,fft2c,ifft2c
from .holography import bandlimit
from .pointcloud_direct import recover_four

ROOT=Path(__file__).resolve().parents[1]

def recover_wave(recording,calibration):
    c=json.loads(Path(calibration).read_text(encoding='utf-8'))
    g=Grid(c['n'],c['dx_m'],c['wavelength_m']);ar=c['reference_amplitude']
    with np.load(recording) as p:
        if c['mode']=='cat':
            x,_=g.xy;R=ar*np.exp(2j*np.pi*c['carrier_per_m']*x)
            converging=bandlimit(p['amplitude_transmittance']*np.conj(R),g,c['bandwidth_per_m'])[0]
            wave=np.conj(converging*c['recording_intensity_peak']/ar**2)
        elif c['mode']=='horse':
            converging=p['slm_amplitude_8bit'].astype(float)/255*np.exp(2j*np.pi*p['slm_phase_8bit'].astype(float)/256)
            wave=np.conj(converging)
        else:
            wave=recover_four(p['camera_counts'],c['camera_intensity_peak'],ar)
    return wave,c

def reconstruct(recording,calibration,output):
    """One fixed search box for all three cases; no source point coordinates."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    wave,c=recover_wave(recording,calibration)
    # Retain all 120/mm object information at 2 μm sampling; discard exterior
    # quantization noise by the same specified physical/analysis pupil.
    old=len(wave);n=512;spec=fft2c(wave);r=(old-n)//2
    cropped=spec[r:r+n,r:r+n]*(n/old);dx=c['n']*c['dx_m']/n
    freq=fft.fftshift(fft.fftfreq(n,dx));q=freq[:,None]**2+freq[None,:]**2
    cropped*=q<=c['bandwidth_per_m']**2
    depths=np.linspace(.0025,.0035,101)
    xy=(np.arange(n)-n//2)*dx;roi=np.flatnonzero(np.abs(xy)<.00032)
    start,stop=int(roi[0]),int(roi[-1]+1)
    volume=[]
    for z in depths:
        H=np.exp(1j*np.pi*c['wavelength_m']*z*q)
        u=ifft2c(cropped*H)
        volume.append(np.abs(u[start:stop,start:stop])**2)
    intensity=np.asarray(volume,np.float32)
    # A disclosed local-contrast focus heuristic. Do not interpret this as
    # compressive inversion, tomography, or recovery of every original point.
    fine=ndimage.gaussian_filter(intensity,(0,.65,.65))
    broad=ndimage.gaussian_filter(intensity,(0,3.,3.))
    score=np.maximum(fine-broad,0)
    peak=float(score.max());threshold=.10*peak
    peaks=(score==ndimage.maximum_filter(score,size=(7,3,3),mode='constant'))&(score>=threshold)
    peaks[:3]=False;peaks[-3:]=False
    zi,yi,xi=np.where(peaks);values=score[zi,yi,xi]
    ax=xy[start:stop];points=np.column_stack([ax[xi],ax[yi],depths[zi]])
    np.savez_compressed(out/'volume.npz',intensity=intensity,focus_score=score,
       x_m=ax,depth_m=depths,estimated_xyz_m=points,peak_score=values)
    settings={'input_recording':str(Path(recording).resolve()),'input_calibration':str(Path(calibration).resolve()),
       'inverse_inputs':'saved hologram/SLM commands + calibration only; original point cloud not read',
       'search_depth_m':[.0025,.0035],'depth_count':101,'depth_step_m':float(depths[1]-depths[0]),
       'xy_pitch_m':dx,'xy_bounds_m':[float(ax[0]),float(ax[-1])],
       'method':'Fresnel depth stack; positive difference of Gaussian-smoothed intensities; 3-D local maxima',
       'focus_score_sigma_xy_pixels':[.65,3.],'local_max_window_zyx':[7,3,3],
       'score_threshold_fraction_of_global_max':.10,'exclude_search_depth_boundary_planes':3,
       'estimated_peak_count':len(points),'tomography':False,'compressive_optimization':False,
       'interpretation':'estimated 3-D focus centers; not the true source point cloud or a watertight surface',
       'limits':'single-view finite-bandwidth data; coherent interference, axial elongation and false/missed peaks remain'}
    (out/'reconstruction.json').write_text(json.dumps(settings,ensure_ascii=False,indent=2),encoding='utf-8')
    ply=['ply','format ascii 1.0',f'element vertex {len(points)}','property float x','property float y','property float z','property float focus_score','end_header']
    ply.extend(' '.join(f'{v:.9g}' for v in row) for row in np.column_stack([points,values]))
    (out/'estimated-focus-points.ply').write_text('\n'.join(ply)+'\n',encoding='ascii')
    return points,values,intensity,ax,depths,settings

def render(out,points,values,intensity,axis,depths):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12})
    d=points[:,2]*1000
    for stem,azim in [('view-a',-65),('view-b',-135)]:
        fig=plt.figure(figsize=(6.1,5.2),layout='constrained');ax=fig.add_subplot(projection='3d')
        p=ax.scatter(points[:,0]*1000,d,points[:,1]*1000,c=d,cmap='viridis',s=4,depthshade=False,vmin=2.5,vmax=3.5)
        ax.set(xlabel='x / mm',ylabel='距全息面 / mm',zlabel='y / mm',xlim=(-.32,.32),ylim=(2.5,3.5),zlim=(-.32,.32))
        ax.set_box_aspect((.64,1.,.64));ax.view_init(elev=18,azim=azim)
        fig.colorbar(p,ax=ax,shrink=.65,pad=.08,label='定位深度 / mm')
        fig.savefig(out/(stem+'.png'),dpi=180,facecolor='white');plt.close(fig)
    # The z-x maximum projection shows the actual reconstructed optical volume,
    # not point geometry or a photograph used to fabricate a 3-D result.
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    mip=intensity.max(axis=1);ax.imshow((mip/mip.max())**.35,cmap='gray',origin='lower',aspect='auto',extent=[axis[0]*1000,axis[-1]*1000,depths[0]*1000,depths[-1]*1000])
    ax.set(xlabel='x / mm',ylabel='重聚焦距离 / mm')
    fig.savefig(out/'depth-section.png',dpi=180,facecolor='white');plt.close(fig)

def evaluate(points,reference):
    """Post hoc only; metrics do not alter estimated points or parameters."""
    true=np.load(reference)['points_xyz_m']
    est_to_true=cKDTree(true).query(points)[0];true_to_est=cKDTree(points).query(true)[0]
    return {'evaluation_only':True,'estimate_to_source_median_um':float(np.median(est_to_true)*1e6),
       'source_to_estimate_median_um':float(np.median(true_to_est)*1e6),
       'estimate_within_30um_of_source_fraction':float(np.mean(est_to_true<30e-6)),
       'source_within_30um_of_estimate_fraction':float(np.mean(true_to_est<30e-6)),
       'not_one_to_one_point_identification':True}

def run():
    for name in ['cat','horse','wolf']:
        folder=ROOT/f'outputs/pointcloud_direct/{name}';out=folder/'volume-reconstruction'
        print('Reconstructing volume '+name,flush=True)
        points,values,vol,axis,depths,settings=reconstruct(folder/'recordings.npz',folder/'calibration.json',out)
        render(out,points,values,vol,axis,depths)
        metrics=evaluate(points,folder/'input-reference.npz')
        (out/'evaluation.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        print(name,json.dumps({'points':len(points),**metrics}),flush=True)

if __name__=='__main__':run()
