"""Direct 3-D coherent point-source holography; no tomography or phase screen.

Each point contributes a continuous-coordinate paraxial spherical wave.  The
three cases differ in recording/encoding and replay, not in geometry inversion.
Saved records/commands alone feed replay. Numerical refocusing uses one field.
"""
from pathlib import Path
import io,json,time
import numpy as np
from scipy import fft
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import Grid,propagate,fft2c,ifft2c
from .holography import bandlimit,carrier_conditions
from .metrics import relative_error

ROOT=Path(__file__).resolve().parents[1]
CASES={'cat':'光学离轴全息','horse':'复振幅计算全息','wolf':'单幅离轴数字全息'}

def load_points(name):
    body=(ROOT/f'data/pointcloud_holography/{name}.pcd').read_text(encoding='ascii').split('DATA ascii\n',1)[1]
    raw=np.loadtxt(io.StringIO(body))[:,:3];p=raw[:,[1,2,0]].astype(float)
    p-=(p.max(axis=0)+p.min(axis=0))/2
    scale=.0007/np.ptp(p,axis=0).max();p*=scale
    theta=np.deg2rad(40);x=p[:,0].copy();z=p[:,2].copy()
    p[:,0]=np.cos(theta)*x+np.sin(theta)*z
    p[:,2]=-np.sin(theta)*x+np.cos(theta)*z
    p[:,2]+=.003
    return p,scale

def point_field(points,g):
    """Sum a_j/z_j exp[ik(z_j-z0)+iπ((x-xj)^2+(y-yj)^2)/(λzj)]."""
    x,y,z=points.T;a=g.x;phase_z=2*np.pi*(z-.003)/g.wavelength
    hx=np.exp(1j*np.pi*(a[None,:]-x[:,None])**2/(g.wavelength*z[:,None]))
    hy=np.exp(1j*np.pi*(a[:,None]-y[None,:])**2/(g.wavelength*z[None,:]))
    hy*=np.exp(1j*phase_z)[None,:]/z[None,:]
    return hy@hx

def interpolate_wave(u,n):
    # Same physical window, band-limited interpolation with unitary FFT scaling.
    old=u.shape[0];pad=np.zeros((n,n),np.complex128);r=(n-old)//2
    pad[r:r+old,r:r+old]=fft2c(u)*(n/old)
    return ifft2c(pad)

def record_four(u,ar=1.5):
    intensity=np.array([np.abs(u+ar*np.exp(1j*t))**2 for t in [0,np.pi/2,np.pi,3*np.pi/2]])
    peak=float(intensity.max());return np.rint(intensity/peak*65535).astype(np.uint16),peak

def recover_four(counts,peak,ar):
    I=counts.astype(float)*(peak/65535)
    return ((I[0]-I[2])+1j*(I[1]-I[3]))/(4*ar)

def replay(folder):
    """Only saved recording/SLM commands and calibration enter this function."""
    c=json.loads((folder/'calibration.json').read_text(encoding='utf-8'))
    g=Grid(c['n'],c['dx_m'],c['wavelength_m']);x,_=g.xy;ar=c['reference_amplitude']
    with np.load(folder/'recordings.npz') as packet:
        if c['mode']=='cat':
            t=packet['amplitude_transmittance']
            R=ar*np.exp(2j*np.pi*c['carrier_per_m']*x)
            # Conjugate illumination gives the converging real-image wave.
            wave=bandlimit(t*np.conj(R),g,c['bandwidth_per_m'])[0]
            wave*=c['recording_intensity_peak']/ar**2
            sign=1
        elif c['mode']=='horse':
            amp=packet['slm_amplitude_8bit'].astype(float)/255
            phi=packet['slm_phase_8bit'].astype(float)*(2*np.pi/256)
            # Uniform illumination and ideal tandem amplitude/phase modulators.
            wave=amp*np.exp(1j*phi);sign=1
        else:
            if c.get('digital_recording') == 'single_offaxis':
                from .pointcloud_single_offaxis import recover_single
                wave=recover_single(packet['camera_counts'],c)
            else:
                wave=recover_four(packet['camera_counts'],c['camera_intensity_peak'],ar)
            sign=-1
    fields=np.array([propagate(wave,g,sign*z,method='fresnel',backend='scipy') for z in c['focus_distances_m']])
    return wave,fields,c

def gray(path,a,peak=None,gamma=1):
    v=np.clip(a/(float(a.max()) if peak is None else peak),0,1)**gamma
    Image.fromarray(np.flipud(np.rint(v*255).astype(np.uint8))).save(path)

def figures(folder,points,fields,c):
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12})
    with np.load(folder/'reconstructed.npz') as saved:sensor=saved['recovered_hologram_field']
    gray(folder/'recovered-intensity.png',np.abs(sensor)**2,gamma=.35)
    gray(folder/'recovered-phase.png',(np.angle(sensor)+np.pi)/(2*np.pi),1)
    d=points[:,2]*1000;near,far=d.min(),d.max()
    fig=plt.figure(figsize=(6,5),layout='constrained');ax=fig.add_subplot(projection='3d')
    ax.scatter(points[:,0]*1000,d,points[:,1]*1000,c=d,cmap='viridis',s=3,depthshade=False)
    ax.set(xlabel='x / mm',ylabel='距全息面 / mm',zlabel='y / mm');ax.set_box_aspect((1,.9,1));ax.view_init(elev=20,azim=-65)
    fig.savefig(folder/'pointcloud-input.png',dpi=180,facecolor='white');plt.close(fig)
    fig,ax=plt.subplots(figsize=(5,5),layout='constrained')
    v=ax.scatter(points[:,0]*1000,points[:,1]*1000,c=d,cmap='viridis',s=4)
    ax.set(xlabel='x / mm',ylabel='y / mm',aspect='equal');fig.colorbar(v,ax=ax,label='距全息面 / mm')
    fig.savefig(folder/'pointcloud-depth.png',dpi=180,facecolor='white');plt.close(fig)
    # A shared object window and common intensity scale across all focus planes.
    g=Grid(c['n'],c['dx_m'],c['wavelength_m']);roi=np.array([g.n//2-640,g.n//2+640]*2)
    # 640 μm square includes the rotated 700 μm model's full transverse extent.
    sl=np.s_[roi[0]:roi[1],roi[2]:roi[3]];I=np.abs(fields)**2;peak=float(I[:,sl[0],sl[1]].max())
    for j in range(3):gray(folder/f'focus-{j}.png',I[j][sl],peak,gamma=.35)
    fig,axs=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for j,ax in enumerate(axs):
        ax.imshow((I[j][sl]/peak)**.35,cmap='gray',vmin=0,vmax=1,origin='lower',extent=[-.32,.32,-.32,.32],interpolation='nearest')
        ax.set(title=f"z = {c['focus_distances_m'][j]*1000:.3f} mm",xlabel='x / mm',ylabel='y / mm')
    fig.savefig(folder/'focus-comparison.png',dpi=180,facecolor='white');plt.close(fig)
    (folder/'display.json').write_text(json.dumps({'intensity_reference':peak,'gamma':.35,'roi_rc':roi.tolist()},indent=2),encoding='utf-8')

def run_case(name):
    start=time.perf_counter();out=ROOT/f'outputs/pointcloud_direct/{name}';out.mkdir(parents=True,exist_ok=True)
    points,scale=load_points(name);coarse=Grid(1024,1e-6,632.8e-9);g=Grid(2048,.5e-6,coarse.wavelength)
    # Check the maximum local chirp frequency before coarse-grid sampling.
    extent=np.maximum(np.abs(points[:,0]),np.abs(points[:,1]))+coarse.size/2
    fmax=float(np.max(extent/(coarse.wavelength*points[:,2])))
    if fmax>=1/(2*coarse.dx):raise ValueError('Point spherical-wave sampling would alias')
    print(name+': summing 3400 spherical point waves',flush=True)
    raw=point_field(points,coarse);limited,_=bandlimit(raw,coarse,120000.)
    normalizer=float(np.abs(limited).max());wave=interpolate_wave(limited/normalizer,g.n)
    interpolation_peak=float(np.abs(wave).max());wave/=interpolation_peak;normalizer*=interpolation_peak
    foc=np.quantile(points[:,2],[.12,.5,.88]).tolist();ar=1.5;carrier=410/g.size
    c={'mode':name,'label':CASES[name],'n':g.n,'dx_m':g.dx,'wavelength_m':g.wavelength,
       'bandwidth_per_m':120000.,'carrier_per_m':carrier,'reference_amplitude':ar,
       'source_point_count':len(points),'uniform_geometric_scale':scale,'rotation_about_vertical_deg':40.,
       'point_distance_min_max_m':[float(points[:,2].min()),float(points[:,2].max())],
       'focus_distances_m':foc,'source_phase':'all point emitters have phase 0; retain their relative axial propagation phases',
       'source_amplitude':'equal coherent point source amplitudes; spherical 1/z decay',
       'point_sum':'continuous lateral/depth coordinates, separable Fresnel approximation to spherical waves',
       'common_axial_phase_removed_at_m':.003,'point_wave_max_sampling_frequency_per_m':fmax,
       'point_sum_grid':[1024,1e-6],'record_grid':[2048,.5e-6],
       'interpolation':'zero-pad unitary spectrum at constant physical window after 120/mm pupil',
       'optical_field_common_normalizer':normalizer,'retained_power_fraction':float(np.sum(np.abs(limited)**2)/np.sum(np.abs(raw)**2)),
       'single_view':True,'tomography':False,'noise':'ideal optical model; digital records 16-bit; no stochastic noise',
       'model_limits':'isotropic miniaturized coherent point emitters; no occlusion, material scattering or multiple scattering',
       'source':json.loads((ROOT/f'data/pointcloud_holography/{name}-source.json').read_text(encoding='utf-8'))}
    if name=='cat':
        x,_=g.xy;R=ar*np.exp(2j*np.pi*carrier*x);I=np.abs(wave+R)**2;peak=float(I.max())
        t=I/peak;c['recording_intensity_peak']=peak
        np.savez_compressed(out/'recordings.npz',amplitude_transmittance=t.astype(np.float64))
        gray(out/'hologram-record.png',I,peak)
        gray(out/'hologram-transmittance.png',t,1)
        target=np.conj(wave)
        c['replay']='conjugate reference illumination, physical 4f circular order selection, forward Fresnel propagation to real image'
        c['carrier_sampling']=carrier_conditions(g,carrier,120000.)
        if 2*carrier+120000>=1/(2*g.dx):raise ValueError('Replay aliases')
    elif name=='horse':
        target=np.conj(wave)
        a=np.rint(np.clip(np.abs(target),0,1)*255).astype(np.uint8)
        p=(np.rint(np.mod(np.angle(target),2*np.pi)*256/(2*np.pi)).astype(np.int32)%256).astype(np.uint8)
        np.savez_compressed(out/'recordings.npz',slm_amplitude_8bit=a,slm_phase_8bit=p)
        Image.fromarray(np.flipud(a)).save(out/'amplitude-command.png');Image.fromarray(np.flipud(p)).save(out/'phase-command.png')
        c['replay']='uniform illumination; ideal tandem amplitude 8-bit and phase 256-level SLM; forward propagation'
    else:
        x,_=g.xy;R=ar*np.exp(2j*np.pi*carrier*x);I=np.abs(wave+R)**2
        peak=float(I.max());counts=np.rint(I/peak*65535).astype(np.uint16)
        c.update(camera_intensity_peak=peak,digital_recording='single_offaxis',camera_frame_count=1)
        np.savez_compressed(out/'recordings.npz',camera_counts=counts)
        gray(out/'camera-single.png',counts,65535)
        target=wave;c['replay']='single off-axis exposure; FFT sideband selection and reference demodulation; negative-distance numerical Fresnel propagation'
    (out/'calibration.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
    del raw,limited
    print(name+': replaying saved hologram/commands',flush=True)
    recovered,fields,c=replay(out)
    metrics={'saved_record_replay_complex_relative_error':relative_error(recovered,target),
      'elapsed_seconds':time.perf_counter()-start,'scope':'complex-field relative L2 error at hologram plane; not 3-D geometry error or algorithm ranking'}
    np.savez_compressed(out/'input-reference.npz',points_xyz_m=points,reference_sensor_field=wave.astype(np.complex64))
    np.savez_compressed(out/'reconstructed.npz',recovered_hologram_field=recovered.astype(np.complex64),focus_fields=fields.astype(np.complex64),focus_distances_m=np.array(foc))
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    figures(out,points,fields,c)
    print(name+': '+json.dumps(metrics),flush=True)
    return c,metrics

if __name__=='__main__':
    for name in CASES:run_case(name)
