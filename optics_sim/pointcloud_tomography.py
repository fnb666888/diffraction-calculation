"""Three point-cloud holography cases with independently reconstructed volumes.

Point geometry is mapped to a smooth weak refractive-index surface density.
Straight-ray phase projections are holographically recorded at many sample
angles. Recovered phase projections, not the source points, enter FBP. This is
projection-approximation holographic tomography, not full diffraction tomography
or a claim of recovering an entire object from one hologram.
"""
from pathlib import Path
import argparse
import base64
import io
import json
import hashlib
import time
import numpy as np
from scipy import fft,ndimage
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import Grid,propagate,fft2c,ifft2c
from .holography import bandlimit
from .metrics import relative_error
from . import phase_comparison_figures

ROOT=Path(__file__).resolve().parents[1]
CASES={'cat':('光学离轴全息','optical'),'horse':('复振幅计算全息','cgh'),'wolf':('相移数字全息','digital')}


def source_volume(name,n=96):
    text=(ROOT/f'data/pointcloud_holography/{name}.pcd').read_text(encoding='ascii')
    raw=np.loadtxt(io.StringIO(text.split('DATA ascii\n',1)[1]))[:,:3]
    # Dataset x is the thin/depth direction, y longitudinal, z vertical.
    xyz=raw[:,[1,2,0]].astype(np.float32)
    center=(xyz.max(axis=0)+xyz.min(axis=0))/2;xyz-=center
    scale=(n*.38)/np.linalg.norm(xyz,axis=1).max();voxel_xyz=xyz*scale+(n-1)/2
    volume=np.zeros((n,n,n),np.float32)
    low=np.floor(voxel_xyz).astype(int);fraction=voxel_xyz-low
    for dx in [0,1]:
        for dy in [0,1]:
            for dz in [0,1]:
                weight=np.prod(np.where(np.array([dx,dy,dz]),fraction,1-fraction),axis=1)
                index=low+np.array([dx,dy,dz]);np.add.at(volume,(index[:,1],index[:,2],index[:,0]),weight)
    volume=ndimage.gaussian_filter(volume,.9)
    volume/=volume.max()
    return volume,(voxel_xyz-(n-1)/2)/n,{'source_point_count':len(raw),'input_axes':'PCL (y,z,x) becomes (horizontal,vertical,depth)',
      'uniform_scale_voxels_per_source_unit':float(scale),'source_center':center.tolist(),'surface_density_gaussian_sigma_voxels':.9}


def project(volume,angle):
    return ndimage.rotate(volume,float(angle),axes=(1,2),reshape=False,order=1,mode='constant',cval=0.,prefilter=False).sum(axis=1,dtype=np.float32)


def four_step_record(field,reference_amplitude=1.5):
    values=np.asarray([np.abs(field+reference_amplitude*np.exp(1j*p))**2 for p in [0,np.pi/2,np.pi,3*np.pi/2]],np.float32)
    peak=float(values.max())
    counts=np.rint(values/peak*65535).astype(np.uint16)
    return counts,peak


def four_step_recover(counts,peak,reference_amplitude):
    f=counts.astype(np.float32)*(peak/65535)
    return ((f[0]-f[2])+1j*(f[1]-f[3]))/(4*reference_amplitude)


def fbp(projections,angles):
    """Parallel-beam Hann-windowed ramp filtered backprojection, slice by slice.

    Projections [angle, y, detector-x]; output [y, z, x]. Detector coordinate is
    x*cos(theta)+z*sin(theta). No source support or input point mask is used.
    """
    _,ny,n=projections.shape;pad=2**int(np.ceil(np.log2(2*n)))
    freq=fft.fftfreq(pad)
    ramp=2*np.abs(freq)*(.5+.5*np.cos(2*np.pi*freq))
    filtered=fft.ifft(fft.fft(projections,n=pad,axis=2)*ramp[None,None,:],axis=2).real[:,:,:n].astype(np.float32)
    axis=np.arange(n,dtype=np.float32)-(n-1)/2;zz,xx=np.meshgrid(axis,axis,indexing='ij')
    output=np.zeros((ny,n,n),np.float32)
    for data,angle in zip(filtered,angles):
        t=np.deg2rad(angle);position=xx*np.cos(t)+zz*np.sin(t)+(n-1)/2
        lo=np.floor(position).astype(int);weight=position-lo
        good=(lo>=0)&(lo+1<n);lo=np.clip(lo,0,n-2)
        output+=(data[:,lo]*(1-weight)+data[:,lo+1]*weight)*good
    output*=np.pi/(2*len(angles))
    return output


def reconstruct(packet_path,calibration_path,output):
    """Independent inverse: recordings + calibration only, no original geometry."""
    out=Path(output);c=json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    g=Grid(c['sensor_n'],c['sensor_pitch_m'],c['wavelength_m']);p=np.load(packet_path)
    n=c['volume_n'];q=c['object_to_sensor_sampling_ratio'];roi=c['object_roi_rc'];sl=np.s_[roi[0]:roi[1],roi[2]:roi[3]]
    sensor_fields=[];phase_projections=[];gain=c['wave_number_per_m']*c['voxel_pitch_m'];wave_errors=[]
    xx,yy=g.xy;R=c['reference_amplitude']*np.exp(2j*np.pi*c['carrier_per_m']*xx)
    counts_all=p['plate_counts'] if c['mode']=='optical' else p['camera_counts']
    peaks=p['recording_intensity_peak'] if c['mode']=='optical' else p['camera_intensity_peak']
    normalizers=p['slm_amplitude_normalizer'] if c['mode']=='cgh' else None
    for j,angle in enumerate(c['angles_deg']):
        if c['mode']=='optical':
            transmission=counts_all[j].astype(np.float32)/65535
            # Reference illumination of the plate, then a physical 4f pupil.
            wave=bandlimit(transmission*R,g,c['optical_pupil_per_m'])[0]*peaks[j]/c['reference_amplitude']**2
        else:
            wave=four_step_recover(counts_all[j],peaks[j],c['reference_amplitude'])
            if normalizers is not None:wave*=normalizers[j]
        object_wave=propagate(wave,g,-c['recording_distance_m'],method='angular',backend='scipy')
        phase=np.angle(object_wave[sl]).astype(np.float32)
        # Point density was intentionally scaled so max physical phase < pi;
        # hence atan2 is unambiguous and no true phase is used to unwrap it.
        phase_low=phase.reshape(n,q,n,q).mean(axis=(1,3))
        phase_projections.append(phase_low)
        if j==0:sensor_fields.append(wave.astype(np.complex64))
    phase_projections=np.asarray(phase_projections)
    density=fbp(phase_projections/gain,np.asarray(c['angles_deg']))
    np.savez_compressed(out/'reconstructed.npz',delta_n_volume=density,phase_projections_rad=phase_projections,
      angles_deg=np.asarray(c['angles_deg']),first_recovered_sensor_field=sensor_fields[0],voxel_pitch_m=c['voxel_pitch_m'])
    p.close()
    return density,phase_projections


def run_case(name,angle_count=90):
    label,mode=CASES[name];out=ROOT/f'outputs/pointcloud_holography/{name}';out.mkdir(parents=True,exist_ok=True)
    volume,points,model=source_volume(name);n=volume.shape[0]
    g=Grid(512,2e-6,632.8e-9);voxel_pitch=4e-6;q=2;patch=n*q;r0=(g.n-patch)//2;roi=[r0,r0+patch,r0,r0+patch]
    angles=np.linspace(0,180,angle_count,endpoint=False);projections=np.asarray([project(volume,a) for a in angles])
    k=2*np.pi/g.wavelength;density_factor=1.8/(k*voxel_pitch*float(projections.max()))
    density=(volume*density_factor).astype(np.float32);phases=projections*(k*voxel_pitch*density_factor)
    config={'object':name,'label':label,'mode':mode,'sensor_n':g.n,'sensor_pitch_m':g.dx,'wavelength_m':g.wavelength,
      'recording_distance_m':.003,'reference_amplitude':1.5,'carrier_per_m':185/g.size,'optical_pupil_per_m':55000.,
      'volume_n':n,'voxel_pitch_m':voxel_pitch,'object_to_sensor_sampling_ratio':q,'object_roi_rc':roi,'angles_deg':angles.tolist(),
      'wave_number_per_m':k,'phase_max_rad':float(phases.max()),'camera_bits':16,
      'volume_model':'smooth weak transparent refractive-index surface density derived from point cloud',
      'projection_model':'straight-ray phase projection, no internal diffraction or multiple scattering',
      'propagation':'scalar angular-spectrum propagation from effective object plane to sensor',
      'inverse':'hologram/SLM-camera decoding, angular-spectrum backpropagation, atan2 phase, Hann-ramp FBP',
      'noise':'ideal records with 16-bit quantization; no shot/read/phase-step noise',
      'full_rotation_requires_multiple_recordings':True,'single_recording_does_not_provide_complete_360_geometry':True,
      'model_preparation':model,'source':json.loads((ROOT/f'data/pointcloud_holography/{name}-source.json').read_text(encoding='utf-8'))}
    (out/'calibration.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'input-reference.npz',delta_n_volume=density,normalized_xyz=points)
    records=[];peaks=[];slm_amp=[];slm_phase=[];normalizers=[];plate=[];plate_peak=[]
    xx,yy=g.xy;R=config['reference_amplitude']*np.exp(2j*np.pi*config['carrier_per_m']*xx)
    for j,angle in enumerate(angles):
        phase=np.zeros((g.n,g.n),np.float32);phase[r0:r0+patch,r0:r0+patch]=np.repeat(np.repeat(phases[j],q,axis=0),q,axis=1)
        wave=propagate(np.exp(1j*phase),g,config['recording_distance_m'],method='angular',backend='scipy')
        if mode=='optical':
            # Physical pupil makes the off-axis object bandwidth explicit.
            wave=bandlimit(wave,g,config['optical_pupil_per_m'])[0]
            I=np.abs(wave+R)**2;peak=float(I.max());plate.append(np.rint(I/peak*65535).astype(np.uint16));plate_peak.append(peak)
        elif mode=='cgh':
            # Ideal tandem amplitude/phase modulation, followed by an independent
            # four-step camera measurement of its optically replayed complex field.
            scale=float(np.abs(wave).max());a=np.rint(np.abs(wave)/scale*255).astype(np.uint8)
            d=(np.rint(np.mod(np.angle(wave),2*np.pi)*256/(2*np.pi)).astype(np.int16)%256).astype(np.uint8)
            played=(a.astype(np.float32)/255)*np.exp(1j*(d.astype(np.float32)*2*np.pi/256))
            counts,peak=four_step_record(played,config['reference_amplitude'])
            records.append(counts);peaks.append(peak);slm_amp.append(a);slm_phase.append(d);normalizers.append(scale)
        else:
            counts,peak=four_step_record(wave,config['reference_amplitude']);records.append(counts);peaks.append(peak)
        if j%30==0:print(f'{name}: recorded {j+1}/{angle_count}',flush=True)
    packet={'angles_deg':angles}
    if mode=='optical':packet.update(plate_counts=np.asarray(plate),recording_intensity_peak=np.asarray(plate_peak))
    else:packet.update(camera_counts=np.asarray(records),camera_intensity_peak=np.asarray(peaks))
    if mode=='cgh':packet.update(slm_amplitude_commands=np.asarray(slm_amp),slm_phase_commands=np.asarray(slm_phase),slm_amplitude_normalizer=np.asarray(normalizers))
    np.savez_compressed(out/'recordings.npz',**packet)
    del records,slm_amp,slm_phase,plate,packet
    reconstructed,recovered_phases=reconstruct(out/'recordings.npz',out/'calibration.json',out)
    positive=np.maximum(reconstructed,0)
    metrics={'phase_projection_relative_error':relative_error(recovered_phases,phases),
      'raw_volume_relative_error':relative_error(reconstructed,density),
      'nonnegative_volume_relative_error':relative_error(positive,density),
      'volume_correlation':float(np.corrcoef(reconstructed.ravel(),density.ravel())[0,1]),
      'reference_peak_delta_n':float(density.max()),'reconstructed_peak_delta_n':float(reconstructed.max()),
      'reconstructed_negative_voxel_fraction':float(np.mean(reconstructed<0)),
      'metric_scope':'full 96^3 grid; physical scaling from wavelength/voxel pitch; no fitted gain, source support, alignment or ground truth in inverse',
      'display':'interactive viewer thresholds positive reconstructed density; raw signed float volume is retained',
      'geometry':'point-cloud derived weak phase surface; no claim of measured material refractive index or hidden-surface completion'}
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    make_figures(out,config,density,reconstructed,points,phases,recovered_phases)
    print(name,json.dumps(metrics),flush=True)
    return config,metrics


def make_figures(out,c,truth,rec,points,phases,recovered):
    names={'cat':'猫','horse':'马','wolf':'狼'};name=names[c['object']]
    fig=plt.figure(figsize=(12,4.5),layout='constrained')
    ax=fig.add_subplot(131,projection='3d');ax.scatter(points[:,0],points[:,2],points[:,1],s=2,c='#356ea6');ax.set_title(f'公开{name}点云（3400点）');ax.set_box_aspect((1,1,1));ax.set(xlabel='x',ylabel='深度',zlabel='竖直',xlim=(-.5,.5),ylim=(-.5,.5),zlim=(-.5,.5));ax.view_init(elev=15,azim=-70)
    for j,(title,v) in enumerate([('输入点云栅格体',truth),('由多角度全息数据重建',np.maximum(rec,0))]):
        ax=fig.add_subplot(1,3,j+2,projection='3d');mask=v>.03*truth.max();r,z,x=np.where(mask);rng=np.random.default_rng(8)
        if len(x)>12000:idx=rng.choice(len(x),12000,replace=False);r,z,x=r[idx],z[idx],x[idx]
        ax.scatter(x,z,r,c='#356ea6',s=1.4,alpha=.75);ax.set_title(title+'（3%阈值）');ax.set_box_aspect((1,1,1));ax.set_xlim(0,len(v));ax.set_ylim(0,len(v));ax.set_zlim(0,len(v));ax.view_init(elev=15,azim=-70)
    fig.savefig(out/'volume-comparison.png',dpi=160,bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,7.8),layout='constrained')
    for j,idx in enumerate([0,len(phases)//3,2*len(phases)//3]):
        for row,data in enumerate([phases,recovered]):
            v=axs[row,j].imshow(data[idx],cmap='gray',vmin=0,vmax=c['phase_max_rad'],interpolation='nearest',origin='lower');axs[row,j].set_title(f"{c['angles_deg'][idx]:.0f}° {'目标相位投影' if row==0 else '全息图恢复的相位'}");axs[row,j].axis('off')
    fig.colorbar(v,ax=axs.ravel().tolist(),shrink=.7,label='相位 / rad');fig.savefig(out/'phase-projections.png',dpi=160,bbox_inches='tight');plt.close(fig)
    center=len(truth)//2;fig,axs=plt.subplots(2,3,figsize=(11,7),layout='constrained')
    for row,v in enumerate([truth,rec]):
        for j,image in enumerate([v[center],v[:,center,:],v[:,:,center]]):
            axs[row,j].imshow(image,cmap='gray',vmin=0,vmax=truth.max());axs[row,j].axis('off');axs[row,j].set_title(('输入参考' if row==0 else '数值重建')+' '+['水平切片','正视切片','侧视切片'][j])
    fig.savefig(out/'volume-slices.png',dpi=160,bbox_inches='tight');plt.close(fig)
    # Small, physically reconstructed previews for the browser angle slider.
    p=np.load(out/'recordings.npz');folder=out/'previews';folder.mkdir(exist_ok=True)
    counts_all=p['plate_counts'] if c['mode']=='optical' else p['camera_counts'][:,0]
    for j in range(len(phases)):
        I=counts_all[j]
        Image.fromarray(np.flipud(np.rint(I.astype(float)/65535*255).astype(np.uint8))).resize((256,256),Image.Resampling.BOX).save(folder/f'hologram-{j:03d}.png')
        phase=np.rint(np.clip(recovered[j]/c['phase_max_rad'],0,1)*255).astype(np.uint8)
        Image.fromarray(np.flipud(phase)).resize((288,288),Image.Resampling.NEAREST).save(folder/f'phase-{j:03d}.png')
    p.close()


def run(angles=90):
    start=time.perf_counter();summaries=[]
    for name in CASES:
        config,metrics=run_case(name,angles);summaries.append({'case':name,'label':config['label'],'angles':angles,'metrics':metrics})
    out=ROOT/'outputs/pointcloud_holography';(out/'summary.json').write_text(json.dumps({'cases':summaries,'elapsed_seconds':time.perf_counter()-start},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Finished all point-cloud holography cases',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--angles',type=int,default=90);args=p.parse_args();run(args.angles)
