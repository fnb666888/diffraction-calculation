"""Single-view measured RGB-D point cloud holography and local WRP experiment.

All generation methods use the same rasterized lateral coordinates, original
depth levels, coherent source weights and periodic discrete Fresnel operator.
The input is an observed surface, not an unseen complete 3-D solid. Intensity
and optical phase are model choices, not quantities measured by the Kinect.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import platform
import time
import numpy as np
from scipy import fft
from .core import Grid,fft2c
from .holography import record,reconstruct_off_axis,carrier_conditions
from .metrics import relative_error

ROOT=Path(__file__).resolve().parents[1]


def read_ascii_pcd(path):
    text=Path(path).read_text(encoding='ascii')
    header,body=text.split('DATA ascii\n',1)
    fields=next(line.split()[1:] for line in header.splitlines() if line.startswith('FIELDS '))
    a=np.loadtxt(io.StringIO(body))
    col={name:a[:,i] for i,name in enumerate(fields)}
    xyz=np.column_stack([col[k] for k in ['x','y','z']])
    bits=np.ascontiguousarray(col['rgb'].astype(np.float32)).view(np.uint32)
    colors=np.column_stack([(bits>>16)&255,(bits>>8)&255,bits&255]).astype(np.uint8)
    image_xy=np.column_stack([col['imX'],col['imY']]).astype(int)
    valid=np.isfinite(xyz).all(axis=1)
    return xyz[valid],colors[valid],image_xy[valid]


def prepare_model(path,n=1024,dx=1.5e-6,height=.0008,central_distance=.006):
    xyz,rgb,image_xy=read_ascii_pcd(path)
    # Dataset axes: x right, y into image (depth), z upward.
    scale=height/np.ptp(xyz[:,2])
    x=(xyz[:,0]-(xyz[:,0].max()+xyz[:,0].min())/2)*scale
    y=-(xyz[:,2]-(xyz[:,2].max()+xyz[:,2].min())/2)*scale
    distance=central_distance+(xyz[:,1]-np.median(xyz[:,1]))*scale
    rows=np.rint(y/dx).astype(int)+n//2;cols=np.rint(x/dx).astype(int)+n//2
    if min(rows.min(),cols.min())<0 or max(rows.max(),cols.max())>=n or distance.min()<=0:
        raise ValueError('Object does not fit the grid or lies beyond the hologram.')
    amplitude=rgb.astype(float)@np.array([.2126,.7152,.0722])/255
    levels,inverse=np.unique(distance,return_inverse=True)
    groups=[np.flatnonzero(inverse==i) for i in range(len(levels))]
    return {'raw_xyz_m':xyz,'rgb':rgb,'image_xy':image_xy,'scale':scale,
            'object_xy_m':np.column_stack([x,y]),'distance_m':distance,
            'rows':rows,'cols':cols,'amplitude':amplitude,'levels':levels,'groups':groups,
            'max_lateral_rounding_m':float(np.abs(np.column_stack([(cols-n//2)*dx-x,(rows-n//2)*dx-y])).max())}


class PointCloudGenerator:
    def __init__(self,model,grid,bandwidth=75000.,wrp_margin=50e-6,factors=(1.,1.5,2.)):
        self.m=model;self.g=grid;self.bandwidth=bandwidth;self.factors=factors
        f=fft.fftfreq(grid.n,grid.dx);q=f*f
        self.pupil=q[:,None]+q[None,:]<=bandwidth**2
        self.wrp_distance=float(model['distance_m'].min()-wrp_margin)
        if self.wrp_distance<=0:raise ValueError('WRP must lie between all object points and the hologram.')
        self.h_full=[];self.h_near=[];self.tf_full=[];self.local={factor:[] for factor in factors}
        n=grid.n
        for distance in model['levels']:
            tf=np.exp(-1j*np.pi*grid.wavelength*distance*q)
            self.tf_full.append(tf);self.h_full.append(fft.ifft(tf,workers=1))
            delta=distance-self.wrp_distance
            hn=fft.ifft(np.exp(-1j*np.pi*grid.wavelength*delta*q),workers=1)
            self.h_near.append(hn)
            for factor in factors:
                radius=min(n//2-1,int(np.ceil(factor*delta*grid.wavelength/(2*grid.dx**2)))+8)
                offsets=np.arange(-radius,radius+1)
                h=hn[offsets%n]
                self.local[factor].append((radius,h[:,None]*h[None,:]))
        self.tf_wrp=np.exp(-1j*np.pi*grid.wavelength*self.wrp_distance*q)

    def finish_spectrum(self,spectrum):
        return fft.ifft2(spectrum*self.pupil,workers=1)

    def layered_fft(self):
        n=self.g.n;total=np.zeros((n,n),complex);layer=np.zeros_like(total)
        for idx,tf in zip(self.m['groups'],self.tf_full):
            layer.fill(0);np.add.at(layer,(self.m['rows'][idx],self.m['cols'][idx]),self.m['amplitude'][idx])
            total+=fft.fft2(layer,workers=1)*tf[:,None]*tf[None,:]
        return self.finish_spectrum(total)

    def full_point_sum(self):
        """Full-domain point sum, using separability and matrix multiplication.

        This is an optimized exact discrete point baseline, not a claimed
        naive exp/point/pixel implementation.
        """
        n=self.g.n;axis=np.arange(n);total=np.zeros((n,n),complex)
        for idx,h in zip(self.m['groups'],self.h_full):
            hy=h[(axis[:,None]-self.m['rows'][idx][None,:])%n]*self.m['amplitude'][idx][None,:]
            hx=h[(axis[None,:]-self.m['cols'][idx][:,None])%n]
            total+=hy@hx
        return self.finish_spectrum(fft.fft2(total,workers=1))

    def local_wrp(self,factor):
        n=self.g.n;u=np.zeros((n,n),complex)
        for idx,(radius,kernel) in zip(self.m['groups'],self.local[factor]):
            for point in idx:
                r=int(self.m['rows'][point]);c=int(self.m['cols'][point]);weight=self.m['amplitude'][point]
                if r-radius>=0 and c-radius>=0 and r+radius<n and c+radius<n:
                    u[r-radius:r+radius+1,c-radius:c+radius+1]+=weight*kernel
                else:
                    rr=np.arange(r-radius,r+radius+1)%n;cc=np.arange(c-radius,c+radius+1)%n
                    u[np.ix_(rr,cc)]+=weight*kernel
        return self.finish_spectrum(fft.fft2(u,workers=1)*self.tf_wrp[:,None]*self.tf_wrp[None,:])


def run(path,output=None,repeats=3):
    out=Path(output or ROOT/'outputs/rgbd_holography/coffee_mug_1').resolve();out.mkdir(parents=True,exist_ok=True)
    g=Grid(1024,1.5e-6,632.8e-9)
    model=prepare_model(path,g.n,g.dx)
    t=time.perf_counter();generator=PointCloudGenerator(model,g);setup=time.perf_counter()-t
    methods={'layered_fft':generator.layered_fft,'full_point_sum':generator.full_point_sum}
    for factor in generator.factors:methods[f'wrp_{factor:g}']=lambda f=factor:generator.local_wrp(f)
    fields={};samples={key:[] for key in methods}
    for name,fn in methods.items():
        t=time.perf_counter();fields[name]=fn()
        print(f'prepared {name}: {time.perf_counter()-t:.3f}s',flush=True)
    names=list(methods)
    for repeat in range(repeats):
        for name in names[repeat%len(names):]+names[:repeat%len(names)]:
            t=time.perf_counter();methods[name]();elapsed=time.perf_counter()-t
            samples[name].append(elapsed*1000)
            print(f'timing {repeat+1}/{repeats} {name}: {elapsed*1000:.1f}ms',flush=True)
    truth=fields['layered_fft'];normalizer=float(np.abs(truth).max())
    metrics={}
    for name,u in fields.items():
        metrics[name]={'median_ms':float(np.median(samples[name])),'samples_ms':samples[name],
                       'sensor_complex_relative_error':relative_error(u,truth),
                       'sensor_intensity_relative_error':relative_error(np.abs(u)**2,np.abs(truth)**2)}
    candidates=[name for name in methods if name.startswith('wrp_')]
    acceptable=[name for name in candidates if metrics[name]['sensor_complex_relative_error']<=.01]
    selected=min(acceptable,key=lambda name:metrics[name]['median_ms']) if acceptable else min(candidates,key=lambda name:metrics[name]['sensor_complex_relative_error'])
    carrier=round(240000*g.size)/g.size;ar=1.5
    x,y=g.xy;reference=ar*np.exp(2j*np.pi*carrier*x)
    arrays={key:model[key] for key in ['raw_xyz_m','rgb','image_xy','object_xy_m','distance_m','rows','cols','amplitude','levels']}
    arrays.update(object_x_m=g.x,pupil=generator.pupil,reference_wave=reference)
    for name,u in fields.items():arrays[name+'_sensor_field']=u/normalizer
    # Record each method's own intensity. Reconstruction uses intensity and known calibration only.
    for name in ['layered_fft',selected]:
        u=fields[name]/normalizer;hologram=record(u,reference)
        recovered=reconstruct_off_axis(hologram,reference,g,generator.bandwidth,ar)
        arrays[name+'_hologram_intensity']=hologram
        arrays[name+'_recovered_sensor_field']=recovered
        metrics[name]['off_axis_complex_relative_error']=relative_error(recovered,u)
    result={'source':str(Path(path).resolve()),'source_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
      'configuration':{'n':g.n,'dx_m':g.dx,'wavelength_m':g.wavelength,'bandwidth_per_m':generator.bandwidth,
        'carrier_per_m':carrier,'reference_amplitude':ar,'object_height_m':.0008,
        'uniform_geometric_scale':model['scale'],'source_point_count':len(model['amplitude']),
        'original_depth_levels_used':len(model['levels']),'no_extra_depth_quantization':True,
        'lateral_rasterization_max_error_m':model['max_lateral_rounding_m'],
        'object_to_hologram_range_m':[float(model['distance_m'].min()),float(model['distance_m'].max())],
        'wrp_to_hologram_m':generator.wrp_distance,'object_to_wrp_range_m':[float(model['distance_m'].min()-generator.wrp_distance),float(model['distance_m'].max()-generator.wrp_distance)],
        'local_window_radius_pixels':{str(f):[min(r for r,k in generator.local[f]),max(r for r,k in generator.local[f])] for f in generator.factors},
        'common_sensor_field_normalizer':normalizer,'fft_workers':1,'precision':'complex128','repeats':repeats,
        'cpu':platform.processor(),'kernel_setup_seconds':setup,
        'phase_convention':'zero initial envelope phase under common forward plane-wave illumination; only a common axial carrier is removed',
        'point_weight':'normalized RGB luminance used as modeled amplitude, not measured optical amplitude/phase',
        'propagation':'same periodic discrete paraxial Fresnel operator for every method; circular pupil at sensor',
        'timing_scope':'cached-kernel forward field generation including common pupil; excludes setup, I/O, plots, reference recording and inverse reconstruction',
        'full_point_baseline':'full-domain discrete point summation accelerated by separability and matrix multiplication, not naive spherical exp for every sample'},
      'carrier_sampling':carrier_conditions(g,carrier,generator.bandwidth),'methods':metrics,'selected_wrp':selected,
      'selection_rule':'fastest WRP with sensor complex relative error <=1%; otherwise lowest error WRP',
      'limitations':['single-view measured surface, no unseen back surface','Kinect colors/depth do not measure coherent optical phase','uniform miniaturization and lateral rasterization are numerical modeling choices','no occlusion transport, multiple scattering or RGB wavelength simulation','benchmark is specific to this model/implementation/hardware']}
    if not result['carrier_sampling']['separated']:raise ValueError('Off-axis orders overlap.')
    np.savez_compressed(out/'data.npz',**arrays)
    (out/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'selected_wrp':selected,'methods':metrics},ensure_ascii=True),flush=True)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--pcd',type=Path,default=ROOT/'data/rgbd/coffee_mug_1/coffee_mug_1_1_100.pcd')
    p.add_argument('--output',type=Path);p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();run(a.pcd,a.output,a.repeats)


if __name__=='__main__':main()
