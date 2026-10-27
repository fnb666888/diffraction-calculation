"""Show the direct holographic pipeline and I/A/phase at several depths.

No geometry inverse, fitted surface, FISTA, or source-point substitution is
used. Replayed fields come from saved holograms/SLM commands and calibration.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,LogNorm
from matplotlib.patches import Circle
from .core import Grid,fft2c
from .pointcloud_direct import replay

ROOT=Path(__file__).resolve().parents[1]

def run():
    plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':12})
    for name in ['cat','horse','wolf']:
        parent=ROOT/f'outputs/pointcloud_direct/{name}'
        folder=parent/'single-offaxis' if name=='wolf' and (parent/'single-offaxis/calibration.json').exists() else parent
        out=folder/'full-process';out.mkdir(exist_ok=True)
        sensor,fields,c=replay(folder)
        g=Grid(c['n'],c['dx_m'],c['wavelength_m'])
        np.savez_compressed(out/'replay-fields.npz',recovered_or_replayed_sensor_field=sensor.astype(np.complex64),
            focus_fields=fields.astype(np.complex64),depths_m=np.array(c['focus_distances_m']))
        # Forward-stage field shown only to explain which object wave is recorded.
        with np.load(parent/'input-reference.npz') as packet:forward=packet['reference_sensor_field']
        ext=[g.x[0]*1000,g.x[-1]*1000,g.x[0]*1000,g.x[-1]*1000]
        fig,axs=plt.subplots(1,2,figsize=(9.4,4.4),layout='constrained')
        a=np.abs(forward);peak=float(a.max())
        for ax,v,title,phase in [(axs[0],a/peak,'到达全息面的物光振幅',False),(axs[1],np.angle(forward),'到达全息面的物光相位',True)]:
            im=ax.imshow(v,origin='lower',extent=ext,cmap='RdBu_r' if phase else 'gray',
                norm=None if phase else PowerNorm(.5,vmin=0,vmax=1),vmin=-np.pi if phase else None,vmax=np.pi if phase else None,interpolation='nearest')
            ax.set(title=title,xlabel='x / mm',ylabel='y / mm')
            fig.colorbar(im,ax=ax,fraction=.04,pad=.025,label='相位 / rad' if phase else '相对振幅')
        fig.savefig(out/'recording-object-field.png',dpi=180,facecolor='white');plt.close(fig)
        if name=='cat':
            with np.load(folder/'recordings.npz') as packet:t=packet['amplitude_transmittance']
            x,_=g.xy;R=c['reference_amplitude']*np.exp(2j*np.pi*c['carrier_per_m']*x)
            spectrum=np.abs(fft2c(t*np.conj(R)))**2;spectrum/=spectrum.max()
            f=g.frequency/1000;freq_ext=[f[0],f[-1],f[0],f[-1]]
            fig,axs=plt.subplots(1,2,figsize=(11.9,4.7),layout='constrained')
            im=axs[0].imshow(t,cmap='gray',vmin=0,vmax=1,origin='lower',extent=ext,interpolation='nearest')
            axs[0].set(title='实际全息片图样：t = I_H / 峰值',xlabel='x / mm',ylabel='y / mm');fig.colorbar(im,ax=axs[0],fraction=.04,pad=.025,label='振幅透过率（线性）')
            im=axs[1].imshow(spectrum,cmap='gray',norm=LogNorm(vmin=1e-8,vmax=1),origin='lower',extent=freq_ext,interpolation='nearest')
            axs[1].add_patch(Circle((0,0),c['bandwidth_per_m']/1000,fill=False,edgecolor='#C4343F',lw=1.5))
            axs[1].set(title='共轭再现光照明后的频谱',xlabel='空间频率 fx / (1/mm)',ylabel='空间频率 fy / (1/mm)')
            fig.colorbar(im,ax=axs[1],fraction=.04,pad=.025,label='归一化频谱功率（对数）')
            fig.savefig(out/'actual-hologram-and-replay.png',dpi=180,facecolor='white');plt.close(fig)
        elif name=='horse':
            with np.load(folder/'recordings.npz') as packet:amp=packet['slm_amplitude_8bit'].astype(float)/255;phase=packet['slm_phase_8bit'].astype(float)*2*np.pi/256
            fig,axs=plt.subplots(1,2,figsize=(9.4,4.4),layout='constrained')
            for ax,v,title,isphase in [(axs[0],amp,'实际振幅调制图（8位）',False),(axs[1],phase,'实际相位调制图（256级）',True)]:
                im=ax.imshow(v,cmap='gray',vmin=0,vmax=2*np.pi if isphase else 1,origin='lower',extent=ext,interpolation='nearest')
                ax.set(title=title,xlabel='x / mm',ylabel='y / mm');fig.colorbar(im,ax=ax,fraction=.04,pad=.025,label='相位 / rad' if isphase else '振幅透过率')
            fig.savefig(out/'actual-cgh-commands.png',dpi=180,facecolor='white');plt.close(fig)
        elif c.get('digital_recording') == 'single_offaxis':
            with np.load(folder/'recordings.npz') as packet:counts=packet['camera_counts']
            spectrum=np.abs(fft2c(counts.astype(float)))**2;spectrum/=spectrum.max()
            f=g.frequency/1000;freq_ext=[f[0],f[-1],f[0],f[-1]]
            fig,axs=plt.subplots(1,2,figsize=(11.9,4.7),layout='constrained')
            im=axs[0].imshow(counts,cmap='gray',vmin=0,vmax=65535,origin='lower',extent=ext,interpolation='nearest')
            axs[0].set(title='实际单幅离轴全息图',xlabel='x / mm',ylabel='y / mm')
            fig.colorbar(im,ax=axs[0],fraction=.04,pad=.025,label='16位相机计数 / ADU（线性）')
            im=axs[1].imshow(spectrum,cmap='gray',norm=LogNorm(vmin=1e-12,vmax=1),origin='lower',extent=freq_ext,interpolation='nearest')
            axs[1].add_patch(Circle((-c['carrier_per_m']/1000,0),c['bandwidth_per_m']/1000,fill=False,edgecolor='#C4343F',lw=1.5))
            axs[1].set(title='二维频谱：红圈选择物光一级',xlabel='空间频率 fx / (1/mm)',ylabel='空间频率 fy / (1/mm)')
            fig.colorbar(im,ax=axs[1],fraction=.04,pad=.025,label='归一化频谱功率（对数）')
            fig.savefig(out/'actual-single-hologram-and-spectrum.png',dpi=180,facecolor='white');plt.close(fig)
        else:
            with np.load(folder/'recordings.npz') as packet:counts=packet['camera_counts']
            fig,axs=plt.subplots(1,4,figsize=(15,4.05),layout='constrained')
            for j,ax in enumerate(axs):
                im=ax.imshow(counts[j],cmap='gray',vmin=0,vmax=65535,origin='lower',extent=ext,interpolation='nearest')
                ax.set(title='参考相位 '+['0','π/2','π','3π/2'][j],xlabel='x / mm',ylabel='y / mm')
            fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.018,pad=.02,label='16位相机计数 / ADU')
            fig.savefig(out/'actual-camera-holograms.png',dpi=180,facecolor='white');plt.close(fig)
        roi=np.s_[g.n//2-640:g.n//2+640,g.n//2-640:g.n//2+640]
        field=fields[:,roi[0],roi[1]];I=np.abs(field)**2;ip=float(I.max());ap=np.sqrt(ip)
        for quantity,title in [('intensity','光强'),('amplitude','振幅'),('phase','相位')]:
            fig,axs=plt.subplots(1,3,figsize=(13,4.55),layout='constrained')
            for j,ax in enumerate(axs):
                if quantity=='intensity':v=I[j]/ip;norm=PowerNorm(.35,vmin=0,vmax=1);cmap='gray';label='相对光强'
                elif quantity=='amplitude':v=np.abs(field[j])/ap;norm=PowerNorm(.5,vmin=0,vmax=1);cmap='gray';label='相对振幅'
                else:v=np.angle(field[j]);norm=plt.Normalize(-np.pi,np.pi);cmap='RdBu_r';label='相位 / rad'
                im=ax.imshow(v,cmap=cmap,norm=norm,origin='lower',extent=[-.32,.32,-.32,.32],interpolation='nearest')
                ax.set(title=f"z = {c['focus_distances_m'][j]*1000:.3f} mm",xlabel='x / mm',ylabel='y / mm')
            fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.018,pad=.02,label=label)
            fig.savefig(out/f'depth-{quantity}.png',dpi=180,facecolor='white');plt.close(fig)
        (out/'display.json').write_text(json.dumps({'intensity_reference':ip,'amplitude_reference':float(ap),
            'intensity_gamma':.35,'amplitude_gamma':.5,'phase_limits_rad':[-float(np.pi),float(np.pi)],
            'phase_mask':False,'shared_scale_across_depths':True,
            'replay_input':'recordings.npz and calibration.json only','geometry_inverse':False},indent=2),encoding='utf-8')
        print('Prepared full holographic flow '+name,flush=True)

if __name__=='__main__':run()
