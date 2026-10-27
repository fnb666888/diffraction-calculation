"""Image-based simulation of thin optical hologram recording and optical replay.

Replay multiplies the recorded amplitude transmittance by illumination, then
models a physical 4f order-selection pupil and propagation. No camera demodulator
or truth data enters replay. Back propagation displays the virtual-image field;
conjugate illumination forms the real image by forward propagation.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from matplotlib.patches import Circle
from .core import Grid, propagate, fft2c
from .holography import bandlimit, record, carrier_conditions
from .metrics import relative_error
from .phase_comparison_figures import image

ROOT=Path(__file__).resolve().parents[1]


def optical_replay(transmittance_path,calibration_path):
    """Replay interface: analog transmittance and optical geometry only.

    Results use a common upright coordinate frame: the fixed 180-degree image
    rotation of a physical 4f relay is compensated in the display coordinates.
    This fixed convention does not use target-dependent registration.
    """
    c=json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    with np.load(transmittance_path) as f:t=f['amplitude_transmittance']
    g=Grid(c['n'],c['dx_m'],c['wavelength_m']);x,_=g.xy
    r=c['reference_amplitude']*np.exp(2j*np.pi*c['carrier_per_m']*x)
    # A physical lens/pupil/lens arrangement selects the on-axis desired order
    # after reference illumination. FFTs implement the optical propagation.
    virtual_exit=t*r;real_exit=t*np.conj(r)
    virtual_order,pupil=bandlimit(virtual_exit,g,c['optical_pupil_bandwidth_per_m'])
    real_order,_=bandlimit(real_exit,g,c['optical_pupil_bandwidth_per_m'])
    virtual=propagate(virtual_order,g,-c['distance_m'],method='fresnel',backend='scipy')
    real=propagate(real_order,g,c['distance_m'],method='fresnel',backend='scipy')
    return {'virtual_exit':virtual_exit,'real_exit':real_exit,'virtual_order':virtual_order,
            'real_order':real_order,'virtual':virtual,'real':real,'pupil':pupil},c


def run(source=None,output=None):
    source=Path(source or ROOT/'data/optical_holography/camera/source.png')
    out=Path(output or ROOT/'outputs/optical_holography/camera');out.mkdir(parents=True,exist_ok=True)
    g=Grid(2048,1.25e-6,632.8e-9);z=.002;bandwidth=44000.;ar=1.5;fc=370/g.size
    width=1024;photo=np.asarray(Image.open(source).convert('L').resize((width,width),Image.Resampling.LANCZOS),float)/255
    r0=(g.n-width)//2;roi=np.array([r0-12,r0+width+12,r0-12,r0+width+12]);sl=np.s_[roi[0]:roi[1],roi[2]:roi[3]]
    axis=(np.arange(width)+.5)/width
    phase=1.2*np.sin(2*np.pi*4*axis[:,None])*np.sin(2*np.pi*4*axis[None,:])
    raw=np.zeros((g.n,g.n),complex);raw[r0:r0+width,r0:r0+width]=np.sqrt(photo)*np.exp(1j*phase)
    target,pupil=bandlimit(raw,g,bandwidth)
    object_wave=propagate(target,g,z,method='fresnel',backend='scipy')
    x,_=g.xy;reference=ar*np.exp(2j*np.pi*fc*x)
    hologram=record(object_wave,reference);imax=float(hologram.max());t=hologram/imax
    sampling=carrier_conditions(g,fc,bandwidth)
    if not sampling['separated'] or 2*fc+bandwidth>=1/(2*g.dx):raise ValueError('Recording/replay orders alias.')
    config={'n':g.n,'dx_m':g.dx,'wavelength_m':g.wavelength,'distance_m':z,
      'carrier_per_m':fc,'reference_amplitude':ar,'optical_pupil_bandwidth_per_m':bandwidth,
      'hologram_intensity_normalizer':imax,'hologram_amplitude_law':'t_H=I_H/max(I_H), real 0..1',
      'phase_amplitude_rad':1.2,'phase_cycles_per_image':4,'object_width_m':width*g.dx,
      'propagation':'periodic discrete Fresnel transfer function; common axial phase removed',
      'replay':'reference or conjugate reference illumination; physical 4f circular pupil selects desired order',
      'comparison_coordinates':'fixed 180-degree 4f relay inversion compensated by the upright coordinate convention; no truth-dependent registration'}
    np.savez_compressed(out/'hologram-transmittance.npz',amplitude_transmittance=t)
    (out/'calibration.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    replay,_=optical_replay(out/'hologram-transmittance.npz',out/'calibration.json')
    gain=ar*ar/imax;virtual=replay['virtual']/gain;real=replay['real']/gain
    unfiltered=propagate(replay['virtual_exit'],g,-z,method='fresnel',backend='scipy')/gain
    metrics={'configuration':config,'sampling':sampling,
      'source':json.loads(source.with_name('source.json').read_text(encoding='utf-8')),
      'replay_max_frequency_per_m':2*fc+bandwidth,'replay_nyquist_per_m':1/(2*g.dx),
      'pupil_retained_input_power':float(np.sum(np.abs(target)**2)/np.sum(np.abs(raw)**2)),
      'known_common_replay_amplitude_gain':gain,
      'virtual_complex_relative_error':relative_error(virtual,target),
      'real_conjugate_complex_relative_error':relative_error(real,np.conj(target)),
      'virtual_phase_rmse_photo_roi_rad':float(np.sqrt(np.mean(np.angle(virtual[sl]*np.conj(target[sl]))**2))),
      'real_conjugate_phase_rmse_photo_roi_rad':float(np.sqrt(np.mean(np.angle(real[sl]*target[sl])**2))),
      'virtual_intensity_relative_error_photo_roi':relative_error(np.abs(virtual[sl])**2,np.abs(target[sl])**2),
      'real_intensity_relative_error_photo_roi':relative_error(np.abs(real[sl])**2,np.abs(target[sl])**2),
      'limitations':['computer simulation; no experimental photograph of a real hologram',
        'photograph specifies a 2D coherent object intensity, assigned optical phase is not measured',
        'ideal linear thin amplitude hologram and ideal optical 4f order-selection pupil',
        'noise, material nonlinearity, finite pixel-area response and lens aberration are omitted',
        'virtual image field is back-propagated for inspection; it is not a forward real-image screen',
        'Fresnel/periodic-grid idealization; all displayed phase pixels retained without masking']}
    np.savez_compressed(out/'fields.npz',x_m=g.x,photo_roi_rc=roi,raw_object=raw.astype(np.complex64),
      effective_input=target.astype(np.complex64),recording_object_wave=object_wave.astype(np.complex64),
      hologram_intensity=hologram.astype(np.float32),virtual=replay['virtual'].astype(np.complex64),
      real=replay['real'].astype(np.complex64),unfiltered_virtual=unfiltered.astype(np.complex64),
      replay_real_exit=replay['real_exit'].astype(np.complex64),selected_order=replay['real_order'].astype(np.complex64),
      optical_pupil=pupil)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    def save(fig,key):fig.savefig(out/(key+'.png'),dpi=180,bbox_inches='tight',pad_inches=.12,facecolor='white');plt.close(fig)
    iref=max(float(np.abs(u[sl]).max()**2) for u in [target,virtual,real])
    fig,axs=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
    axs[0].imshow(np.asarray(Image.open(source)),cmap='gray',vmin=0,vmax=255);axs[0].axis('off');axs[0].set_title('Camera 摄影师原图')
    image(axs[1],np.abs(target)**2,g.x,roi,'有效输入光强',iref)
    image(axs[2],np.angle(target),g.x,roi,'有效输入相位',phase=True);save(fig,'input')
    record_roi=np.array([r0-100,r0+width+100,r0-100,r0+width+100])
    fig,axs=plt.subplots(1,2,figsize=(12.8,5.2),layout='constrained')
    image(axs[0],np.abs(object_wave)**2,g.x,record_roi,'到达记录面的物光光强',float(np.abs(object_wave).max()**2))
    image(axs[1],hologram,g.x,record_roi,'全息干涉光强',imax);save(fig,'recording')
    fig,axs=plt.subplots(1,2,figsize=(12.8,5.2),layout='constrained')
    r1,r2,c1,c2=record_roi;ext=[g.x[c1]*1e6,g.x[c2-1]*1e6,g.x[r2-1]*1e6,g.x[r1]*1e6]
    v=axs[0].imshow(t[r1:r2,c1:c2],extent=ext,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
    axs[0].set(title='全息片振幅透过率',xlabel='x / μm',ylabel='y / μm');plt.colorbar(v,ax=axs[0],fraction=.04,pad=.025,label='振幅透过率（线性）')
    spec=np.abs(fft2c(replay['real_exit']))**2;spec/=spec.max();f=g.frequency/1000;df=(f[1]-f[0])/2
    v=axs[1].imshow(spec,extent=[f[0]-df,f[-1]+df,f[-1]+df,f[0]-df],cmap='gray',norm=LogNorm(vmin=1e-9,vmax=1),interpolation='nearest')
    axs[1].add_patch(Circle((0,0),bandwidth/1000,fill=False,edgecolor='#D23B31',lw=1.8));axs[1].set(xlim=(-375,100),ylim=(160,-160),title='共轭再现光照明后的频谱',xlabel=r'$f_x$ / mm$^{-1}$',ylabel=r'$f_y$ / mm$^{-1}$')
    plt.colorbar(v,ax=axs[1],fraction=.04,pad=.025,label='归一化频谱功率（对数）');save(fig,'plate-and-replay')
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for j,(u,heading) in enumerate([(target,'有效输入'),(virtual,'原参考光再现：虚像'),(real,'共轭参考光再现：实像')]):
        image(axs[0,j],np.abs(u)**2,g.x,roi,heading+'光强',iref)
        image(axs[1,j],np.angle(u),g.x,roi,heading+'相位',phase=True)
    save(fig,'reconstruction')
    # A complete result view also retains the unselected field, without claiming
    # that the clean image appears before the optical spatial filter.
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    total_ref=max(float(np.abs(unfiltered[sl]).max()**2),iref)
    image(axs[0],np.abs(unfiltered)**2,g.x,roi,'未作光学选级',total_ref)
    image(axs[1],np.abs(virtual)**2,g.x,roi,'光学选级后的虚像',total_ref);save(fig,'order-selection')
    text=f'''# 光学离轴全息：Camera完整流程模拟

## 输入
公开Camera照片，作者Lav Varshney，CC0。[数据说明]({metrics['source']['page']})。灰度指定光强，振幅为其平方根。为观察共轭像的相位符号，设置幅值1.2 rad、横纵各4周期的二维正弦初相位。相位是数值建模设置。光瞳带宽44/mm，输入经过相同光瞳带限后作为对照。

## 记录
2048²网格、1.25 μm间距、632.8 nm波长，照片物理宽1.28 mm，传播2 mm到记录面。参考光振幅1.5、水平载频{fc/1000:.6f}/mm。干涉光强为物光加参考光的模平方。采用理想薄振幅全息片模型 t=I/max(I)，真实振幅透过率在0到1之间。未模拟材料显影过程。

## 光学再现
重放函数只读取保存的全息片振幅透过率与标定。分别使用原参考光和共轭参考光照明，出射场为t乘再现光。用理想4f光路的圆形频谱光阑选出目标衍射级，再传播。这里FFT用于模拟物理透镜与传播，不执行数字全息的相机解调算法。

- 原参考光选级后物光项与原物光成正比，回溯2 mm显示虚像物面复场。虚像不是在前方屏幕上形成的实像。
- 共轭参考光选级后得到共轭物光，正向传播2 mm形成共轭实像。4f系统的固定180°倒像按转正坐标展示，两者光强相同，相位符号相反。这一固定坐标约定不使用真值配准。
- 若未作选级，零级和其他分量仍然存在，`order-selection.png`保留这一对照。
- 仅用已知参考强度与记录归一化系数校正公共增益，没有用真值拟合相位或缩放。

## 结果与范围
虚像复场相对误差{metrics['virtual_complex_relative_error']:.3g}，实像相对共轭目标误差{metrics['real_conjugate_complex_relative_error']:.3g}，均为无噪声理想模型的数值一致性，不代表实验精度。再现最高空间频率{(2*fc+bandwidth)/1000:.3f}/mm，小于Nyquist400/mm。光强显示统一γ=0.30，透过率线性，相位统一−π到π且不屏蔽暗区。

所有图像均为本项目计算机模拟。采用二维相干薄物体、理想线性薄振幅全息片、理想物光选级和菲涅耳传播。未加入相机、材料噪声、非线性、像差或三维遮挡。虚像/实像的物面光强共用同一显示标度。

## 文件与复现
`hologram-transmittance.npz`与`calibration.json`是光学再现的全部输入。`fields.npz`以complex64保存各阶段复场供查看，`metrics.json`的误差基于内存complex128计算。运行 `python -m optics_sim.optical_holography_demo`。
'''
    (out/'实验说明.md').write_text(text,encoding='utf-8')
    body=''.join(f'<h2>{label}</h2><img src="{key}.png">' for key,label in [('input','输入'),('recording','物光传播与干涉记录'),('plate-and-replay','全息片与再现光照明'),('reconstruction','虚像与共轭实像'),('order-selection','选级前后')])
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>光学离轴全息完整流程</title><style>body{font:18px/1.7 Microsoft YaHei,sans-serif;max-width:1200px;margin:36px auto;color:#24334b}img{width:100%}h1,h2{color:#174994}</style><h1>光学离轴全息完整流程模拟</h1><p>Camera摄影师照片作为模拟物体。记录为理想薄振幅全息片，再现为参考光照明、光学4f选级及传播。全部环节均为计算机模拟。</p>'+body+'<p><a href="实验说明.md">算法、参数及适用范围</a></p>',encoding='utf-8')
    print(json.dumps({k:metrics[k] for k in ['virtual_complex_relative_error','real_conjugate_complex_relative_error','virtual_phase_rmse_photo_roi_rad','pupil_retained_input_power']},indent=2),flush=True)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();run(a.source,a.output)
