"""Two-plane CGH synthesis, double-phase SLM encoding and optical replay.

Source RGB/depth images are extracted from the supplied assignment. The 40/60 mm
planes are explicit new simulation settings, not recovered document parameters.
Optical replay reads only the exported phase command and calibration, creates a
unit-amplitude phase-modulated incident beam, models a physical Fourier pupil,
then propagates FORWARD to the two object planes.
"""
from pathlib import Path
import argparse
import json
import hashlib
from io import BytesIO
from zipfile import ZipFile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,LogNorm
from matplotlib.patches import Circle
from .core import Grid,propagate,fft2c
from .holography import bandlimit
from .metrics import relative_error
from .phase_comparison_figures import image

ROOT=Path(__file__).resolve().parents[1]


def replay_slm(command_path,calibration_path):
    """Actual replay input is the 8-bit phase command, not the complex CGH."""
    c=json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    command=np.asarray(Image.open(command_path),dtype=float)
    if command.shape!=(c['slm_n'],c['slm_n']):raise ValueError('SLM command shape mismatch')
    g=Grid(c['slm_n'],c['slm_pitch_m'],c['wavelength_m'])
    phase=2*np.pi*command/c['phase_levels']
    exit_wave=c['incident_amplitude']*np.exp(1j*phase)
    selected,pupil=bandlimit(exit_wave,g,c['fourier_pupil_cutoff_per_m'])
    fields=np.asarray([propagate(selected,g,d,method='fresnel',backend='scipy') for d in c['plane_distances_m']])
    return {'phase_rad':phase,'exit_wave':exit_wave,'selected_wave':selected,'pupil':pupil,'fields':fields},c


def run(output=None):
    out=Path(output or ROOT/'outputs/cgh_slm/two_planes');out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/cgh_slm/two_planes';data.mkdir(parents=True,exist_ok=True)
    # Reuse lossless extracted inputs; a clean run can extract the two original
    # embedded images directly from the supplied document. No build cache needed.
    rgb_file=data/'input-color.png';depth_file=data/'input-depth-levels.png'
    if not (rgb_file.exists() and depth_file.exists() and (data/'source.json').exists()):
        document=Path('D:/桌面/22063101 朱熙莱 全息课设 第三次大作业.docx')
        members=['word/media/image20.tiff','word/media/image21.tiff']
        with ZipFile(document) as package:raw=[package.read(name) for name in members]
        Image.open(BytesIO(raw[0])).convert('RGB').save(rgb_file)
        Image.open(BytesIO(raw[1])).convert('L').save(depth_file)
        source={'document':document.as_posix(),'embedded_images':members,
          'image_sha256':[hashlib.sha256(item).hexdigest() for item in raw],
          'assigned_distances_m':{'cube':.040,'ball':.060},
          'distances_are_new_simulation_settings':True}
        (data/'source.json').write_text(json.dumps(source,ensure_ascii=False,indent=2),encoding='utf-8')
    source=json.loads((data/'source.json').read_text(encoding='utf-8'))
    rgb=np.asarray(Image.open(rgb_file).convert('RGB'));depth=np.asarray(Image.open(depth_file).convert('L'))
    macro=Grid(512,8e-6,632.8e-9);fine=Grid(1024,4e-6,macro.wavelength);cutoff=18000.;distances=np.array([.040,.060])
    gray=rgb.astype(float)@np.array([.2126,.7152,.0722])/255
    values=sorted(int(x) for x in np.unique(depth) if x>0)
    if len(values)!=2:raise ValueError('Expected exactly two non-background depth levels')
    layers=[];r0=(macro.n-depth.shape[0])//2;c0=(macro.n-depth.shape[1])//2
    for level in values:
        a=np.zeros((macro.n,macro.n),complex)
        a[r0:r0+depth.shape[0],c0:c0+depth.shape[1]]=np.sqrt(gray)*(depth==level)
        layers.append(bandlimit(a,macro,cutoff)[0])
    # The darker depth-map shape is the upper cube, the lighter one the ball.
    hologram=sum(propagate(u,macro,-float(d),method='fresnel',backend='scipy') for u,d in zip(layers,distances))
    normalizer=float(np.abs(hologram).max());h=hologram/normalizer
    amplitude=np.clip(np.abs(h),0,1);phase=np.angle(h);delta=np.arccos(amplitude)
    plus=phase+delta;minus=phase-delta
    # Each complex macro sample is represented by 2x2 phase-only pixels.
    pp=np.repeat(np.repeat(plus,2,axis=0),2,axis=1);pm=np.repeat(np.repeat(minus,2,axis=0),2,axis=1)
    parity=(np.indices((fine.n,fine.n)).sum(axis=0)%2)==0
    encoded_phase=np.where(parity,pp,pm)
    commands=np.rint(np.mod(encoded_phase,2*np.pi)*256/(2*np.pi)).astype(np.int64)%256
    Image.fromarray(commands.astype(np.uint8)).save(out/'slm-phase-8bit.png')
    config={'wavelength_m':fine.wavelength,'macro_n':macro.n,'macro_pitch_m':macro.dx,
      'slm_n':fine.n,'slm_pitch_m':fine.dx,'macro_cell_slm_pixels':[2,2],
      'plane_distances_m':distances.tolist(),'phase_levels':256,'incident_amplitude':1.,
      'fourier_pupil_cutoff_per_m':cutoff,'complex_field_normalizer':normalizer,
      'generation':'sum of backward Fresnel fields from the two target planes',
      'encoding':'normalized double phase, complementary 2x2 checkerboard superpixels',
      'optical_replay':'phase-only SLM, normally incident plane wave, physical Fourier low-pass pupil, positive-distance Fresnel propagation',
      'coordinates':'fixed 4f relay inversion compensated by common upright display coordinates',
      'distances_are_assigned_not_document_values':True}
    (out/'calibration.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    replay,c=replay_slm(out/'slm-phase-8bit.png',out/'calibration.json')
    # The ideal complex-modulator reference has identical pixel hold and pupil,
    # so differences below isolate encoding/quantization, not different optics.
    complex_pixels=np.repeat(np.repeat(h,2,axis=0),2,axis=1)
    ideal_selected,_=bandlimit(complex_pixels,fine,cutoff)
    ideal_fields=np.asarray([propagate(ideal_selected,fine,d,method='fresnel',backend='scipy') for d in distances])
    exact_exit=np.exp(1j*encoded_phase)
    exact_selected,_=bandlimit(exact_exit,fine,cutoff)
    macro_average=exact_exit.reshape(macro.n,2,macro.n,2).mean(axis=(1,3))
    # Region containing the two shapes, plus a small margin, common to all views.
    rr,cc=np.where(depth>0)
    roi=np.array([2*(r0+rr.min())-30,2*(r0+rr.max()+1)+30,
                  2*(c0+cc.min())-30,2*(c0+cc.max()+1)+30],int)
    roi[[0,2]]=np.maximum(roi[[0,2]],0);roi[[1,3]]=np.minimum(roi[[1,3]],fine.n)
    sl=np.s_[roi[0]:roi[1],roi[2]:roi[3]]
    metrics={'source':source,'configuration':config,
      'macro_average_relative_error':relative_error(macro_average,h),
      'unquantized_optical_selected_field_relative_error':relative_error(exact_selected,ideal_selected),
      'quantized_optical_selected_field_relative_error':relative_error(replay['selected_wave'],ideal_selected),
      'slm_exit_intensity_min_max':[float(np.abs(replay['exit_wave']).min()**2),float(np.abs(replay['exit_wave']).max()**2)],
      'passed_power_fraction_of_incident':float(np.mean(np.abs(replay['selected_wave'])**2)),
      'plane_complex_relative_errors':[relative_error(u,v) for u,v in zip(replay['fields'],ideal_fields)],
      'plane_intensity_relative_errors_roi':[relative_error(np.abs(u[sl])**2,np.abs(v[sl])**2) for u,v in zip(replay['fields'],ideal_fields)],
      'comparison':'same optical pupil, pixel hold, input power and distance; no truth-dependent rescaling/registration',
      'limitations':['all results are computer simulations, not SLM experiment photographs',
       '40/60 mm are assigned here; document distances remain unknown',
       '2-plane image scene, initial phase zero, coherent scalar paraxial model',
       'ideal phase response with 256 levels and unity fill factor; no aberration or stray zero-order leakage',
       'finite spatial superpixel encoding and pupil cause residual errors and energy loss',
       'fixed relay image inversion shown in upright coordinates; no autofocus or GS iteration']}
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'fields.npz',macro_x_m=macro.x,x_m=fine.x,distances_m=distances,scene_roi_rc=roi,
      macro_layers=np.asarray(layers,dtype=np.complex64),calculated_hologram=h.astype(np.complex64),
      slm_phase_rad=replay['phase_rad'].astype(np.float32),slm_exit_wave=replay['exit_wave'].astype(np.complex64),
      optical_selected_wave=replay['selected_wave'].astype(np.complex64),ideal_selected_wave=ideal_selected.astype(np.complex64),
      reconstructed_fields=replay['fields'].astype(np.complex64),ideal_fields=ideal_fields.astype(np.complex64),pupil=replay['pupil'])
    def save(fig,name):fig.savefig(out/(name+'.png'),dpi=185,bbox_inches='tight',pad_inches=.12,facecolor='white');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    axs[0].imshow(rgb);axs[0].set_title('文档原始彩色图样');axs[0].axis('off')
    depth_display=np.full(depth.shape,np.nan);depth_display[depth==values[0]]=40;depth_display[depth==values[1]]=60
    v=axs[1].imshow(depth_display,cmap='viridis',vmin=40,vmax=60);axs[1].set_title('本次设定的两层距离');axs[1].axis('off');plt.colorbar(v,ax=axs[1],fraction=.04,pad=.03,ticks=[40,60],label='距SLM的距离 / mm');save(fig,'input')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');roi_all=[0,macro.n,0,macro.n]
    image(axs[0],np.abs(h)**2,macro.x,roi_all,'计算得到的全息面光强',float(np.abs(h).max()**2))
    image(axs[1],np.angle(h),macro.x,roi_all,'计算得到的全息面相位',phase=True);save(fig,'calculated-hologram')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    v=axs[0].imshow(commands,cmap='gray',vmin=0,vmax=255,interpolation='nearest');axs[0].set(title='实际送入模拟SLM的相位命令',xlabel='SLM列像素',ylabel='SLM行像素');plt.colorbar(v,ax=axs[0],fraction=.04,pad=.03,ticks=[0,128,255],label='相位命令 / 0–255')
    spec=np.abs(fft2c(replay['exit_wave']))**2;spec/=spec.max();f=fine.frequency/1000;df=(f[1]-f[0])/2
    v=axs[1].imshow(spec,extent=[f[0]-df,f[-1]+df,f[-1]+df,f[0]-df],cmap='gray',norm=LogNorm(vmin=1e-9,vmax=1),interpolation='nearest')
    axs[1].add_patch(Circle((0,0),cutoff/1000,fill=False,edgecolor='#D23B31',lw=1.8));axs[1].set(title='SLM出射场频谱与光学光阑',xlabel=r'$f_x$ / mm$^{-1}$',ylabel=r'$f_y$ / mm$^{-1}$');plt.colorbar(v,ax=axs[1],fraction=.04,pad=.03,label='归一化功率（对数）');save(fig,'slm-and-pupil')
    ref=max(float(np.abs(a[:,roi[0]:roi[1],roi[2]:roi[3]]).max()**2) for a in [ideal_fields,replay['fields']])
    fig,axs=plt.subplots(2,2,figsize=(10,9),layout='constrained')
    for j,d in enumerate(distances):
        image(axs[0,j],np.abs(ideal_fields[j])**2,fine.x,roi,f'理想复调制参考：{d*1000:.0f} mm',ref)
        image(axs[1,j],np.abs(replay['fields'][j])**2,fine.x,roi,f'纯相位SLM光学再现：{d*1000:.0f} mm',ref)
    save(fig,'optical-replay')
    fig,axs=plt.subplots(1,2,figsize=(10,5.5),layout='constrained')
    for j,d in enumerate(distances):image(axs[j],np.abs(replay['fields'][j])**2,fine.x,roi,f'SLM光学再现：{d*1000:.0f} mm',ref)
    save(fig,'replay-two-planes')
    text=f'''# 计算全息：数值生成、相位SLM、光学再现模拟

## 数据与距离
输入是作业文档原始嵌入的魔方/足球彩色图和两个灰度的深度图。文档未给定实际距离，本次明确设魔方40 mm、足球60 mm，不能将这些数值回填为文档原参数。灰度指定光强，振幅取平方根，初相位为0，分别经18/mm圆形光瞳限制带宽。

## 生成
波长632.8 nm，复场宏像素512²、间距8 μm。把两层目标分别按−40 mm、−60 mm菲涅耳传播到SLM面并相加，得到H。负距离只用于设计全息图，最终再现按正距离传播。

## 编码
归一化H=a exp(iφ)，0≤a≤1，取ψ±=φ±arccos(a)，则(exp(iψ+)+exp(iψ−))/2=H。每个宏像素用2×2 SLM子像素的互补棋盘排列，转换为256级相位命令PNG。模拟SLM1024²、间距4 μm，窗宽4.096 mm。

## 光学再现
`replay_slm`只读`slm-phase-8bit.png`和`calibration.json`。单位振幅平面波照明相位SLM，出射场exp(iψ)，出射光强处处为1。物理4f频谱面18/mm圆光阑选择目标低频级，再分别正向传播40/60 mm。FFT在这里用于计算光学传播，没有把原复场直接交给再现算法，也没有再现时读取原图/深度图。4f固定倒像按转正坐标展示。

## 结果
- 解析2×2宏像素平均误差：{metrics['macro_average_relative_error']:.4g}。
- 未量化双相位图经物理光阑后相对理想复调制误差：{100*metrics['unquantized_optical_selected_field_relative_error']:.3f}%。
- 256级相位SLM经光阑后复场误差：{100*metrics['quantized_optical_selected_field_relative_error']:.3f}%。
- 两层图样区域光强误差：{', '.join(f'{100*e:.3f}%' for e in metrics['plane_intensity_relative_errors_roi'])}。
- 光阑通过功率占整个入射平面波功率：{100*metrics['passed_power_fraction_of_incident']:.3f}%。
参考与SLM模拟共用像素保持、光学带宽、距离和光强显示标度，不依据真值拟合增益或配准。比较基准为理想复振幅调制器的同光路输出。两组光强统一γ=.30。

## 范围
这是简化相干标量模型的光学再现计算机模拟，不是实测SLM照片。包含空间双相位编码、256级量化、有限光学光阑和离焦。未模拟器件响应误差、填充因子、杂散零级、像差、遮挡或噪声。没有GS迭代或自聚焦。文档的逆传播图只能作为单独的数值验证，旧编码图不作为本次模拟的输入。

## 参考与复现
[Mendoza-Yero et al., Encoding complex fields by using a phase-only optical element (2014)](https://doi.org/10.1364/OL.39.001740)，两相位采样与4f低通原理。[双相位方法及256级棋盘编码示例](https://www.mdpi.com/2076-3417/13/7/4479)。
运行 `python -m optics_sim.cgh_slm_demo`。原始提取数据在`data/cgh_slm/two_planes/`。宏像素平均的解析恒等式并不意味着有限光学低通后误差为零，实际误差另行记录。
'''
    (out/'实验说明.md').write_text(text,encoding='utf-8')
    body=''.join(f'<h2>{label}</h2><img src="{key}.png">' for key,label in [('input','双深度输入'),('calculated-hologram','计算生成复场'),('slm-and-pupil','纯相位SLM与光学光阑'),('optical-replay','正向光学再现')])
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>计算全息SLM光学再现</title><style>body{font:18px/1.7 Microsoft YaHei,sans-serif;max-width:1150px;margin:36px auto;color:#24334b}img{width:100%}h1,h2{color:#174994}</style><h1>计算全息：SLM光学再现模拟</h1><p>文档图样重新建模，距离设为40/60 mm。由保存的256级相位图驱动模拟SLM，光照后经光学光阑并正向传播。</p>'+body+'<p><a href="实验说明.md">算法、数据与范围说明</a></p>',encoding='utf-8')
    print(json.dumps({k:metrics[k] for k in ['macro_average_relative_error','quantized_optical_selected_field_relative_error','plane_intensity_relative_errors_roi','passed_power_fraction_of_incident']},indent=2),flush=True)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);a=p.parse_args();run(a.output)
