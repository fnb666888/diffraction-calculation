"""Textbook single-plane Fourier phase-only CGH with Gerchberg-Saxton.

Design uses a public NASA photograph as a prescribed focal-plane intensity.
Replay reads only an exported 8-bit phase command and optical calibration.
The forward operator models an ideal front-/back-focal-plane lens transform.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
from scipy import fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from .metrics import relative_error
from . import phase_comparison_figures

ROOT=Path(__file__).resolve().parents[1]


def fft2c(a):
    return fft.fftshift(fft.fft2(fft.ifftshift(a),norm='ortho',workers=2))


def ifft2c(a):
    return fft.fftshift(fft.ifft2(fft.ifftshift(a),norm='ortho',workers=2))


def replay_phase_slm(command_path,calibration_path):
    """No photograph, target amplitude, target phase or saved ideal field input."""
    c=json.loads(Path(calibration_path).read_text(encoding='utf-8'))
    command=np.asarray(Image.open(command_path),np.float32)
    if command.shape!=(c['slm_n'],c['slm_n']):raise ValueError('SLM command shape mismatch')
    phase=command*(2*np.pi/c['phase_levels'])
    amplitude=np.sqrt(c['incident_discrete_power'])/c['slm_n']
    exit_wave=(amplitude*np.exp(1j*phase)).astype(np.complex64)
    focal_wave=fft2c(exit_wave)
    x=fft.fftshift(fft.fftfreq(c['slm_n'],c['slm_pitch_m']))*c['wavelength_m']*c['focal_length_m']
    return {'slm_phase_rad':phase,'slm_exit_wave':exit_wave,'focal_wave':focal_wave,
      'focal_intensity':np.abs(focal_wave)**2,'focal_x_m':x},c


def run(output=None,iterations=500):
    data=ROOT/'data/cgh_standard/astronaut'
    out=Path(output or ROOT/'outputs/cgh_standard/astronaut');out.mkdir(parents=True,exist_ok=True)
    source=np.asarray(Image.open(data/'source.png').convert('RGB'))
    resized=np.asarray(Image.fromarray(source).resize((256,256),Image.Resampling.LANCZOS),np.float32)/255
    gray=resized@np.array([.2126,.7152,.0722],np.float32)
    n=1024;patch=gray.shape[0];r0=(n-patch)//2;roi=[r0,r0+patch,r0,r0+patch];sl=np.s_[roi[0]:roi[1],roi[2]:roi[3]]
    target=np.zeros((n,n),np.float32);target[sl]=gray
    target/=target.sum(dtype=np.float64)
    amplitude=np.sqrt(target);incident_amplitude=np.float32(1/n)
    rng=np.random.default_rng(20261020)
    phase=rng.uniform(-np.pi,np.pi,(n,n)).astype(np.float32)
    history=[];snapshots={};snapshot_phases={}
    # GS imposes target amplitude in the focal plane, then uniform illumination
    # amplitude in the SLM plane, while retaining the currently computed phases.
    for k in range(iterations+1):
        slm=(incident_amplitude*np.exp(1j*phase)).astype(np.complex64)
        focal=fft2c(slm);intensity=np.abs(focal)**2
        history.append({'iteration':k,'amplitude_relative_error':relative_error(np.abs(focal),amplitude),
          'intensity_relative_error_full':relative_error(intensity,target),
          'intensity_relative_error_roi':relative_error(intensity[sl],target[sl]),
          'power_in_target_roi':float(intensity[sl].sum(dtype=np.float64))})
        if k in [0,1,20,iterations]:snapshots[str(k)]=intensity[sl].copy();snapshot_phases[str(k)]=phase.copy()
        if k%100==0:print(json.dumps(history[-1]),flush=True)
        if k==iterations:break
        constrained=amplitude*np.exp(1j*np.angle(focal))
        phase=np.angle(ifft2c(constrained.astype(np.complex64)))
    continuous=focal.copy()
    command=(np.rint(np.mod(phase,2*np.pi)*256/(2*np.pi)).astype(np.int16)%256).astype(np.uint8)
    Image.fromarray(command).save(out/'slm-phase-8bit.png')
    c={'slm_n':n,'slm_pitch_m':8e-6,'wavelength_m':632.8e-9,'focal_length_m':.200,
      'phase_levels':256,'incident_discrete_power':1.,'gs_iterations':iterations,'random_seed':20261020,
      'target_roi_rc':roi,'target_samples':[patch,patch],'target_intensity_sum':float(target.sum(dtype=np.float64)),
      'target_amplitude':'sqrt(grayscale intensity), globally normalized to unit discrete power',
      'target_phase':'unconstrained; the algorithm solves an intensity projection task',
      'slm_amplitude':'uniform, 1/N, unit incident discrete power',
      'initial_slm_phase':'uniform random [-pi,pi], deterministic seed',
      'geometry':'SLM in front focal plane, ideal Fourier lens, observation in back focal plane',
      'forward_operator':'centered unitary FFT, normalized power per focal sample',
      'focal_sample_pitch_m':632.8e-9*.200/(n*8e-6),
      'additional_fourier_lowpass':False,'double_phase_encoding':False,
      'display_gamma':.50,'model':'ideal discrete scalar Fourier-optics model',
      'precision':'complex64','assumptions':['ideal phase calibration and lens','no aberration or stray unmodulated reflection','no pixel-internal aperture/fill-factor response or higher pixel diffraction orders modeled','single monochromatic coherent illumination','simulation, not SLM experiment']}
    (out/'calibration.json').write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf-8')
    replay,c=replay_phase_slm(out/'slm-phase-8bit.png',out/'calibration.json')
    rec=replay['focal_intensity'];ref=target[sl]
    correlation=float(np.corrcoef(rec[sl].ravel(),ref.ravel())[0,1])
    metrics={'source':json.loads((data/'source.json').read_text(encoding='utf-8')),'configuration':c,
      'slm_phase_command_sha256':hashlib.sha256((out/'slm-phase-8bit.png').read_bytes()).hexdigest(),
      'continuous_phase_final':history[-1],
      'quantized_phase_final':{'intensity_relative_error_roi':relative_error(rec[sl],ref),
        'intensity_relative_error_full':relative_error(rec,target),
        'amplitude_relative_error_full':relative_error(np.sqrt(rec),amplitude),
        'intensity_correlation_roi':correlation,'power_in_target_roi':float(rec[sl].sum(dtype=np.float64)),
        'total_focal_discrete_power':float(rec.sum(dtype=np.float64)),
        'quantized_to_continuous_intensity_error_roi':relative_error(rec[sl],np.abs(continuous[sl])**2)},
      'evaluation':'unit incident/target power fixed before design; no fitted gain, registration, image filtering, denoising or reference used by replay'}
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    np.savez_compressed(out/'fields.npz',target_intensity=target,target_amplitude=amplitude,
      slm_phase_rad=replay['slm_phase_rad'],slm_exit_wave=replay['slm_exit_wave'],focal_wave=replay['focal_wave'],
      focal_x_m=replay['focal_x_m'],continuous_phase_focal_wave=continuous,
      **{f'iteration_{key}_intensity_roi':value for key,value in snapshots.items()})
    Image.fromarray(np.rint(gray*255).astype(np.uint8)).save(data/'target-gray.png')
    render(out,c,source,target,replay,history,snapshots)
    report=f'''# 标准计算全息：GS傅里叶相位全息

## 公开图片
输入为NASA宇航员Eileen Collins公开照片，经scikit-image公开数据仓库获取。原始512×512彩色照片保存于data/cgh_standard/astronaut/source.png，来源、许可及SHA-256保存在同目录source.json。彩图仅说明素材来源，实际计算采用缩小为256×256的灰度目标。使用现有RGB灰度权重0.2126/0.7152/0.0722指定相对光强，没有给照片虚构实测相位或深度。

## 标准光路与参数
单色632.8nm准直平面波照明1024×1024纯相位SLM，像素间距8μm，SLM位于理想透镜前焦面，焦距200mm，后焦面观察再现。该前后焦面几何对应光学傅里叶变换，由中心化FFT模拟。焦平面采样间距{1000*c['focal_sample_pitch_m']:.6f}mm，256像素图像宽{256*c['focal_sample_pitch_m']*1000:.3f}mm。不采用魔方/足球双层、WRP或棋盘双相位编码，也不添加原18/mm圆形低通。

## GS设计过程
灰度指定目标光强I_T，A_T=sqrt(I_T)，将整个目标离散功率归一为1。SLM上均匀振幅1/N，入射离散功率为1。固定随机种子20261020初始化SLM相位。在SLM面和透镜后焦面之间交替FFT/IFFT，每次在焦面替换目标振幅并保留当前相位，在SLM面恢复均匀振幅并保留当前相位，共{iterations}次。目标相位自由，因此这是光强投影设计，不宣称恢复照片原本的物理相位。

## 导出与光学再现模拟
最终相位量化为256级并保存slm-phase-8bit.png。replay_phase_slm只读取该图与calibration.json，建立已知均匀入射光，通过SLM调相，再模拟透镜光学傅里叶变换，光强为所得复场模平方。再现函数不读取照片、目标振幅、原始迭代复场，不在再现时重新施加目标约束。全程没有后处理滤波、去噪或真值亮度拟合。

## 当前结果
- 目标图像区域光强L2相对误差：{100*metrics['quantized_phase_final']['intensity_relative_error_roi']:.3f}%。
- 目标区域光强相关系数：{correlation:.6f}。
- 目标矩形内包含的离散功率：{100*metrics['quantized_phase_final']['power_in_target_roi']:.3f}%。
- 256级相位相对连续相位的目标区域光强变化：{100*metrics['quantized_phase_final']['quantized_to_continuous_intensity_error_roi']:.3f}%。
输入和再现共用同一光强显示标度、γ=0.50，物理误差由未做γ变换的光强计算。指标区域和归一化方式与旧双层案例不同，不直接比较旧百分数。

## 范围
这是经典GS方法的一种单平面傅里叶相位全息演示，不将其称为唯一标准方法。模型为理想离散标量傅里叶光学，不含实物SLM像素内部孔径、填充因子导致的包络和更高衍射级、杂散零级、透镜像差或噪声，不能将图像称为实物实验照片。单位离散功率对应正交FFT的归一化，图上显示相对光强，不冒用W/m²。无需自聚焦，焦平面由已知光路指定。

## 复现与依据
运行 `.venv/Scripts/python.exe -m optics_sim.cgh_standard`。

- 图片说明：https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut
- GS焦面设计原理：https://doi.org/10.1364/OE.389897
- 相位CGH及GS实现说明：https://www.mdpi.com/2076-3417/10/12/4283
'''
    (out/'实验说明.md').write_text(report,encoding='utf-8')
    panels=[('input','公开照片与目标光强'),('phase-convergence','GS相位全息图与迭代收敛'),('reconstruction','目标与相位SLM光学再现'),('iterations','迭代前后结果'),('full-field','整个焦平面与目标区域')]
    body=''.join(f'<h2>{title}</h2><img src="{key}.png">' for key,title in panels)
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>标准GS傅里叶计算全息</title><style>body{font:18px/1.7 Microsoft YaHei,sans-serif;max-width:1200px;margin:32px auto;color:#24334b}img{width:100%}h1,h2{color:#174994}</style><h1>GS傅里叶相位全息：计算、SLM与光学再现模拟</h1><p>公开NASA照片，1024²相位SLM，632.8nm，200mm傅里叶透镜。再现只读取导出的256级相位图与标定。</p>'+body+'<p><a href="实验说明.md">完整算法与模型范围</a></p>',encoding='utf-8')
    print(json.dumps(metrics['quantized_phase_final'],indent=2),flush=True)
    return metrics


def render(out,c,source,target,replay,history,snapshots):
    roi=c['target_roi_rc'];sl=np.s_[roi[0]:roi[1],roi[2]:roi[3]];x=replay['focal_x_m']*1000
    ext=[x[roi[2]],x[roi[3]-1],x[roi[1]-1],x[roi[0]]]
    ref=max(float(target.max()),float(replay['focal_intensity'][sl].max()))
    gamma=c['display_gamma']
    def save(fig,name):fig.savefig(out/(name+'.png'),dpi=185,bbox_inches='tight',pad_inches=.12,facecolor='white');plt.close(fig)
    def intensity(ax,a,title,extent=ext):
        im=ax.imshow(a/ref,cmap='gray',norm=PowerNorm(gamma,vmin=0,vmax=1),interpolation='nearest',extent=extent)
        ax.set(title=title,xlabel='x / mm',ylabel='y / mm');cb=plt.colorbar(im,ax=ax,fraction=.04,pad=.025,ticks=[0,.1,.5,1]);cb.set_label('归一化光强');return im
    fig,axs=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    axs[0].imshow(source);axs[0].set_title('NASA公开原始照片');axs[0].axis('off')
    intensity(axs[1],target[sl],'目标灰度光强');save(fig,'input')
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    phase=replay['slm_phase_rad'];v=axs[0].imshow(phase,cmap='RdBu_r',vmin=0,vmax=2*np.pi,interpolation='nearest');axs[0].set(title='256级SLM相位全息图',xlabel='SLM列像素',ylabel='SLM行像素');cb=plt.colorbar(v,ax=axs[0],fraction=.04,pad=.025,ticks=[0,np.pi,2*np.pi]);cb.ax.set_yticklabels(['0','π','2π']);cb.set_label('相位 / rad')
    axs[1].plot([h['iteration'] for h in history],[h['amplitude_relative_error'] for h in history],color='#174994',lw=2);axs[1].set(title='GS振幅约束的迭代收敛',xlabel='迭代次数',ylabel='全焦面振幅相对误差');axs[1].grid(alpha=.2);save(fig,'phase-convergence')
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    intensity(axs[0],target[sl],'目标光强');intensity(axs[1],replay['focal_intensity'][sl],'相位SLM光学再现')
    error=np.abs(replay['focal_intensity'][sl]-target[sl])/ref
    v=axs[2].imshow(error,cmap='gray',vmin=0,vmax=float(error.max()),extent=ext,interpolation='nearest');axs[2].set(title='绝对光强差（线性显示）',xlabel='x / mm',ylabel='y / mm');plt.colorbar(v,ax=axs[2],fraction=.04,pad=.025,label='|ΔI| / 共同光强标度');save(fig,'reconstruction')
    fig,axs=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for ax,key in zip(axs,['1','20',str(c['gs_iterations'])]):intensity(ax,snapshots[key],f'GS第{key}次迭代')
    save(fig,'iterations')
    fig,axs=plt.subplots(1,2,figsize=(10,4.8),layout='constrained')
    intensity(axs[0],target,'整个目标焦平面',[x[0],x[-1],x[-1],x[0]])
    intensity(axs[1],replay['focal_intensity'],'整个再现焦平面',[x[0],x[-1],x[-1],x[0]]);save(fig,'full-field')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--iterations',type=int,default=500);args=p.parse_args();run(args.output,args.iterations)
