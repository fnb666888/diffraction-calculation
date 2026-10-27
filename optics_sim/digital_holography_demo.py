"""Simulated optical recording, camera digitization and digital reconstruction.

The SEM photograph supplies a phase-target texture, not measured optical phase.
The inverse reads only camera counts and calibration, never source/phase truth.
"""
from pathlib import Path
import argparse
import json
import html
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal.windows import tukey
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, LogNorm
from matplotlib.patches import Circle
from .core import Grid, propagate, fft2c
from .holography import bandlimit, record, reconstruct_off_axis, carrier_conditions
from .metrics import relative_error
from .phase_comparison_figures import image

ROOT=Path(__file__).resolve().parents[1]


def reconstruct_camera(recording, calibration):
    """Recover the sensor-plane object wave from counts and calibration only."""
    c=json.loads(Path(calibration).read_text(encoding='utf-8'))
    counts=np.array(Image.open(recording),dtype=float)
    g=Grid(c['n'],c['dx_m'],c['wavelength_m'])
    intensity=(counts-c['bias_electrons']/c['electrons_per_adu'])/c['adu_per_intensity']
    x,_=g.xy
    reference=c['reference_amplitude']*np.exp(2j*np.pi*c['carrier_per_m']*x)
    sensor=reconstruct_off_axis(intensity,reference,g,c['bandwidth_per_m'],c['reference_amplitude'])
    return sensor,intensity,c


def phase_gradient_energy(field, dx_m, roi_rc):
    """Wrapped phase-gradient energy in a user-selected region of interest."""
    r0,r1,c0,c1=map(int,roi_rc)
    phase=np.angle(field[r0:r1,c0:c1])
    gx=np.angle(np.exp(1j*(phase[:,1:]-phase[:,:-1])))/dx_m
    gy=np.angle(np.exp(1j*(phase[1:,:]-phase[:-1,:])))/dx_m
    return float(np.mean(gx**2)+np.mean(gy**2))


def autofocus_reconstruct(sensor, grid, z_min_m, z_max_m, z_step_m, roi_rc):
    """Scan candidate planes and choose the maximum phase-gradient focus score."""
    values=(z_min_m,z_max_m,z_step_m)
    if not all(np.isfinite(v) for v in values) or z_min_m<0 or z_max_m<=z_min_m or z_step_m<=0:
        raise ValueError('自聚焦范围和步长必须为有限值，且 z_max > z_min、步长为正。')
    r0,r1,c0,c1=map(int,roi_rc)
    if not (0<=r0<r1<=grid.n and 0<=c0<c1<=grid.n) or (r1-r0)<3 or (c1-c0)<3:
        raise ValueError('自聚焦ROI必须是网格内至少3×3像素的矩形区域。')
    count=int(np.floor((z_max_m-z_min_m)/z_step_m+1e-9))+1
    z_values=z_min_m+np.arange(count)*z_step_m
    if z_values[-1] < z_max_m-z_step_m*1e-6:
        z_values=np.append(z_values,z_max_m)
    scores=np.empty(len(z_values),dtype=float)
    for i,z in enumerate(z_values):
        candidate=propagate(sensor,grid,-float(z),method='angular',backend='scipy')
        scores[i]=phase_gradient_energy(candidate,grid.dx,roi_rc)
    best=int(np.argmax(scores));z_best=float(z_values[best])
    # A parabolic fit around the discrete maximum refines z without another scan.
    if 0<best<len(scores)-1:
        left,center,right=scores[best-1:best+2]
        denom=left-2*center+right
        if denom<0:
            offset=float(np.clip(.5*(left-right)/denom,-1,1))
            z_best=float(np.clip(z_best+offset*z_step_m,z_min_m,z_max_m))
    field=propagate(sensor,grid,-z_best,method='angular',backend='scipy')
    return field,z_best,z_values,scores


def run(source=None,output=None):
    source=Path(source or ROOT/'data/digital_holography/diatom/source.png')
    out=Path(output or ROOT/'outputs/digital_holography/diatom');out.mkdir(parents=True,exist_ok=True)
    g=Grid(1024,2.5e-6,632.8e-9);z=.020;bandwidth=44000.;ar=1.5
    fc=370/g.size
    # Remove SEM instrument annotation by selecting the specimen ROI for modeling.
    # This crop defines the numerical input and does not alter the source file.
    src=Image.open(source).convert('L');source_box=(260,160,780,620)
    texture=src.crop(source_box)
    width=448;height=round(width*texture.height/texture.width)
    texture=np.asarray(texture.resize((width,height),Image.Resampling.LANCZOS),dtype=float)/255
    q0,q1=np.quantile(texture,[.01,.99]);texture=np.clip((texture-q0)/(q1-q0),0,1)
    texture=gaussian_filter(texture,.8)
    apod=tukey(height,.12)[:,None]*tukey(width,.12)[None,:]
    phase=np.zeros((g.n,g.n));r0=(g.n-height)//2;c0=(g.n-width)//2
    phase[r0:r0+height,c0:c0+width]=2.4*texture*apod
    raw=np.exp(1j*phase)  # Unit-amplitude phase object under plane-wave illumination.
    target,pupil=bandlimit(raw,g,bandwidth)
    sensor=propagate(target,g,z,method='angular',backend='scipy')
    x,_=g.xy;reference=ar*np.exp(2j*np.pi*fc*x)
    optical=record(sensor,reference)
    sampling=carrier_conditions(g,fc,bandwidth)
    if not sampling['separated']:raise ValueError('The off-axis orders are not separated.')
    # Point-sampled sensor model, with Poisson shot noise, read noise, ADC bias,
    # and 12-bit quantization. Pixel area integration/aberration is not modeled.
    full_well=40000.;bias=100.;read_noise=3.;peak_mean=30000.;bits=12
    e_per_i=peak_mean/float(optical.max());e_per_adu=full_well/(2**bits-1)
    rng=np.random.default_rng(20261020)
    electrons=rng.poisson(optical*e_per_i).astype(float)+rng.normal(0,read_noise,optical.shape)+bias
    unbounded=np.rint(electrons/e_per_adu)
    counts=np.clip(unbounded,0,2**bits-1).astype(np.uint16)
    Image.fromarray(counts).save(out/'camera-hologram-12bit.png')
    focus_roi=np.array([r0-12,r0+height+12,c0-12,c0+width+12])
    calibration={'n':g.n,'dx_m':g.dx,'wavelength_m':g.wavelength,'distance_m':z,
        'bandwidth_per_m':bandwidth,'carrier_per_m':fc,'reference_amplitude':ar,
        'autofocus_range_m':[.010,.030],'autofocus_step_m':.0005,
        'autofocus_roi_rc':focus_roi.tolist(),
        'electrons_per_adu':e_per_adu,'bias_electrons':bias,'adu_per_intensity':e_per_i/e_per_adu,
        'adc_bits':bits,'full_well_electrons':full_well,'peak_mean_electrons':peak_mean,
        'read_noise_electrons_rms':read_noise,'seed':20261020,
        'sensor_model':'point sampling; Poisson shot noise, Gaussian read noise, fixed bias, 12-bit ADC',
        'propagation':'periodic discrete exact angular-spectrum propagation; common axial phase removed'}
    (out/'calibration.json').write_text(json.dumps(calibration,ensure_ascii=False,indent=2),encoding='utf-8')
    recovered_sensor,measured,_=reconstruct_camera(out/'camera-hologram-12bit.png',out/'calibration.json')
    recovered,autofocus_z,focus_z,focus_scores=autofocus_reconstruct(
        recovered_sensor,g,*calibration['autofocus_range_m'],calibration['autofocus_step_m'],focus_roi)
    ideal_sensor=reconstruct_off_axis(optical,reference,g,bandwidth,ar)
    ideal=propagate(ideal_sensor,g,-z,method='angular',backend='scipy')
    roi=np.array([r0-12,r0+height+12,c0-12,c0+width+12]);sl=np.s_[roi[0]:roi[1],roi[2]:roi[3]]
    difference=np.angle(recovered*np.conj(target))
    metrics={'configuration':calibration,'sampling':sampling,
      'source_metadata':json.loads(source.with_name('source.json').read_text(encoding='utf-8')),
      'source_roi_xyxy':source_box,'object_roi_rc':roi.tolist(),
      'phase_model':'photograph grayscale mapped to 0..2.4 rad and tapered at the image boundary; raw amplitude=1',
      'optical_phase_is_assigned_not_measured':True,
      'pupil_retained_power':float(np.sum(np.abs(target)**2)/np.sum(np.abs(raw)**2)),
      'ideal_complex_relative_error':relative_error(ideal,target),
      'camera_complex_relative_error_full_grid':relative_error(recovered,target),
      'camera_complex_relative_error_photo_roi':relative_error(recovered[sl],target[sl]),
      'intensity_relative_error_photo_roi':relative_error(np.abs(recovered[sl])**2,np.abs(target[sl])**2),
      'phase_rmse_photo_roi_rad':float(np.sqrt(np.mean(difference[sl]**2))),
      'phase_evaluation':'full photo rectangle including background, no phase mask or truth-dependent alignment',
      'autofocus':{'method':'maximum wrapped phase-gradient energy','selected_z_m':autofocus_z,
        'range_m':calibration['autofocus_range_m'],'step_m':calibration['autofocus_step_m'],
        'roi_rc':focus_roi.tolist(),'roi_source':'simulated object bounds; real data require a user-selected ROI',
        'focus_score_at_selected_z':float(np.interp(autofocus_z,focus_z,focus_scores))},
      'saturated_fraction':float(np.mean(unbounded>=2**bits-1)),
      'limitations':['SEM texture supplies assigned optical phase, not physical sample thickness',
       'finite-NA target is the comparison truth; source gray levels are not used by the inverse',
       'camera noise is a simplified model; no spatial pixel integration, lens aberration, or calibration error',
       'autofocus searches only the supplied z range and ROI; the demo ROI is derived from the simulated object bounds, and the selected distance is not guaranteed outside the search range',
       'single-frame off-axis reconstruction does not use iterative phase retrieval or phase unwrapping']}
    scan_z=np.array([.010,.020,.030]);scan=np.asarray([propagate(recovered_sensor,g,-d,method='angular',backend='scipy') for d in scan_z])
    np.savez_compressed(out/'fields.npz',x_m=g.x,photo_roi_rc=roi,autofocus_roi_rc=focus_roi,
        autofocus_z_m=autofocus_z,autofocus_scan_z_m=focus_z,autofocus_focus_score=focus_scores,
        autofocus_field=recovered,
        assigned_phase_rad=phase,raw_object=raw,effective_input=target,pupil=pupil,
        optical_hologram_intensity=optical,camera_counts=counts,recovered_sensor=recovered_sensor,
        recovered_object=recovered,phase_error_rad=difference,distances_m=scan_z,refocus_fields=scan)
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    def save(fig,name):
        fig.savefig(out/(name+'.png'),dpi=190,bbox_inches='tight',pad_inches=.12,facecolor='white');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    axs[0].imshow(texture,cmap='gray',vmin=0,vmax=1);axs[0].axis('off');axs[0].set_title('硅藻照片纹理')
    iref=max(float(np.abs(target[sl]).max()**2),float(np.abs(recovered[sl]).max()**2))
    image(axs[1],np.abs(target)**2,g.x,roi,'有效输入光强',iref)
    image(axs[2],np.angle(target),g.x,roi,'有效输入相位',phase=True)
    save(fig,'input')
    fig,axs=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    ext=[g.x[0]*1e3,g.x[-1]*1e3,g.x[-1]*1e3,g.x[0]*1e3]
    v=axs[0].imshow(counts,extent=ext,cmap='gray',norm=PowerNorm(.30,vmin=0,vmax=4095),interpolation='nearest')
    axs[0].set(title='相机记录的干涉光强',xlabel='x / mm',ylabel='y / mm');plt.colorbar(v,ax=axs[0],fraction=.04,pad=.025,label='12位相机计数 / ADU')
    spectrum=np.abs(fft2c(measured))**2;spectrum/=spectrum.max();f=g.frequency/1000;h=(f[1]-f[0])/2
    v=axs[1].imshow(spectrum,extent=[f[0]-h,f[-1]+h,f[-1]+h,f[0]-h],cmap='gray',norm=LogNorm(vmin=1e-9,vmax=1),interpolation='nearest')
    axs[1].add_patch(Circle((-fc/1000,0),bandwidth/1000,fill=False,edgecolor='#D23B31',lw=1.8))
    axs[1].set(title='一级频谱选取',xlabel=r'$f_x$ / mm$^{-1}$',ylabel=r'$f_y$ / mm$^{-1}$')
    plt.colorbar(v,ax=axs[1],fraction=.04,pad=.025,label='归一化频谱功率（对数）')
    save(fig,'recording')
    fig,axs=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    image(axs[0,0],np.abs(target)**2,g.x,roi,'有效输入光强',iref)
    image(axs[0,1],np.abs(recovered)**2,g.x,roi,'数字重建光强',iref)
    image(axs[1,0],np.angle(target),g.x,roi,'有效输入相位',phase=True)
    image(axs[1,1],np.angle(recovered),g.x,roi,'数字重建相位',phase=True)
    extent=[g.x[roi[2]]*1e6,g.x[roi[3]-1]*1e6,g.x[roi[1]-1]*1e6,g.x[roi[0]]*1e6]
    ierr=(np.abs(recovered)**2-np.abs(target)**2)/iref
    lim_i=float(np.max(np.abs(ierr[sl])));lim_p=float(np.max(np.abs(difference[sl])))
    for ax,a,title_,lim,unit in [(axs[0,2],ierr,'光强差（重建−输入）',lim_i,'归一化光强差'),(axs[1,2],difference,'相位差（重建−输入）',lim_p,'相位差 / rad')]:
        v=ax.imshow(a[sl],extent=extent,cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
        ax.set(title=title_,xlabel='x / μm',ylabel='y / μm');plt.colorbar(v,ax=ax,fraction=.04,pad=.025,label=unit)
    save(fig,'reconstruction')
    fig,axs=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    for ax,d,u in zip(axs,scan_z,scan):image(ax,np.angle(u),g.x,roi,f'数值重建距离 {d*1000:.0f} mm',phase=True)
    save(fig,'refocus')
    fig,ax=plt.subplots(figsize=(7.5,4.2),layout='constrained')
    ax.plot(focus_z*1e3,focus_scores/focus_scores.max(),color='#24579a',lw=2)
    ax.axvline(autofocus_z*1e3,color='#c43b35',ls='--',lw=1.8,label=f'自动选焦 z = {autofocus_z*1e3:.2f} mm')
    ax.set(title='相位梯度能量自聚焦',xlabel='候选重建距离 z / mm',ylabel='归一化焦点评分')
    ax.grid(alpha=.25);ax.legend();save(fig,'autofocus')
    fig,ax=plt.subplots(figsize=(5.5,1.1),layout='constrained')
    ax.plot(focus_z*1e3,focus_scores/focus_scores.max(),color='#24579a',lw=1.8)
    ax.axvline(autofocus_z*1e3,color='#c43b35',ls='--',lw=1.5,label=f'z = {autofocus_z*1e3:.2f} mm')
    ax.set(xlim=(focus_z[0]*1e3,focus_z[-1]*1e3),xlabel='候选重建距离 z / mm',yticks=[])
    ax.tick_params(axis='x',labelsize=9);ax.grid(axis='x',alpha=.2);ax.legend(fontsize=9,loc='upper left')
    save(fig,'autofocus_compact')
    description=f'''# 数字全息模拟：硅藻图像相位靶标

## 流程
平面波照明相位靶标，角谱传播20 mm，叠加倾斜参考光，计算干涉光强。相机模型加入光子噪声、3 e⁻ RMS读出噪声和12位量化。重建函数只读取保存的相机计数图片与标定文件，分离一级频谱、解调参考光；随后在标定给出的 z 范围内逐平面反向传播，计算选区内的包裹相位梯度能量，并自动选取指标峰值。本仿真选区按模拟目标边界设置；实测数据需由用户指定选区。

## 输入与约定
- [原图及CC0许可]({metrics['source_metadata']['page']})，作者Reconrabbit。保留原始文件，选取硅藻区域用于输入。
- SEM照片只作为纹理，归一化灰度映射为0到2.4 rad相位，边缘加渐变窗。原始振幅为1。不是硅藻实测光学相位或厚度。
- 圆形光瞳带宽44/mm，比较真值为带限后的有效输入场，输入、重建使用同一光强标度和γ=0.30，相位固定−π到π，不屏蔽暗区。
- 网格1024²、间距2.5 μm、波长632.8 nm。载频{fc/1000:.5f}/mm，参考振幅1.5。满足3B<fc<fNy−B。
- 相机使用理想点采样，满阱40000 e⁻，光强峰值期望30000 e⁻，偏置100 e⁻，随机种子20261020。未模拟像素面积积分或像差。

## 结果
- 照片矩形区域内光强相对误差：{100*metrics['intensity_relative_error_photo_roi']:.3f}%。
- 同一区域内相位RMSE：{metrics['phase_rmse_photo_roi_rad']:.5f} rad。包含全部像素，无相位屏蔽或真值相位对齐。
- 理想无噪声逆传播复场相对误差：{metrics['ideal_complex_relative_error']:.3g}。
- ADC饱和比例：{100*metrics['saturated_fraction']:.4f}%。
- 自动搜索范围为{calibration['autofocus_range_m'][0]*1e3:g}–{calibration['autofocus_range_m'][1]*1e3:g} mm，步长{calibration['autofocus_step_m']*1e3:g} mm；算法选出 z={autofocus_z*1e3:.2f} mm。本仿真ROI按模拟目标边界设置，实测数据需由用户指定。
- [查看自聚焦评分曲线](autofocus.png)。10/20/30 mm图像展示同一记录的数值再聚焦。

## 算法与概念
现有离轴强度记录与数值逆传播同样属于数字全息模拟。光学全息、数字全息与计算全息可描述不同记录/再现环节，不构成彼此排斥的三类。此例强调相机采样和数字复振幅恢复。单帧离轴重建不使用HIO、GS、迭代恢复或相位展开，恢复相位来自复场的atan2。

参考：[Cuche et al., Digital holography for quantitative phase-contrast imaging (1999)](https://doi.org/10.1364/OL.24.000291)。

## 复现
运行 `python -m optics_sim.digital_holography_demo`。`camera-hologram-12bit.png`以16位PNG容器保存0..4095的真实ADC数值，普通查看器可能显示偏暗，演示图为`recording.png`。`calibration.json`保存重建、自聚焦范围和用户ROI需要的标定。`fields.npz`保存复场真值供事后评价，不是逆算法输入。`metrics.json`记录参数与误差。
'''
    (out/'实验说明.md').write_text(description,encoding='utf-8')
    body=''.join(f'<h2>{label}</h2><img src="{name}.png">' for name,label in [('input','输入相位靶标'),('recording','模拟相机记录'),('reconstruction','数字重建'),('autofocus','自聚焦评分曲线'),('refocus','数字再聚焦')])
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>数字全息模拟</title><style>body{font:18px/1.7 Microsoft YaHei,sans-serif;max-width:1200px;margin:36px auto;color:#24334b}img{width:100%}h1,h2{color:#174994}</style><h1>数字全息模拟</h1><p>硅藻照片提供模拟相位纹理。计算机模拟光学干涉记录、相机采样和数字重建，未使用真实光学测量数据。</p>'+body+'<p><a href="实验说明.md">参数与算法说明</a></p>',encoding='utf-8')
    print(json.dumps({k:metrics[k] for k in ['ideal_complex_relative_error','camera_complex_relative_error_photo_roi','intensity_relative_error_photo_roi','phase_rmse_photo_roi_rad','saturated_fraction']},indent=2),flush=True)
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();run(args.source,args.output)
