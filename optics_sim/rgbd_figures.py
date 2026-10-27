"""Scientific figures/report for the measured coffee-mug RGB-D experiment."""
from pathlib import Path
import argparse
import html
import io
import json
import numpy as np
from scipy import fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm,LogNorm
from matplotlib.patches import Circle
from PIL import Image
from .core import Grid,propagate,fft2c
from .phase_comparison_figures import image,crop

ROOT=Path(__file__).resolve().parents[1]


def run(directory=None):
    out=Path(directory or ROOT/'outputs/rgbd_holography/coffee_mug_1').resolve()
    m=json.loads((out/'metrics.json').read_text(encoding='utf-8'));selected=m['selected_wrp']
    with np.load(out/'data.npz') as d:
        arrays={k:d[k] for k in d.files}
    config=m['configuration'];g=Grid(config['n'],config['dx_m'],config['wavelength_m'])
    rows=arrays['rows'];cols=arrays['cols'];pad=24
    roi=np.array([max(0,int(rows.min())-pad),min(g.n,int(rows.max())+pad+1),
                  max(0,int(cols.min())-pad),min(g.n,int(cols.max())+pad+1)])
    raw=arrays['raw_xyz_m'];rgb=arrays['rgb'];imxy=arrays['image_xy']
    xx=imxy[:,0]-imxy[:,0].min();yy=imxy[:,1]-imxy[:,1].min()
    rgb_image=np.full((yy.max()+1,xx.max()+1,3),255,np.uint8);rgb_image[yy,xx]=rgb
    depth=np.full(rgb_image.shape[:2],np.nan);depth[yy,xx]=raw[:,1]*1000
    Image.fromarray(rgb_image).save(out/'input-color.png')
    fig=plt.figure(figsize=(13,4.3),layout='constrained')
    ax=fig.add_subplot(131);ax.imshow(rgb_image);ax.axis('off');ax.set_title('真实咖啡杯颜色投影')
    ax=fig.add_subplot(132);q=ax.imshow(depth,cmap='viridis');ax.axis('off');ax.set_title('采集深度');cb=plt.colorbar(q,ax=ax,fraction=.04,pad=.03);cb.set_label('深度 / mm')
    ax=fig.add_subplot(133,projection='3d');ax.scatter(raw[:,0]*1000,raw[:,1]*1000,raw[:,2]*1000,c=rgb/255,s=1,depthshade=False)
    ax.set(xlabel='x / mm',ylabel='深度 / mm',zlabel='竖直 / mm',title=f'单视角点云，{len(raw)}点')
    ax.set_box_aspect(np.ptp(raw,axis=0));ax.view_init(elev=15,azim=-70)
    fig.savefig(out/'input-preview.png',dpi=180,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    distances=np.quantile(arrays['distance_m'],[.05,.5,.95])
    methods=['layered_fft',selected];recon={}
    for name in methods:
        sensor=arrays[name+'_recovered_sensor_field']
        recon[name]=[propagate(sensor,g,-float(z),backend='scipy') for z in distances]
    ref=max(float(np.abs(u).max()**2) for key in methods for u in recon[key])
    fig,axes=plt.subplots(2,3,figsize=(12,9),layout='constrained')
    for i,name in enumerate(methods):
        for j,(z,u) in enumerate(zip(distances,recon[name])):
            image(axes[i,j],np.abs(u)**2,g.x,roi,f'重建距离 {z*1000:.3f} mm' if i==0 else '',ref)
        axes[i,0].text(-.36,.5,'分层FFT基准' if i==0 else '局部WRP',rotation=90,va='center',ha='center',
                       fontsize=18,color='#174994',fontweight='bold',transform=axes[i,0].transAxes)
    fig.savefig(out/'reconstruction.png',dpi=180,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    # Common propagation is unitary, so this whole-grid error should be stable with distance.
    reconstruction_errors=[float(np.linalg.norm(a-b)/np.linalg.norm(a)) for a,b in zip(recon['layered_fft'],recon[selected])]
    np.savez_compressed(out/'reconstruction.npz',distances_m=distances,photo_roi_rc=roi,
      object_x_m=g.x,baseline_fields=np.asarray(recon['layered_fft']),wrp_fields=np.asarray(recon[selected]))
    # A single known hologram, several fixed propagation distances; no autofocus selection.
    scan_distances=np.linspace(float(np.quantile(arrays['distance_m'],.03))-.0001,
                               float(np.quantile(arrays['distance_m'],.97))+.0001,21)
    intensities=[]
    for z in scan_distances:
        u=propagate(arrays[selected+'_recovered_sensor_field'],g,-float(z),backend='scipy')
        intensities.append(crop(np.abs(u)**2,roi))
    scan_ref=max(float(a.max()) for a in intensities)
    depth_groups=[arrays['distance_m']<=np.quantile(arrays['distance_m'],.2),
                  arrays['distance_m']>=np.quantile(arrays['distance_m'],.8)]
    energy=np.array([[float(np.average(value[rows[mask]-roi[0],cols[mask]-roi[2]],weights=arrays['amplitude'][mask]))
                     for mask in depth_groups] for value in intensities])
    medians=[float(np.median(arrays['distance_m'][mask])) for mask in depth_groups]
    peaks=[float(scan_distances[np.argmax(energy[:,j])]) for j in range(2)]
    fig,ax=plt.subplots(figsize=(8.3,4.2),layout='constrained')
    for j,(label,color) in enumerate([('较近20%物点','#174994'),('较远20%物点','#AE1324')]):
        ax.plot(scan_distances*1000,energy[:,j]/energy[:,j].max(),label=label,color=color,lw=2)
        ax.axvline(medians[j]*1000,color=color,ls='--',lw=1,alpha=.6)
    ax.set(xlabel='重建距离 / mm',ylabel='物点位置光强（按各组峰值归一化）',ylim=(.7,1.03),
           title='不同深度物点的聚焦检查')
    ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'focus-evaluation.png',dpi=180,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    np.savez(out/'focus-evaluation.npz',distances_m=scan_distances,point_energy=energy,group_median_depth_m=medians)
    fig,ax=plt.subplots(figsize=(4.5,5.8),layout='constrained')
    r0,r1,c0,c1=roi;half=g.dx/2
    ext=np.array([g.x[c0]-half,g.x[c1-1]+half,g.x[r1-1]+half,g.x[r0]-half])*1e6
    im=ax.imshow(intensities[0]/scan_ref,cmap='gray',norm=PowerNorm(.30,0,1),extent=ext,interpolation='nearest')
    ax.set(xlabel='x / μm',ylabel='y / μm');ax.set_title('');frames=[]
    for z,value in zip(scan_distances,intensities):
        im.set_data(value/scan_ref);ax.set_title(f'WRP全息重建  {z*1000:.3f} mm',fontsize=16)
        b=io.BytesIO();fig.savefig(b,format='png',dpi=110,facecolor='white');b.seek(0)
        frames.append(Image.open(b).convert('RGB').copy())
    frames[0].save(out/'focus-scan.gif',save_all=True,append_images=frames[1:]+frames[-2:0:-1],duration=220,loop=0)
    plt.close(fig)
    # A static overview at the median depth, plus true input color/depth.
    fig,axes=plt.subplots(1,3,figsize=(11,4.7),layout='constrained')
    axes[0].imshow(rgb_image);axes[0].axis('off');axes[0].set_title('原始点云颜色投影')
    q=axes[1].imshow(depth,cmap='viridis');axes[1].axis('off');axes[1].set_title('真实采集深度 / mm');plt.colorbar(q,ax=axes[1],fraction=.04,pad=.02)
    image(axes[2],np.abs(recon[selected][1])**2,g.x,roi,f'WRP重建光强\n{distances[1]*1000:.3f} mm',ref)
    fig.savefig(out/'overview.png',dpi=180,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,5),layout='constrained')
    holo=arrays[selected+'_hologram_intensity'];spectrum=np.abs(fft2c(holo))**2
    im=axes[0].imshow(holo/holo.max(),cmap='gray',norm=PowerNorm(.30,0,1),extent=np.array([-g.size/2,g.size/2,g.size/2,-g.size/2])*1e6)
    axes[0].set(title='咖啡杯离轴全息干涉光强',xlabel='x / μm',ylabel='y / μm');plt.colorbar(im,ax=axes[0],fraction=.04,pad=.02)
    f=g.frequency/1000;df=(f[1]-f[0])/2
    im=axes[1].imshow(spectrum/spectrum.max(),cmap='gray',norm=LogNorm(1e-8,1),extent=[f[0]-df,f[-1]+df,f[-1]+df,f[0]-df])
    axes[1].add_patch(Circle((-config['carrier_per_m']/1000,0),config['bandwidth_per_m']/1000,fill=False,edgecolor='#AE1324',lw=1.5))
    axes[1].set(title='全息图二维频谱',xlabel=r'$f_x$ / mm$^{-1}$',ylabel=r'$f_y$ / mm$^{-1}$');plt.colorbar(im,ax=axes[1],fraction=.04,pad=.02)
    fig.savefig(out/'hologram.png',dpi=180,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    m['reconstruction']={'distances_m':distances.tolist(),'wrp_vs_baseline_complex_relative_error':reconstruction_errors,
                         'autofocus':False,'scan_distances_m':scan_distances.tolist(),'gamma':.30,
                         'intensity_reference':ref,'scan_intensity_reference':scan_ref}
    m['focus_evaluation']={'groups':'nearest/farthest 20% points by known model depth',
        'metric':'amplitude-weighted mean reconstructed intensity at known point x/y positions',
        'group_median_depth_m':medians,'scan_peak_distance_m':peaks,
        'scope':'post-reconstruction diagnostic using known point locations, not autofocus or inverse input'}
    (out/'metrics.json').write_text(json.dumps(m,ensure_ascii=False,indent=2),encoding='utf-8')
    write_report(out,m)
    print(json.dumps({'output':str(out),'distances_mm':(distances*1000).tolist(),'reconstruction_errors':reconstruction_errors}))


def write_report(out,m):
    cfg=m['configuration'];selected=m['selected_wrp'];a=m['methods'];base=a['layered_fft']['median_ms'];point=a['full_point_sum']['median_ms'];chosen=a[selected]
    labels={'layered_fft':'分层FFT（71个原始深度层）','full_point_sum':'整面逐点求和（可分离矩阵优化）','wrp_1':'WRP窗口系数1.0','wrp_1.5':'WRP窗口系数1.5','wrp_2':'WRP窗口系数2.0'}
    rows='\n'.join(f"| {labels[k]} | {v['median_ms']:.1f} | {v['sensor_complex_relative_error']*100:.3f}% | {v['sensor_intensity_relative_error']*100:.3f}% |" for k,v in a.items())
    fe=m['focus_evaluation'];near,far=np.array(fe['group_median_depth_m'])*1000;near_peak,far_peak=np.array(fe['scan_peak_distance_m'])*1000
    md=f'''# 真实咖啡杯点云的三维全息试算

## 数据与模型

使用华盛顿大学RGB-D Object Dataset的coffee_mug_1，帧coffee_mug_1_1_100.pcd，共{cfg['source_point_count']}个彩色点，{cfg['original_depth_levels_used']}个原始深度值。原始点云为单视角可见表面，不包含不可见背面。颜色和深度来自实际采集。

![原始颜色、深度和点云](input-preview.png)

整个几何等比例缩小{cfg['uniform_geometric_scale']:.8f}倍，物体高度0.8 mm。相对深度与几何比例一起缩放，不把颜色灰度当成深度。物面到全息面的距离为{cfg['object_to_hologram_range_m'][0]*1000:.3f}–{cfg['object_to_hologram_range_m'][1]*1000:.3f} mm。横向坐标投到光学网格，最大舍入误差{cfg['lateral_rasterization_max_error_m']*1e6:.3f} μm；原有71个深度层全部保留，没有新增深度分箱。

光学设置为1024×1024，间距1.5 μm，波长632.8 nm，物光带宽75/mm，参考光振幅1.5。颜色亮度只作为数值振幅权重。采用共同平面波照明下的零初始包络相位，省略共同轴向载波；真实物体的光学反射相位并未测量。该模拟为相干标量点源/离散菲涅耳模型，不模拟不透明遮挡传播、多次散射或RGB三波长。

## 全息记录与多距离重建

各方法生成全息面物光，与同一倾斜参考光叠加得到各自干涉光强。重建只使用该光强、参考光和已知带宽，经解调、分离一级和反向传播得到复场。不把原始点云或生成物光直接作为重建输入。

![全息记录与二维频谱](hologram.png)

从点云距离分布选取5%、50%、95%分位对应的三个固定重建距离。单个三维物体的不同部分处于不同深度，离焦量随重建距离改变。没有将物体复制到多个平面，没有使用自聚焦自动选面。光强共用标度并使用γ=0.30，所有数值增强仅影响显示。

![分层FFT和WRP在三个距离的重建](reconstruction.png)

![同一全息图的连续距离扫描](focus-scan.gif)

### 深度聚焦检查

事后按已知点云深度选取最近20%和最远20%物点，统计各组已知横向位置处的振幅加权平均重建光强。下图各曲线按自身峰值归一化，虚线为该组真实模型深度的中位数。近组深度中位数{near:.3f} mm、扫描能量峰值{near_peak:.3f} mm，远组深度中位数{far:.3f} mm、扫描能量峰值{far_peak:.3f} mm。这是使用已知点位置的事后检查，不作为反演输入，也没有据此自动选择前述三个重建平面。

![不同深度的聚焦检查](focus-evaluation.png)

重建中有网格状相干纹理，来自离散点源、有限采样和相干叠加。本次没有做表面插值或散斑抑制。

## 耗时和误差

相同输入、采样、精度、传播核和光瞳，CPU单线程complex128，预热后轮换顺序各3次，取中位数。仅计缓存传播核后的前向物光生成，包含公共光瞳，排除预计算、文件读写、绘图、参考光记录及逆重建。整面逐点基准利用菲涅耳可分离性和矩阵乘法优化，不等同于最朴素的逐像素指数求和。全套核预计算另耗时{cfg['kernel_setup_seconds']:.3f} s。

| 方法 | 耗时 / ms | 物光复场相对误差 | 物光光强相对误差 |
| --- | ---: | ---: | ---: |
{rows}

误差相对分层FFT基准计算。数值近似相等的结果不代表实测成像误差为零。WRP在所有物点与全息面之间设置共同虚拟平面，距全息面{cfg['wrp_to_hologram_m']*1000:.3f} mm。每个物点仅在局部矩形窗口内用相同的近传播离散核累加复数场，然后进行一次整面传播。窗口半径由近传播距离、波长、采样间距及扩展系数确定，并增加8像素裕量。

预先设定选取规则：复场误差≤1%时取最快的WRP版本，否则取误差最小者。本次选择窗口系数1.5，耗时{chosen['median_ms']/1000:.3f} s，相对分层FFT加速{base/chosen['median_ms']:.2f}倍，相对优化逐点求和加速{point/chosen['median_ms']:.2f}倍，物光复场误差{chosen['sensor_complex_relative_error']*100:.3f}%，物光光强误差{chosen['sensor_intensity_relative_error']*100:.3f}%。结果只适用于本次点云、参数、缓存与CPU实现，不能外推所有三维物体。

## 复现与原始数据

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& '.venv/Scripts/python.exe' -m optics_sim.rgbd_holography --repeats 3
& '.venv/Scripts/python.exe' -m optics_sim.rgbd_figures
```

下载归档和选中原始帧在data/rgbd/coffee_mug_1。source.json保存下载地址、字节数和SHA256。当前目录data.npz保存原点云、变换坐标、全部方法的物光、两种方法的记录光强和重建记录面复场。reconstruction.npz保存三个距离的复场，metrics.json保存逐次耗时、误差和参数。动画可从保存的记录面复场重算。

## 来源

- [华盛顿大学RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/)，限非商业科研和教学使用。
- Lai, Bo, Ren, Fox, A Large-Scale Hierarchical Multi-View RGB-D Object Dataset, ICRA2011。
- [点云格式及坐标说明](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/README.txt)。
- Shimobaba, Masuda, Ito, Simple and fast calculation algorithm for computer-generated hologram with wavefront recording plane, Optics Letters34,3133–3135(2009), [DOI](https://doi.org/10.1364/OL.34.003133)。
'''
    (out/'实验结果.md').write_text(md,encoding='utf-8')
    tr=''.join(f"<tr><td>{labels[k]}</td><td>{v['median_ms']:.1f}</td><td>{v['sensor_complex_relative_error']*100:.3f}%</td></tr>" for k,v in a.items())
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>咖啡杯三维全息试算</title>
<style>body{{max-width:1080px;margin:40px auto;padding:0 24px;font-family:'Microsoft YaHei',sans-serif;color:#26323d;line-height:1.8}}h1,h2{{color:#174994}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #dbe1e8;padding:10px;text-align:left}}th{{background:#174994;color:white}}.animation{{max-width:450px;display:block;margin:auto}}</style>
<h1>真实咖啡杯点云的三维全息试算</h1><p>华盛顿大学RGB-D Object Dataset，5521个真实彩色点、71个原始深度层。等比例缩小到0.8 mm高，单视角可见表面。颜色只指定数值振幅，光学相位为模型设置。</p>
<img src="input-preview.png" alt="原始颜色、深度和三维点云"><h2>全息记录</h2><img src="hologram.png" alt="咖啡杯全息图及频谱"><h2>不同距离的重建</h2><p>同一全息图在不同距离逆传播，各图共用标度，γ=0.30。重建使用记录光强和已知参考光，没有读取原点云或自动选焦。</p><img src="reconstruction.png" alt="分层FFT与WRP重建比较"><img class="animation" src="focus-scan.gif" alt="连续距离重建动画">
<h2>深度聚焦检查</h2><p>已知较近、较远两组物点的模型深度中位数为{near:.3f}、{far:.3f} mm，扫描光强峰值分别位于{near_peak:.3f}、{far_peak:.3f} mm。曲线仅用于事后检查，不作为反演输入或自动选焦依据。重建仍有离散相干纹理。</p><img src="focus-evaluation.png" alt="近、远两组物点的聚焦检查">
<h2>速度与误差</h2><p>缓存核、CPU单线程、complex128，预热后各3次中位数，计前向物光生成。WRP选中版本相对分层FFT快{base/chosen['median_ms']:.2f}倍，物光复场误差{chosen['sensor_complex_relative_error']*100:.3f}%。</p><table><tr><th>方法</th><th>耗时 / ms</th><th>复场误差</th></tr>{tr}</table>
<p>这是当前点云和数值设置下的结果。没有模拟不可见背面、遮挡传输、多次散射和真实物体反射相位。</p><p>完整参数、复现命令及数据说明见同目录“实验结果.md”和“metrics.json”。<a href="https://rgbd-dataset.cs.washington.edu/dataset/">数据来源</a>，<a href="https://doi.org/10.1364/OL.34.003133">WRP原理</a>。</p></html>'''
    (out/'report.html').write_text(body,encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path);a=p.parse_args();run(a.directory)


if __name__=='__main__':main()
