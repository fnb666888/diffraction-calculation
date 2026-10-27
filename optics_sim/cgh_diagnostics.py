"""Controlled numerical analysis of the existing two-plane CGH demonstration.

Keeps original data, synthesis and phase command unchanged. Comparisons use a
known design normalization, never a fitted gain or truth-based registration.
"""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from .core import Grid,propagate
from .holography import bandlimit
from .metrics import relative_error
from . import phase_comparison_figures  # project CJK fonts

ROOT=Path(__file__).resolve().parents[1]


def expand(a):
    return np.repeat(np.repeat(a,2,axis=0),2,axis=1)


def synthesis(raw_layers,cutoff,macro,fine,distances):
    layers=[bandlimit(u,macro,cutoff)[0] for u in raw_layers]
    hs=[propagate(u,macro,-float(d),method='fresnel',backend='scipy') for u,d in zip(layers,distances)]
    h=sum(hs);scale=float(np.abs(h).max());h=h/scale
    a=np.clip(np.abs(h),0,1);phi=np.angle(h);delta=np.arccos(a)
    parity=np.indices((fine.n,fine.n)).sum(axis=0)%2==0
    phase=np.where(parity,expand(phi+delta),expand(phi-delta))
    command=np.rint(np.mod(phase,2*np.pi)*256/(2*np.pi)).astype(np.int64)%256
    phase_q=2*np.pi*command/256
    exact,_=bandlimit(np.exp(1j*phase),fine,cutoff)
    quant,_=bandlimit(np.exp(1j*phase_q),fine,cutoff)
    ideal,_=bandlimit(expand(h),fine,cutoff)
    components=[bandlimit(expand(v),fine,cutoff)[0] for v in hs]
    outputs=[propagate(quant,fine,float(d),method='fresnel',backend='scipy')*scale for d in distances]
    ideal_outputs=[propagate(ideal,fine,float(d),method='fresnel',backend='scipy')*scale for d in distances]
    single=[propagate(c,fine,float(d),method='fresnel',backend='scipy') for c,d in zip(components,distances)]
    other=[propagate(components[1-j],fine,float(d),method='fresnel',backend='scipy') for j,d in enumerate(distances)]
    return dict(layers=layers,h=h,scale=scale,command=command,ideal_outputs=ideal_outputs,
      outputs=outputs,single=single,other=other,
      encoding_field_error=relative_error(quant,ideal),
      quantization_only_field_error=relative_error(quant,exact))


def run():
    source=ROOT/'outputs/cgh_slm/two_planes';out=source/'diagnostics';out.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'calibration.json').read_text(encoding='utf-8'))
    macro=Grid(config['macro_n'],config['macro_pitch_m'],config['wavelength_m'])
    fine=Grid(config['slm_n'],config['slm_pitch_m'],config['wavelength_m'])
    distances=config['plane_distances_m'];B=config['fourier_pupil_cutoff_per_m']
    rgb=np.asarray(Image.open(ROOT/'data/cgh_slm/two_planes/input-color.png').convert('RGB'))
    depth=np.asarray(Image.open(ROOT/'data/cgh_slm/two_planes/input-depth-levels.png').convert('L'))
    gray=rgb.astype(float)@np.array([.2126,.7152,.0722])/255
    r0=(macro.n-depth.shape[0])//2;c0=(macro.n-depth.shape[1])//2
    values=sorted(x for x in np.unique(depth) if x>0)
    raw=[];masks=[];rois=[]
    for value in values:
        a=np.zeros((macro.n,macro.n),complex);m=np.zeros_like(a,dtype=bool)
        a[r0:r0+depth.shape[0],c0:c0+depth.shape[1]]=np.sqrt(gray)*(depth==value)
        m[r0:r0+depth.shape[0],c0:c0+depth.shape[1]]=depth==value
        raw.append(a);masks.append(expand(m))
        rr,cc=np.where(expand(m));margin=18
        rois.append((max(0,int(rr.min())-margin),min(fine.n,int(rr.max())+margin+1),
          max(0,int(cc.min())-margin),min(fine.n,int(cc.max())+margin+1)))
    calc=synthesis(raw,B,macro,fine,distances)
    saved=np.load(source/'fields.npz')
    previous_command=np.asarray(Image.open(source/'slm-phase-8bit.png'))
    consistency={'recomputed_phase_command_identical':bool(np.array_equal(previous_command,calc['command'])),
      'recomputed_to_saved_complex64_field_errors':[relative_error(u,saved['reconstructed_fields'][j]*calc['scale']) for j,u in enumerate(calc['outputs'])]}
    raw_i=[np.abs(expand(a))**2 for a in raw]
    stages=[raw_i,[np.abs(expand(a))**2 for a in calc['layers']],
      [np.abs(a)**2 for a in calc['single']],
      [np.abs(a)**2 for a in calc['ideal_outputs']],
      [np.abs(a)**2 for a in calc['outputs']]]
    names=['原始灰度目标','仅输入带限','单层理想复调制','双层理想复调制','双层相位SLM']
    objects=[]
    for j,label in enumerate(['魔方','足球']):
        m=masks[j];own=calc['single'][j];other=calc['other'][j]
        own_i=np.abs(own)**2;other_i=np.abs(other)**2;cross=2*np.real(own*np.conj(other))
        rr0,rr1,cc0,cc1=rois[j];roi_mask=np.zeros(m.shape,bool);roi_mask[rr0:rr1,cc0:cc1]=True
        objects.append({'object':label,'distance_mm':distances[j]*1000,'roi_rc':rois[j],
          'foreground_intensity_relative_error_to_raw_by_stage':{name:relative_error(stage[j][m],raw_i[j][m]) for name,stage in zip(names,stages)},
          'foreground_intensity_change_by_transition':{names[k-1]+' -> '+names[k]:relative_error(stages[k][j][m],stages[k-1][j][m]) for k in range(1,5)},
          'other_layer_power_relative_to_own_in_foreground':float(other_i[m].sum()/own_i[m].sum()),
          'coherent_cross_term_rms_relative_to_own_intensity':float(np.linalg.norm(cross[m])/np.linalg.norm(own_i[m])),
          'own_layer_vs_other_plus_interference_intensity_change':relative_error(np.abs(own+other)[m]**2,own_i[m]),
          'single_vs_bandlimited_complex_error_in_foreground':relative_error(own[m],expand(calc['layers'][j])[m]),
          'quantized_vs_ideal_intensity_error_in_foreground':relative_error(stages[4][j][m],stages[3][j][m]),
          'linear_to_gamma03_mean_gray_in_foreground':None})
    iref=max(float(s[j][slice(rois[j][0],rois[j][1]),slice(rois[j][2],rois[j][3])].max()) for s in stages for j in range(2))
    for j in range(2):
        v=stages[4][j][masks[j]]/iref
        objects[j]['linear_to_gamma03_mean_gray_in_foreground']=[float(v.mean()),float((v**.30).mean())]
    def figure(gamma):
        fig,axs=plt.subplots(2,5,figsize=(18,7.5),layout='constrained')
        for j in range(2):
            r1,r2,c1,c2=rois[j]
            for k,stage in enumerate(stages):
                ax=axs[j,k];ax.imshow(stage[j][r1:r2,c1:c2]/iref,cmap='gray',norm=PowerNorm(gamma,vmin=0,vmax=1),interpolation='nearest')
                ax.set_title(names[k],fontsize=16);ax.set_xticks([]);ax.set_yticks([])
                if k==0:ax.set_ylabel(['魔方 40 mm','足球 60 mm'][j],fontsize=16)
                e=objects[j]['foreground_intensity_relative_error_to_raw_by_stage'][names[k]]
                ax.set_xlabel('形状内光强误差 '+f'{100*e:.1f}%',fontsize=13)
        fig.suptitle(f'逐步拆分当前计算全息流程（统一光强标度，显示γ={gamma:.2f}）',fontsize=21)
        fig.savefig(out/f'stages-gamma-{gamma:.2f}.png',dpi=180,bbox_inches='tight');plt.close(fig)
    figure(.30);figure(1.)
    sweep=[]
    for cutoff in [12000.,18000.,24000.,30000.]:
        c=calc if cutoff==B else synthesis(raw,cutoff,macro,fine,distances)
        sweep.append({'cutoff_per_mm':cutoff/1000,'encoding_field_error':c['encoding_field_error'],
          'foreground_intensity_error_to_raw':{'ideal':[relative_error(np.abs(u[m])**2,raw_i[j][m]) for j,(u,m) in enumerate(zip(c['ideal_outputs'],masks))],
            'slm':[relative_error(np.abs(u[m])**2,raw_i[j][m]) for j,(u,m) in enumerate(zip(c['outputs'],masks))]},
          'input_bandlimit_intensity_error_to_raw':[relative_error(np.abs(expand(a)[m])**2,raw_i[j][m]) for j,(a,m) in enumerate(zip(c['layers'],masks))]})
    # Propagation-kernel comparison isolates Fresnel approximation at the same H
    # and optical pupil, without redesigning or adjusting a phase map.
    angular=[]
    for j,d in enumerate(distances):
        selected=saved['optical_selected_wave']*calc['scale']
        u=propagate(selected,fine,float(d),method='angular',backend='scipy')
        angular.append(relative_error(np.abs(u[masks[j]])**2,stages[4][j][masks[j]]))
    # Hold the existing finite SLM command/aperture fixed and zero-pad its exit
    # field, so the new result approximates non-periodic optical convolution.
    # This isolates periodic-window replay effects; it is not a full synthesis
    # grid convergence study and does not redesign or enlarge the active SLM.
    padded=Grid(2*fine.n,fine.dx,fine.wavelength);start=fine.n//2
    exit_pad=np.zeros((padded.n,padded.n),complex)
    exit_pad[start:start+fine.n,start:start+fine.n]=saved['slm_exit_wave']
    filtered_pad,_=bandlimit(exit_pad,padded,B)
    padding_change=[]
    for j,d in enumerate(distances):
        u=propagate(filtered_pad,padded,float(d),method='fresnel',backend='scipy')
        center=u[start:start+fine.n,start:start+fine.n]*calc['scale']
        padding_change.append(relative_error(np.abs(center[masks[j]])**2,stages[4][j][masks[j]]))
    result={'source_fields_sha256':hashlib.sha256((source/'fields.npz').read_bytes()).hexdigest(),
      'consistency':consistency,'metric_definition':'L2 intensity error inside the prescribed object silhouette, known design gain restored; no fitted gain/registration; stage errors are not additive',
      'common_display_intensity_reference':iref,'objects':objects,'bandwidth_sweep':sweep,
      'quantization_only_selected_field_error':calc['quantization_only_field_error'],
      'same_slm_field_fresnel_to_angular_foreground_intensity_change':angular,
      'fixed_slm_aperture_zero_padded_replay_foreground_intensity_change':padding_change,
      'padding_scope':'fixed original 4.096 mm active SLM, exit field zero outside aperture; 2x window, same pitch and optical cutoff; replay only, not synthesis convergence',
      'display_transfer_examples':{'one_percent_intensity_gamma03':.01**.3,'ten_percent_intensity_gamma03':.1**.3}}
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'stage-intensities.npz',stage_names=np.asarray(names),
      **{f'object{j}_stage{k}':s[j][rois[j][0]:rois[j][1],rois[j][2]:rois[j][3]].astype(np.float32) for k,s in enumerate(stages) for j in range(2)})
    html='<!doctype html><meta charset="utf-8"><title>计算全息原因分析</title><style>body{font:18px/1.7 Microsoft YaHei,sans-serif;max-width:1500px;margin:24px auto}img{width:100%}</style><h1>计算全息逐阶段原因分析</h1><p>只进行对照分析，原始SLM相位命令与PPT未修改。各图共用光强标度，误差按对应物体轮廓内的原始灰度光强计算，未拟合增益。不同阶段误差不能相加。</p><h2>当前γ=0.30显示</h2><img src="stages-gamma-0.30.png"><h2>线性显示</h2><img src="stages-gamma-1.00.png"><p><a href="analysis.json">完整指标与带宽扫描</a></p>'
    (out/'report.html').write_text(html,encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':run()
