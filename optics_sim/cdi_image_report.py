"""Export a self-contained, linear-intensity report from saved image-CDI data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export_report(directory):
    directory=Path(directory)
    data=directory/'06_数据'
    metrics=json.loads((data/'metrics.json').read_text(encoding='utf-8'))
    config=json.loads((data/'配置.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':12,'savefig.dpi':160})
    with np.load(data/'data.npz') as d:
        x=d['object_x_m'];dx=float(x[1]-x[0])
        limit=(max(metrics['support_samples_yx'])+14)*dx/2
        spatial=np.abs(x)<=limit
        fields=[d[k] for k in ['truth_field','er_aligned_field','recovered_aligned_field']]
        ref=max(float(np.abs(u).max()**2) for u in fields)

        def field(ax,u,title,ref=None):
            value=np.abs(u[np.ix_(spatial,spatial)])**2
            value=value/max(float(value.max() if ref is None else ref),1e-30)
            xx=x[spatial]*1e6;half=dx*1e6/2
            im=ax.imshow(value,cmap='gray',vmin=0,vmax=1,origin='upper',interpolation='nearest',
                extent=[xx[0]-half,xx[-1]+half,xx[-1]+half,xx[0]-half])
            ax.set(title=title,xlabel='x / μm',ylabel='y / μm')
            plt.colorbar(im,ax=ax,fraction=.046,pad=.035,label='归一化光强')

        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        field(axes[0],d['truth_field'],'原图光强',ref)
        f=d['frequency_per_m']
        freq_limit=min(.8*np.max(np.abs(f)),8/(max(metrics['image_samples_yx'])*dx))
        select=np.abs(f)<=freq_limit
        intensity=d['measurement_intensity'];values=intensity[np.ix_(select,select)]/intensity.max()
        xx=d['detector_x_m'][select]*1e3;half=(xx[1]-xx[0])/2
        im=axes[1].imshow(values,cmap='gray',vmin=0,vmax=1,origin='upper',interpolation='nearest',
            extent=[xx[0]-half,xx[-1]+half,xx[-1]+half,xx[0]-half])
        axes[1].set(title='衍射光强（中心区域）',xlabel='X / mm',ylabel='Y / mm')
        fig.colorbar(im,ax=axes[1],fraction=.046,pad=.035,label='ID / max(ID)')
        xx=x*1e6
        axes[2].imshow(d['support'],cmap='gray',vmin=0,vmax=1,origin='upper',extent=[xx[0],xx[-1],xx[-1],xx[0]])
        axes[2].set(title='反演矩形支集',xlabel='x / μm',ylabel='y / μm')
        fig.savefig(directory/'cdi-measurement.png',facecolor='white');plt.close(fig)

        fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
        for ax,u,label in zip(axes,fields,['原图光强','ER恢复光强','HIO + ER恢复光强']):
            field(ax,u,label,ref)
        fig.savefig(directory/'cdi-reconstruction.png',facecolor='white');plt.close(fig)

        fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
        for key,label,color in [('er_residual',f'ER，seed={config["seeds"][0]}','#C47B30'),
                                ('hio_residual',f'HIO + ER，seed={metrics["selected_seed"]}','#174994')]:
            values=np.minimum.accumulate(d[key])
            ax.semilogy(np.arange(1,len(values)+1),values,color=color,label=label)
        switch=config['iterations']-min(50,config['iterations']//4)
        if switch>0:ax.axvline(switch,color='#777777',linestyle='--',linewidth=.8)
        ax.set(title='与测量数据的一致性',xlabel='迭代次数',ylabel='截至该步的最小傅里叶模相对残差')
        ax.legend();ax.grid(alpha=.17)
        fig.savefig(directory/'cdi-convergence.png',facecolor='white');plt.close(fig)

    sections=[('cdi-measurement','图片、模拟测量与支集'),('cdi-reconstruction','图片反演结果'),('cdi-convergence','迭代收敛')]
    page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>真实照片CDI反演</title><style>body{max-width:1400px;margin:36px auto;padding:0 24px;font:18px/1.8 system-ui;color:#203047}img{max-width:100%}a{color:#174994}</style>
<h1>真实照片的CDI反演</h1><p>反演输入为完整傅里叶光强和已知矩形支集，原图仅用于正向模拟与最后评价。物体采用实非负振幅、零相位模型。</p>
<p>光强使用线性灰度，不对弱条纹加增益。测量图只裁切显示范围，反演使用全部测量数据。恢复图片在迭代后消除平移、共轭反演和全局相位歧义。</p>'''
    for key,title in sections:page+=f'<h2>{title}</h2><img src="{key}.png">'
    page+='<p>误差曲线的对数纵轴仅用于显示收敛。当前为理想无噪声数值例，不能保证任意复物体的唯一恢复。</p><p><a href="06_数据/data.npz">原始数据</a>　<a href="06_数据/metrics.json">误差指标</a>　<a href="06_数据/配置.json">配置</a></p>'
    (directory/'index.html').write_text(page,encoding='utf-8')
