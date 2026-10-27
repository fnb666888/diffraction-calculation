"""Compare Fourier-plane masks on a user photograph; retain physical image fields."""
from pathlib import Path
import argparse
import html
import json
import shutil
import numpy as np
from .abbe import AbbeConfig,FILTER_LABELS,invert,run
from .paths import OUTPUTS_DIR
from .reporting import plt
from .workflow_catalog import _portable_parameters


def generate(image_path,directory,n=512,dx=5e-6,image_samples=256,wavelength=632.8e-9):
    image_path=Path(image_path).resolve(strict=True)
    directory=Path(directory).resolve()
    directory.mkdir(parents=True,exist_ok=True)
    captured=directory/("原始输入图片"+image_path.suffix.lower())
    if captured!=image_path:
        shutil.copyfile(image_path,captured)
    kinds=["none","lowpass","highpass","bandpass","slit","vortex"]
    fig,axes=plt.subplots(2,3,figsize=(15,9),constrained_layout=True)
    inventory=[]
    for axis,kind in zip(axes.flat,kinds):
        c=AbbeConfig(n=n,dx=dx,wavelength=wavelength,image_path=str(captured),filter_kind=kind,
                     cutoff=12000,inner_cutoff=4000,image_samples=image_samples)
        r=run(c)
        # The supplementary comparison is registered to the source direction;
        # the saved physical image field remains inverted, with all SI factors.
        shown=np.abs(invert(r.planes['image'].field))**2
        source=np.abs(r.planes['object'].field)>0
        rows,cols=np.nonzero(source)
        image=shown[rows.min():rows.max()+1,cols.min():cols.max()+1]
        axis.imshow(image/max(float(image.max()),1e-30),cmap="gray",vmin=0,vmax=1,origin="upper")
        axis.set_title(FILTER_LABELS[kind]+f" · 功率比例 {r.metrics['power_fraction_of_input']:.4f}",fontsize=13)
        axis.axis('off')
        r.parameters=_portable_parameters(r.parameters)
        r.save(directory/'06_数据'/kind)
        inventory.append({'filter_kind':kind,'label':FILTER_LABELS[kind],
                          'power_fraction_of_input':r.metrics['power_fraction_of_input'],
                          'spectral_power_error':r.metrics['spectral_power_balance_relative_error']})
    fig.suptitle(image_path.name+" · 阿贝成像频谱滤波对比",fontsize=20)
    fig.supxlabel("仅此对比图把倒像转回物方方向，并按各图峰值归一化以比较形态；原始像面复场保留物理倒像和绝对标度。",fontsize=11)
    fig.savefig(directory/'滤波效果对比.png',dpi=150)
    plt.close(fig)
    fig.clear()
    (directory/'计算摘要.json').write_text(json.dumps(inventory,ensure_ascii=False,indent=2),encoding='utf-8')
    links=''.join('<li>'+html.escape(item['label'])+f'：功率比例{item["power_fraction_of_input"]:.6f} · <a href="06_数据/{item["filter_kind"]}/data.npz">原始复场</a></li>' for item in inventory)
    page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>阿贝成像图片滤波对比</title><style>body{max-width:1250px;margin:32px auto;padding:0 22px;font:17px/1.8 system-ui;color:#203047}img{max-width:100%}a{color:#17639a}</style><h1>阿贝成像 · '+html.escape(image_path.name)+'</h1>'
    page+=f'<p>相干照明；照片灰度作为振幅透过率，初始相位为0。f1=f2=100 mm，NA=0.03，λ={wavelength*1e9:g} nm，dx={dx*1e6:g} μm，{n}×{n}网格，照片最长边{image_samples}样本。</p>'
    page+='<p>本对比图统一转回物方方向，各图强度按自身峰值归一化，仅比较形态；原始4f倒像、像面坐标及真实功率保存在各NPZ和metrics.json中。</p><img src="滤波效果对比.png"><ul>'+links+'</ul>'
    from .paths import WORKFLOWS_DIR
    import os
    catalog=os.path.relpath(WORKFLOWS_DIR/'index.html',directory).replace('\\','/')
    page+='<p><a href="'+html.escape(catalog,quote=True)+'">孔径、光栅与照片结果目录</a></p>'
    (directory/'index.html').write_text(page,encoding='utf-8')
    print(directory/'index.html',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    generate(args.image,args.output or OUTPUTS_DIR/'abbe_comparison'/args.image.stem)


if __name__=='__main__':
    main()
