"""Simulate a supplied photograph using the existing complete workflow exporter."""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import html
import json
import shutil
import sys

import numpy as np
from PIL import Image

from .core import Grid, propagate
from .diffraction import DiffractionConfig
from .holography import HolographyConfig
from .metrics import relative_error
from .objects import image_object
from .paths import OUTPUTS_DIR, PROJECT_ROOT
from .reporting import plt, image
from .workflow_catalog import Case, _compute, export_case


def _padding_check(result, config):
    """Keep the input sampling fixed and compare a larger periodic window."""
    count = config["n"] + max(32, config["n"] // 3)
    count += count % 2
    grid = Grid(count, config["dx"], config["wavelength"])
    source = image_object(grid, config["image_path"], config["image_samples"], "amplitude")
    larger = propagate(source, grid, config["z"], backend=config["backend"])
    start = (count - config["n"]) // 2
    matching = larger[start:start + config["n"], start:start + config["n"]]
    # Compare the actual image footprint, not a changing outer numerical boundary.
    original = np.abs(result.planes["object"].field) > 0
    rows, columns = np.nonzero(original)
    roi = np.ix_(np.arange(rows.min(), rows.max() + 1),
                 np.arange(columns.min(), columns.max() + 1))
    baseline = result.planes["propagated"].field
    return {"window_samples": [config["n"], count], "dx_m": grid.dx,
            "comparison_region": "original image rectangular footprint",
            "complex_relative_difference": relative_error(baseline[roi], matching[roi]),
            "intensity_relative_difference": relative_error(np.abs(baseline[roi])**2,
                                                             np.abs(matching[roi])**2)}


def _comparison(diffraction, holography, directory, source_name):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    source = diffraction.planes["object"]
    holography_x = holography.planes["object"].x
    far = diffraction.planes.get("far_field_zoom", diffraction.planes["far_field"])
    mask = np.abs(source.field) > 0
    rows, cols = np.nonzero(mask)
    margin = source.x[1] - source.x[0]
    bounds = (source.x[cols.min()] - margin, source.x[cols.max()] + margin,
              source.x[rows.min()] - margin, source.x[rows.max()] + margin)
    datasets = (
        (np.abs(source.field), source.x, "照片灰度 → 输入振幅；初相位为0", False, False),
        (diffraction.planes["propagated"].intensity, source.x,
         f"{diffraction.parameters['z']*100:g} cm 近场衍射强度（对数）", False, True),
        (far.intensity, far.x,
         "夫琅禾费远场强度（对数）", False, True),
        (holography.arrays["hologram_intensity_only"], holography_x,
         "模拟全息条纹 · 中心64×64样本", False, False),
        (np.abs(holography.planes["virtual_image"].field), holography_x,
         "全息重建振幅（有限带宽）", False, False),
        (np.ma.masked_where(np.abs(holography.planes["virtual_image"].field) <
                            np.abs(holography.planes["virtual_image"].field).max() * .02,
                            np.angle(holography.planes["virtual_image"].field)), holography_x,
         "全息重建相位（有效振幅区）", True, False),
    )
    for index, (axis, (values, x, title, phase, log)) in enumerate(zip(axes.flat, datasets)):
        image(axis, values, x, title, phase=phase, log=log)
        if index in (0, 4, 5):
            axis.set(xlim=(bounds[0]*1e6, bounds[1]*1e6),
                     ylim=(bounds[2]*1e6, bounds[3]*1e6))
        elif index == 1:
            limit = max(abs(v) for v in bounds)*1.5e6
            axis.set(xlim=(-limit, limit), ylim=(-limit, limit))
        elif index == 2:
            limit = 5 * diffraction.parameters["wavelength"] * diffraction.parameters["workflow_far_z_m"] / (
                diffraction.parameters["image_samples"] * diffraction.parameters["dx"]) * 1e6
            axis.set(xlim=(-limit, limit), ylim=(-limit, limit))
        elif index == 3:
            limit = 32 * diffraction.parameters["dx"] * 1e6
            axis.set(xlim=(-limit, limit), ylim=(-limit, limit))
        axis.invert_yaxis()
    fig.suptitle(source_name+" · 振幅透明片的衍射与离轴全息模拟", fontsize=18)
    fig.supxlabel("灰度作为振幅透过率；相位0是模型假设。空间图采用图像坐标（y向下）。全息误差相对带限输入评价。", fontsize=11)
    fig.savefig(directory/"结果对比.png", dpi=150)
    plt.close(fig)


def _project_path(path):
    try:
        return Path(path).relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def generate(image_path, directory, n=1024, dx=2.5e-6, wavelength=632.8e-9,
             image_samples=512, z=.02, bandwidth=44000., theta_deg=5.25):
    source = Path(image_path).resolve(strict=True)
    with Image.open(source) as original:
        original_size = original.size
        scaled = original.copy()
        scaled.thumbnail((image_samples, image_samples), Image.Resampling.LANCZOS)
        sampled_size = scaled.size
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    shared = dict(n=n, dx=dx, wavelength=wavelength, image_mode="amplitude",
                  image_samples=image_samples, z=z)
    diffraction_shared = {**shared, "n": n + n // 2}
    configs = [asdict(DiffractionConfig(**diffraction_shared, cdi_iterations=0)),
               asdict(HolographyConfig(**shared, bandwidth=bandwidth, theta_deg=theta_deg))]
    specs = [("01", "照片振幅衍射", "diffraction", "image_amplitude"),
             ("02", "照片离轴全息", "holography", "off_axis")]
    results, case_paths = [], []
    for config, (cid, title, family, variant) in zip(configs, specs):
        case_dir = directory/(cid+"_"+title)
        capture = case_dir/"01_输入图"/("原始输入图片"+source.suffix.lower())
        capture.parent.mkdir(parents=True, exist_ok=True)
        if capture.resolve() != source:
            shutil.copyfile(source, capture)
        config["image_path"] = str(capture)
        case = Case(cid, title, family, variant, config)
        print("计算："+title, flush=True)
        result = _compute(case, {})
        result.parameters["display_image_y_down"] = True
        result.notes += ["本例使用完整照片，经灰度转换与缩放后令透过率t=G、初始相位为0；图片支集外透过率为0。",
                         "灰度是指定的振幅透过率，未当作经过校准的强度；不从照片推断真实光学相位或三维结构。",
                         "空间图采用图像坐标，y向下，保持原照片的视觉方向；数值数组始终为[y,x]。"]
        if family == "diffraction":
            result.parameters["dense_far_field"] = True
            result.metrics["image_padding_check"] = _padding_check(result, config)
            result.notes.append("远场展示使用601×601样本的SciPy Zoom FFT精算局部频谱，不是图片插值；原始全范围FFT复场另存。")
        command = f'.\\.venv\\Scripts\\python.exe -m optics_sim.image_workflows --image "{_project_path(capture)}" --output "{_project_path(directory)}"'
        export_case(case, result, case_dir, reproduction_command=command)
        results.append(result)
        case_paths.append(case_dir.relative_to(directory).as_posix())
        print("完成："+str(case_dir/"index.html"), flush=True)
    diffraction, holography = results
    _comparison(diffraction, holography, directory, source.name)
    # Display-free hologram for reuse; floating-point measurements remain in NPZ.
    measured = holography.arrays["hologram_intensity_only"]
    Image.fromarray(np.rint(measured/measured.max()*65535).astype(np.uint16)).save(
        directory/case_paths[1]/"03_计算过程"/"全息强度_归一化16位.png")
    summary = {"source_name": source.name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
               "original_pixels": list(original_size), "sampled_pixels": list(sampled_size),
               "object_size_m": [v*dx for v in sampled_size],
               "diffraction_n": diffraction_shared["n"], "holography_n": n, "dx_m": dx,
               "wavelength_m": wavelength, "z_m": z, "input_model": "t=G, phase=0; outside image t=0",
               "diffraction_far_z_m": diffraction.parameters["workflow_far_z_m"],
               "far_field_display_method": "SciPy Zoom FFT, direct frequency evaluation, no image interpolation",
               "far_field_display_samples": 601,
               "diffraction_power_ratio": diffraction.metrics["power_ratio"],
               "diffraction_padding_check": diffraction.metrics["image_padding_check"],
               "holography_sampling": holography.metrics["sampling"],
               "holography_bandlimit_retained_power": holography.metrics["optical_pupil_retained_power"],
               "holography_complex_relative_error": holography.metrics["virtual_complex_relative_error"],
               "holography_intensity_similarity": holography.metrics["virtual_intensity_similarity"],
               "cases": case_paths}
    (directory/"计算摘要.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    links = "".join(f'<li><a href="{path}/index.html">{html.escape(path)}</a></li>' for path in case_paths)
    page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(source.name)+' · 衍射与全息</title><style>body{max-width:1250px;margin:32px auto;padding:0 24px;font:17px/1.8 system-ui;color:#203047;background:#f4f7fa}img{max-width:100%}a{color:#17639a}pre{overflow:auto;background:white;padding:20px}</style>'
    page += '<h1>'+html.escape(source.name)+' · 衍射与离轴全息模拟</h1><p>完整照片转换为灰度振幅透过率，初始相位设为0。各模拟单独保存完整流程与复场数据。</p><ol>'+links+'</ol>'
    page += '<p>输入大小：'+str(sampled_size[0])+'×'+str(sampled_size[1])+'样本；λ='+f'{wavelength*1e9:g} nm；dx={dx*1e6:g} μm；传播距离={z:g} m。远场独立采用距离 '+f'{summary["diffraction_far_z_m"]:.6g} m。</p>'
    page += '<p>远场中心区域采用601×601样本的SciPy Zoom FFT直接求值；全范围原始FFT复场与局部精算复场均保存在NPZ中。</p>'
    page += '<p>全息重建相对带限输入评价；有限带宽会平滑照片细节。<a href="'+case_paths[1]+'/03_计算过程/全息强度_归一化16位.png">全息强度16位PNG</a>经过峰值归一化，原始浮点强度保存在全息案例的06_数据/data.npz中。</p><img src="结果对比.png"><details><summary>计算摘要</summary><pre>'+html.escape(json.dumps(summary, ensure_ascii=False, indent=2))+'</pre></details>'
    (directory/"index.html").write_text(page, encoding="utf-8")
    captured_source = directory/case_paths[0]/"01_输入图"/("原始输入图片"+source.suffix.lower())
    reproducible_source = _project_path(captured_source)
    reproducible_output = _project_path(directory)
    (directory/"流程说明.md").write_text(
        f"# {source.name}：衍射与离轴全息\n\n先打开index.html，两个案例分别保存全流程。\n\n"
        "照片灰度指定为振幅透过率，初始相位为0。全息误差以有限带宽后的输入为真值；16位PNG经过峰值归一化，原始浮点强度在NPZ中。\n\n"
        f"衍射网格{diffraction_shared['n']}×{diffraction_shared['n']}，全息网格{n}×{n}，两者输入图片采样和物理尺寸相同。\n\n"
        "在项目根目录运行下列命令可重新计算；--image后应填写项目内捕获的原图路径：\n\n"
        f"```powershell\n.\\.venv\\Scripts\\python.exe -m optics_sim.image_workflows --image \"{reproducible_source}\" --output \"{reproducible_output}\"\n```\n",
        encoding="utf-8")
    print("汇总入口："+str(directory/"index.html"), flush=True)
    return summary


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    generate(args.image, args.output or OUTPUTS_DIR/"image_simulations"/args.image.stem)


if __name__ == "__main__":
    main()
