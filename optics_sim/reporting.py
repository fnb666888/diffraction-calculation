"""Shared scientific figures and a local, self-contained experiment report."""
from pathlib import Path
import html
import json
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 110,
                     "savefig.dpi": 160, "font.size": 10})


def image(ax, data, x, title, phase=False, log=False, frequency=False):
    scale = 1e-3 if frequency else 1e6
    unit = r"频率 ($\mathrm{mm}^{-1}$)" if frequency else "位置 (μm)"
    d = (x[1]-x[0])/2
    extent = [(x[0]-d)*scale, (x[-1]+d)*scale]*2
    values = np.asanyarray(data)
    kwargs = {"vmin": -np.pi, "vmax": np.pi} if phase else {}
    if log:
        values = np.log10(values/max(float(values.max()), 1e-30)+1e-8)
        kwargs = {"vmin": -6, "vmax": 0}
    im = ax.imshow(values, origin="lower", extent=extent, cmap="twilight" if phase else "magma", **kwargs)
    ax.set(title=title, xlabel="x "+unit, ylabel="y "+unit)
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04, label="rad" if phase else (r"$\log_{10}(I/I_\mathrm{max})$" if log else "a.u."))


def triptych(plane, title):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), constrained_layout=True)
    image(axes[0], plane.intensity, plane.x, "强度", log=True)
    image(axes[1], np.abs(plane.field), plane.x, "振幅")
    phase = np.ma.masked_where(np.abs(plane.field)<np.abs(plane.field).max()*1e-5, np.angle(plane.field))
    image(axes[2], phase, plane.x, "相位（低振幅区域屏蔽）", phase=True)
    fig.suptitle(title)
    return fig


def _save(fig, directory, name, pictures):
    fig.savefig(directory/(name+".png"))
    plt.close(fig)
    pictures.append(name+".png")


def diffraction_plots(r, directory, pictures):
    _save(triptych(r.planes["propagated"], "衍射传播 · "+r.planes["propagated"].method), directory, "propagation", pictures)
    _save(triptych(r.planes["far_field"], "夫琅禾费远场 · 强度 / 振幅 / 相位"), directory, "far_field", pictures)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    image(ax[0, 0], r.planes["object"].intensity, r.planes["object"].x, "输入物体")
    image(ax[0, 1], r.planes["far_field"].intensity, r.planes["far_field"].x, "夫琅禾费远场", log=True)
    a = r.arrays
    ax[1, 0].plot(a["airy_x_m"]*1e6, a["airy_numeric"], label="FFT 圆孔")
    ax[1, 0].plot(a["airy_x_m"]*1e6, a["airy_analytic"], "--", label="Airy 解析解")
    limit = 5*r.parameters["wavelength"]*r.parameters["z"]/r.parameters["width"]*1e6
    ax[1, 0].set(xlim=(-limit, limit), title="圆孔远场中心截面", xlabel="x (μm)", ylabel="I/Imax")
    ax[1, 0].legend()
    ax[1, 1].plot(a["square_x_m"]*1e6, a["square_numeric"], label="单 FFT 方孔")
    ax[1, 1].plot(a["square_x_m"]*1e6, a["square_analytic"], "--", label="Fresnel 积分解析解")
    ax[1, 1].set(xlim=(-limit, limit), title="方孔菲涅耳中心截面", xlabel="x (μm)", ylabel="强度")
    ax[1, 1].legend()
    _save(fig, directory, "diffraction_overview", pictures)
    if "cdi_truth" in a:
        fig, ax = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
        for axis, key, title in zip(ax.flat, ("cdi_truth", "cdi_intensity_only", "cdi_er", "cdi_hio_aligned"),
                                    ("CDI 真值", "仅保留的远场强度（对数）", "ER 恢复", "HIO 恢复（歧义对齐后）")):
            values = a[key] if key=="cdi_intensity_only" else np.abs(a[key])**2
            if key=="cdi_intensity_only":
                values = np.log10(values+1e-8)
            axis.imshow(values, origin="lower", cmap="magma")
            axis.set_title(title)
        ax[1, 1].semilogy(a["cdi_er_residual"], label="ER")
        ax[1, 1].semilogy(a["cdi_hio_residual"], label="HIO")
        ax[1, 1].set(title="傅里叶振幅相对残差", xlabel="迭代次数")
        ax[1, 1].legend()
        ax[1, 2].imshow(np.abs(a["cdi_hio_aligned"]-a["cdi_truth"]), origin="lower", cmap="magma")
        ax[1, 2].set_title("复场误差绝对值")
        _save(fig, directory, "cdi", pictures)


def talbot_plots(r, directory, pictures):
    a, zt = r.arrays, r.metrics["talbot_distance_m"]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    x, z = a["carpet_x_m"], a["carpet_z_m"]
    central = np.abs(x)<=r.parameters["period"]*4
    dx = (x[1]-x[0])/2
    carpet = ax[0, 0].imshow(a["carpet_intensity"][:, central], origin="lower", aspect="auto", cmap="magma",
                             extent=[(x[central][0]-dx)*1e6, (x[central][-1]+dx)*1e6, z[0]/zt, z[-1]/zt])
    ax[0, 0].set(title="泰伯 x–z 强度演化", xlabel="x (μm)", ylabel="z / zT")
    plt.colorbar(carpet, ax=ax[0, 0], label="强度")
    for name, label in (("object", "输入"), ("half", "zT/2：平移像"), ("full", "zT：原像")):
        line = r.planes[name].intensity[len(x)//2]
        ax[0, 1].plot(x[central]*1e6, line[central], label=label)
    ax[0, 1].set(title="自像中心截面", xlabel="x (μm)", ylabel="强度")
    ax[0, 1].legend()
    ax[1, 0].stem(a["orders"], a["order_power_numeric"], linefmt="C0-", markerfmt="C0o", basefmt=" ", label="FFT 数值")
    if "order_power_analytic" in a:
        ax[1, 0].plot(a["orders"], a["order_power_analytic"], "C1x", label="连续光栅解析值")
    ax[1, 0].set(title="衍射级次及包络", xlabel="级次 m", ylabel="级次功率")
    ax[1, 0].legend()
    for name, label in (("full", "相干"), ("partial_coherence_full", "部分相干模态平均")):
        line = r.planes[name].intensity[len(x)//2]
        ax[1, 1].plot(x[central]*1e6, line[central], label=label)
    ax[1, 1].set(title="相干性对自像衬度的影响", xlabel="x (μm)", ylabel="强度")
    ax[1, 1].legend()
    _save(fig, directory, "talbot_overview", pictures)
    _save(triptych(r.planes["half"], "半泰伯距离 · 横移半个周期"), directory, "half_talbot", pictures)
    _save(triptych(r.planes["full"], "整泰伯距离 · 周期自成像"), directory, "full_talbot", pictures)
    fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, name, title in zip(ax[0], ("eighth", "quarter", "curved_quarter"), ("zT/8 分数泰伯", "zT/4 分数泰伯", "曲率入射 · zT/4")):
        image(axis, r.planes[name].intensity, x, title)
    image(ax[1, 0], a["tli_flat_intensity"], x, "双光栅 · 平面波")
    image(ax[1, 1], a["tli_curved_intensity"], x, "双光栅 · 曲率波前")
    ax[1, 2].plot(x*1e6, a["tli_flat_envelope"], label="平面波")
    ax[1, 2].plot(x*1e6, a["tli_curved_envelope"], label="曲率波前")
    ax[1, 2].set(title="莫尔包络（光栅周期略失配）", xlabel="x (μm)", ylabel="平滑强度")
    ax[1, 2].legend()
    _save(fig, directory, "fractional_and_tli", pictures)


def holography_plots(r, directory, pictures):
    a, x = r.arrays, r.planes["object"].x
    fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    image(ax[0, 0], r.planes["object"].intensity, x, "带限物体真值")
    image(ax[0, 1], a["hologram_intensity_only"], x, "离轴全息图 · 仅强度记录")
    image(ax[0, 2], a["hologram_spectrum"], a["frequency_per_m"], "0 级 / ±1 级频谱", log=True, frequency=True)
    image(ax[1, 0], r.planes["virtual_image"].intensity, x, "数字一级滤波 · 虚像")
    for axis, key, title in ((ax[1, 1], "virtual_image", "恢复物体相位（有效振幅区）"),
                             (ax[1, 2], "twin_image", "孪生实像 · 相位共轭")):
        u = r.planes[key].field
        phase = np.ma.masked_where(np.abs(u)<np.abs(u).max()*.02, np.angle(u))
        image(axis, phase, x, title, phase=True)
    support = np.abs(r.planes["object"].field)>np.abs(r.planes["object"].field).max()*.02
    rows, columns = np.nonzero(support)
    limit = max(np.max(np.abs(x[rows])), np.max(np.abs(x[columns])))*1.25e6
    for axis in (ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[1, 2]):
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit))
    _save(fig, directory, "holography_overview", pictures)
    _save(triptych(r.planes["virtual_image"], "离轴全息 · 复振幅恢复"), directory, "reconstruction", pictures)
    fig, ax = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, key, title in zip(ax.flat[:4], ("object", "aberrated_image", "common_path_image", "reverse_image"),
                               ("物体真值", "薄相位屏 · 畸变基线", "共光路 · 畸变相消", "逆参考波 · 共轭补偿")):
        image(axis, r.planes[key].intensity, x, title)
    ax[1, 1].semilogy(a["angle_scan_deg"], np.maximum(a["angle_scan_errors"], 1e-16), "o-")
    ax[1, 1].set(title="参考角扫描 · 频谱分离", xlabel="参考角 (°)", ylabel="记录面复场相对误差")
    ax[1, 2].semilogy(a["pixel_scan_n"], np.maximum(a["pixel_scan_errors"], 1e-16), "o-")
    ax[1, 2].set(title="同物理窗口 · 像素数扫描", xlabel="每边像素 n", ylabel="记录面复场相对误差")
    _save(fig, directory, "aberration_and_sampling", pictures)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4), constrained_layout=True)
    for axis, key, title in zip(ax, ("raw_virtual", "raw_real", "phase_shift_image"),
                               ("原参考照明 · 未分级虚像路径", "共轭参考照明 · 未分级实像路径", "四步相移 · 复场重建")):
        image(axis, r.planes[key].intensity, x, title, log=True)
    _save(fig, directory, "optical_and_phase_shift", pictures)


def export_result(result, directory):
    directory = result.save(directory)
    from .input_visualization import export_input_visuals
    input_pictures = export_input_visuals(result, directory)
    pictures = []
    {"diffraction": diffraction_plots, "talbot": talbot_plots, "holography": holography_plots}[result.module](result, directory, pictures)
    title = {"diffraction": "衍射计算", "talbot": "泰伯效应", "holography": "离轴全息"}[result.module]
    metrics = html.escape(json.dumps(result.metrics, ensure_ascii=False, indent=2))
    params = html.escape(json.dumps(result.parameters, ensure_ascii=False, indent=2))
    notes = "".join("<li>"+html.escape(note)+"</li>" for note in result.notes)
    def figures(names):
        return "".join(f'<figure><a href="{name}"><img src="{name}" alt="{name}"></a></figure>' for name in names)
    ordered = [input_pictures[0], input_pictures[4], *input_pictures[1:4], *pictures]
    (directory/"figures.json").write_text(json.dumps(ordered, ensure_ascii=False, indent=2), encoding="utf-8")
    stages = '<h2>1. 物理输入图与具体函数</h2>'+figures([input_pictures[0], input_pictures[4]])
    stages += '<h2>2. 输入复振幅的振幅与相位</h2>'+figures([input_pictures[3]])
    stages += '<p>单独查看：<a href="02_输入复振幅/振幅.png">振幅图</a> · <a href="02_输入复振幅/相位.png">相位图</a> · <a href="02_输入复振幅/输入复场.npz">输入复场数据</a></p>'
    stages += '<h2>3. 传播、成像与验证结果</h2>'+figures(pictures)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>{title} · 仿真报告</title><style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1240px;margin:32px auto;padding:0 24px;color:#203047;background:#f5f7fb}}h1{{color:#124778}}img{{width:100%;border-radius:10px}}figure{{margin:24px 0}}pre{{overflow:auto;background:white;padding:20px;border:1px solid #dde4ed;border-radius:8px;font-size:13px}}a{{color:#17639a}}details{{margin:18px 0}}</style>
<h1>{title} · 计算机模拟</h1><p>统一单位：m · 复振幅保留 · 默认传播后端 LightPipes 2.1.5</p>
<p><a href="data.npz">原始复场与强度 NPZ</a> · <a href="metrics.json">参数与定量指标 JSON</a></p>
{stages}<details><summary>模型约定与说明</summary><ul>{notes}</ul></details><details open><summary>定量指标</summary><pre>{metrics}</pre></details>
<details><summary>运行参数（SI）</summary><pre>{params}</pre></details></html>'''
    (directory/"report.html").write_text(page, encoding="utf-8")
    return directory/"report.html"


def export_index(directory, reports):
    directory = Path(directory)
    # Re-running one module in an existing experiment folder must keep the other
    # completed reports reachable from the same index.
    labels = {"diffraction": "衍射计算", "talbot": "泰伯效应", "holography": "离轴全息"}
    present = dict(reports)
    for name, label in labels.items():
        if (directory/name/"report.html").is_file():
            present[name] = label
    reports = [(name, present[name]) for name in labels if name in present]
    links = "".join(f'<li><a href="{name}/report.html">{label}</a></li>' for name, label in reports)
    (directory/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>光学模拟实验</title><style>body{font:18px/1.8 system-ui;margin:50px;color:#203047}a{color:#17639a}</style><h1>光学模拟实验</h1><p>共用 LightPipes 传播框架 · 三个独立模块</p><ul>'+links+'</ul>', encoding="utf-8")
