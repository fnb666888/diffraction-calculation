"""Run every concrete example and export a complete, self-contained workflow folder."""
from dataclasses import asdict, dataclass
from pathlib import Path
import argparse
import copy
import gc
import html
import json
import shutil
import sys
import numpy as np
from PIL import Image
from .core import Grid, Plane, far_field, far_field_window, propagate, fft2c, transfer_function
from .diffraction import DiffractionConfig, run as diffraction, cdi_demo, square_fresnel, airy_intensity
from .talbot import TalbotConfig, run as talbot
from .holography import HolographyConfig, run as holography
from .abbe import AbbeConfig, FILTER_LABELS, run as abbe
from .objects import grating
from .metrics import relative_error
from .results import Result
from .input_visualization import export_input_visuals, _view_limit
from .input_models import model_for
from .reporting import plt, image
from .paths import PROJECT_ROOT, EXAMPLES_DIR, WORKFLOWS_DIR


@dataclass
class Case:
    id: str
    title: str
    family: str
    variant: str
    config: dict

    @property
    def folder(self):
        return self.id+"_"+self.title


def cases():
    base = EXAMPLES_DIR
    out = []
    for cid, title, variant, settings in (
        ("01", "圆孔衍射", "circle", {"kind": "circle"}),
        ("02", "方孔衍射", "square", {"kind": "square"}),
        ("03", "单缝衍射", "slit", {"kind": "slit"}),
        ("04", "振幅图片衍射", "image_amplitude", {"image_path": str(base/"amplitude_target.png"), "image_mode": "amplitude"}),
        ("05", "相位图片衍射", "image_phase", {"image_path": str(base/"phase_target.png"), "image_mode": "phase"})):
        out.append(Case(cid, title, "diffraction", variant, asdict(DiffractionConfig(cdi_iterations=0, **settings))))
    out.append(Case("06", "CDI迭代相位恢复", "cdi", "cdi", {"n": 128, "dx": 5e-6, "wavelength": 632.8e-9, "iterations": 400, "seed": 7}))
    for cid, title, kind, dims in (
        ("07", "一维二值振幅光栅泰伯效应", "binary", 1),
        ("08", "一维正弦振幅光栅泰伯效应", "sine", 1),
        ("09", "一维二值相位光栅泰伯效应", "phase", 1),
        ("10", "二维振幅光栅泰伯效应", "binary", 2),
        ("11", "二维相位光栅泰伯效应", "phase", 2),
        ("12", "一维正弦相位光栅泰伯效应", "phase_sine", 1)):
        out.append(Case(cid, title, "talbot", "talbot", asdict(TalbotConfig(kind=kind, dimensions=dims))))
    for cid, title, variant in (("13", "分数泰伯与波前曲率", "fractional"), ("14", "双光栅泰伯莫尔条纹", "tli"), ("15", "部分相干照明泰伯效应", "coherence")):
        out.append(Case(cid, title, "talbot", variant, asdict(TalbotConfig())))
    # Keep analytic apertures/gratings here. Holography and Abbe presentation
    # examples now use captured photographs in outputs/image_simulations/.
    return out


def _compute(case, cache):
    key = case.family+json.dumps(case.config, sort_keys=True)
    if key not in cache:
        if case.family=="diffraction":
            cache[key] = diffraction(DiffractionConfig(**case.config))
        elif case.family=="talbot":
            cache[key] = talbot(TalbotConfig(**case.config))
        elif case.family=="holography":
            cache[key] = holography(HolographyConfig(**case.config))
        elif case.family=="abbe":
            cache[key] = abbe(AbbeConfig(**case.config))
        else:
            m, a = cdi_demo(case.config["n"], case.config["iterations"], case.config["seed"])
            g = Grid(case.config["n"], case.config["dx"], case.config["wavelength"])
            cache[key] = Result("cdi", dict(case.config), m, {"object": Plane(a["cdi_truth"], g.x, "true target")}, a,
                                ["支集与非负振幅是恢复先验；真值仅用于迭代结束后的评价。",
                                 "解允许平移、全局相位及共轭反转；原始解与评价对齐后的解分别保留。"])
    result = copy.copy(cache[key])
    result.parameters, result.metrics = dict(result.parameters), dict(result.metrics)
    result.planes, result.arrays = dict(result.planes), dict(result.arrays)
    result.notes = list(result.notes)
    if case.family=="diffraction":
        result.planes = {name: result.planes[name] for name in ("object", "propagated", "far_field")}
        result.arrays = {}
        result.metrics = {name: result.metrics[name] for name in ("sampling", "boundary", "power_ratio")}
    if case.variant=="coherence":
        g = Grid(case.config["n"], case.config["dx"], case.config["wavelength"])
        X, _ = g.xy
        reference_mode = result.planes["object"].field*np.exp(2j*np.pi*result.arrays["coherence_tilts_per_m"][0]*X)
        result.planes["object"] = Plane(reference_mode, g.x, "first illumination mode")
        result.notes.append("输入振幅/相位图表示第一个照明模态；部分相干总输入和输出均没有单一相位。")
        result.parameters["input_field_meaning"] = "first coherent illumination mode"
    if case.variant=="phase_steps":
        result.parameters["measurement_reference_angle_deg"] = 0.
        result.parameters["measurement_scheme"] = "four on-axis exposures, reference phase 0/pi/2/pi/3pi/2"
    result.parameters["workflow_variant"] = case.variant
    return result


class Writer:
    def __init__(self, directory, case, result):
        self.directory, self.case, self.result = Path(directory), case, result
        self.stages, self.overview = [], []
        self.directory.mkdir(parents=True, exist_ok=True)

    def figure(self, fig, folder, name, title, text="", overview=True):
        path = self.directory/folder/(name+".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=.15)
        plt.close(fig)
        fig.clear()
        entry = {"path": path.relative_to(self.directory).as_posix(), "title": title, "description": text}
        self.stages.append(entry)
        if overview:
            self.overview.append(entry)
        gc.collect()
        return entry

    def scalar(self, data, x, folder, name, title, text="", phase=False, log=False, frequency=False, overview=True, limit=None):
        fig, ax = plt.subplots(figsize=(6.5, 5.8), constrained_layout=True)
        image(ax, data, x, title, phase=phase, log=log, frequency=frequency)
        if limit is not None:
            ax.set(xlim=(-limit*1e6, limit*1e6), ylim=(-limit*1e6, limit*1e6))
        if self.result.parameters.get("display_image_y_down") and not frequency:
            ax.invert_yaxis()
        return self.figure(fig, folder, name, title, text, overview)

    def plane(self, plane, folder, prefix, title, modes=("amplitude", "phase", "intensity"), overview=True):
        limit = None
        c = self.case.config
        if self.case.family=="diffraction" and folder=="04_输出结果":
            width = c["image_samples"]*c["dx"] if c["image_path"] else c["width"]
            height = width if c["image_path"] else c["height"] or (width*5 if c["kind"]=="slit" else width)
            limit = 5*c["wavelength"]*self.result.parameters["workflow_far_z_m"]/width if prefix=="远场" else max(width, height)*.65+3*np.sqrt(c["wavelength"]*c["z"])
            limit = min(limit, max(abs(plane.x)))
        elif self.case.family=="holography" and folder=="04_输出结果":
            limit = _view_limit(self.result)*1e-6
        elif self.case.family=="abbe" and folder=="04_输出结果":
            limit=_view_limit(self.result)*(c["f2"]/c["f1"])*1e-6
        elif self.case.family=="talbot" and self.case.variant in ("talbot", "fractional"):
            limit = c["period"]*4
        for mode in modes:
            if mode=="phase":
                values = np.ma.masked_where(np.abs(plane.field)<=np.abs(plane.field).max()*1e-5, np.angle(plane.field))
                suffix, label = "相位", "相位（有效振幅区）"
            elif mode=="amplitude":
                values, suffix, label = np.abs(plane.field), "振幅", "振幅"
            else:
                values, suffix, label = plane.intensity, "强度", "强度（对数）"
            self.scalar(values, plane.x, folder, prefix+"_"+suffix, title+" · "+label,
                        phase=mode=="phase", log=mode=="intensity", overview=overview, limit=limit)

    def card(self, title, lines, folder="05_验证", name="定量指标", overview=True):
        fig, ax = plt.subplots(figsize=(6.5, 4.4), constrained_layout=True)
        ax.axis("off")
        ax.text(.03, .94, title, va="top", fontsize=15, weight="bold", color="#184e73", transform=ax.transAxes)
        ax.text(.03, .8, "\n\n".join(lines), va="top", fontsize=11, transform=ax.transAxes, wrap=True)
        return self.figure(fig, folder, name, title, overview=overview)


def _diffraction_stages(w):
    r, c = w.result, w.case.config
    g = Grid(c["n"], c["dx"], c["wavelength"])
    u0 = r.planes["object"].field
    tf = np.fft.fftshift(transfer_function(g, c["z"]))
    r.arrays["propagation_transfer_function_centered"] = tf
    r.arrays["input_spectrum_centered"] = fft2c(u0)
    w.scalar(np.angle(tf), g.frequency, "03_计算过程", "传递函数相位", "菲涅耳传递函数相位", phase=True, frequency=True, overview=False)
    w.scalar(np.abs(fft2c(u0))**2, g.frequency, "03_计算过程", "输入场频谱", "输入复场的频谱功率", log=True, frequency=True, overview=False)
    w.card("本例复场传播计算", model_for(r).propagation+[f"LightPipes 后端；z={c['z']:g} m"],
           "03_计算过程", "传播计算公式")
    w.plane(r.planes["propagated"], "04_输出结果", "近场", f"近场 z = {c['z']:g} m")
    width = c["image_samples"]*g.dx if c["image_path"] else c["width"]
    height = width if c["image_path"] else c["height"] or (width*5 if c["kind"]=="slit" else width)
    far_z = max(.5, (max(width, height)/2)**2/(g.wavelength*.05))
    far = far_field(u0, g, far_z)
    r.planes["far_field"] = far
    r.parameters["workflow_far_z_m"] = far_z
    far_view = far
    if r.parameters.get("dense_far_field"):
        limit = 5*g.wavelength*far_z/width
        far_view = far_field_window(u0, g, far_z, limit)
        r.planes["far_field_zoom"] = far_view
    w.plane(far_view, "04_输出结果", "远场", f"远场 z = {far_z:.4g} m")
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    if w.case.variant=="circle":
        line = far.intensity[g.n//2]
        line = line/line.max()
        analytic = airy_intensity(far.x, width, g.wavelength, far_z)
        roi = np.abs(far.x)<5*g.wavelength*far_z/width
        ax.plot(far.x[roi]*1e3, line[roi], label="圆孔 FFT")
        ax.plot(far.x[roi]*1e3, analytic[roi], "--", label="Airy 解析解")
        error = relative_error(line[roi], analytic[roi])
        r.arrays.update(airy_numeric_profile=line, airy_analytic_profile=analytic, airy_x_m=far.x)
        title = "圆孔远场中心截面 · Airy 验证"
        ax.set(xlabel="x (mm)", ylabel="I/Imax")
    elif w.case.variant in ("square", "slit"):
        exact = square_fresnel(g.x, width, g.wavelength, c["z"], height)
        r.arrays["continuous_fresnel_reference"] = exact
        numeric, analytic = r.planes["propagated"].intensity, np.abs(exact)**2
        roi = np.abs(g.x)<max(width, 3*np.sqrt(g.wavelength*c["z"]))
        select = np.ix_(roi, roi)
        error = relative_error(numeric[select], analytic[select])
        ax.plot(g.x[roi]*1e6, numeric[g.n//2, roi], label="LightPipes 近场")
        ax.plot(g.x[roi]*1e6, analytic[g.n//2, roi], "--", label="矩形 Fresnel 积分")
        title = "方孔" if w.case.variant=="square" else "单缝"
        title += "中心截面 · 连续积分验证"
        ax.set(xlabel="x (μm)", ylabel="强度")
    else:
        returned = propagate(r.planes["propagated"].field, g, -c["z"])
        r.planes["back_propagated"] = Plane(returned, g.x, "round-trip validation")
        error = relative_error(returned, u0)
        mid = g.n//2
        ax.plot(g.x*1e6, np.abs(u0[mid]), label="原始输入振幅")
        ax.plot(g.x*1e6, np.abs(returned[mid]), "--", label="传播后反向回传")
        title = "图片物体 · 传播往返验证"
        ax.set(xlabel="x (μm)", ylabel="振幅")
    ax.set_title(title)
    ax.legend()
    w.figure(fig, "05_验证", "中心截面与基准", title)
    r.metrics["workflow_reference_relative_error"] = error
    r.metrics["workflow_far_fresnel_number"] = (max(width, height)/2)**2/(g.wavelength*far_z)
    w.card("本例定量验证", [f"参考解/往返相对误差：{error:.4g}", f"同采样传播功率比：{r.metrics['power_ratio']:.8f}",
                           f"近场距离：{c['z']:g} m；远场距离：{far_z:.4g} m"], overview=False)
    return ["按本例具体透过率函数与采样约定构造输入复场，公式见输入表达式图。", "LightPipes 菲涅耳传播得到近场复场，输出振幅、相位及强度。",
            f"远场单 FFT 独立采用 z={far_z:.6g} m（NF≤0.05），输出坐标 x=λzf，同时展示振幅、相位和强度。相位保留球面波二次相位，省略共同轴向相位 exp(ikz)，低振幅区屏蔽。", "仅使用本例对应的解析解或往返校核验证结果。"]


def _cdi_stages(w):
    r, a = w.result, w.result.arrays
    g = Grid(**{k: w.case.config[k] for k in ("n", "dx", "wavelength")})
    w.scalar(a["cdi_intensity_only"], g.frequency, "03_计算过程", "仅强度测量", "FFT远场测量 · 仅保留强度", log=True, frequency=True)
    w.scalar(a["cdi_support"].astype(float), g.x, "03_计算过程", "支集先验", "恢复先验 · 矩形支集", overview=False)
    w.card("仅强度迭代恢复", ["随机相位初始化", "频域：测量振幅替换 + 保留当前相位", "物域：支集与非负约束", "反复 FFT / IFFT；比较 ER 与 HIO"], "03_计算过程", "迭代恢复流程")
    r.planes["hio_raw"] = Plane(a["cdi_hio_raw"], g.x, "raw reconstruction")
    r.planes["hio_aligned"] = Plane(a["cdi_hio_aligned"], g.x, "evaluation aligned reconstruction")
    w.plane(r.planes["hio_raw"], "04_输出结果", "HIO原始解", "HIO 原始恢复", modes=("amplitude", "phase"), overview=False)
    w.plane(r.planes["hio_aligned"], "04_输出结果", "HIO对齐解", "HIO 恢复（评价歧义对齐）", modes=("amplitude", "phase"))
    w.scalar(np.abs(a["cdi_er"]), g.x, "04_输出结果", "ER振幅", "ER 恢复振幅", overview=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ax.semilogy(a["cdi_er_residual"], label="ER")
    ax.semilogy(a["cdi_hio_residual"], label="HIO")
    ax.set(title="仅强度约束 · 迭代收敛", xlabel="迭代次数", ylabel="傅里叶振幅相对残差")
    ax.legend()
    w.figure(fig, "05_验证", "迭代残差", "ER/HIO 迭代残差")
    w.scalar(np.abs(a["cdi_hio_aligned"]-a["cdi_truth"]), g.x, "05_验证", "恢复误差", "恢复复场绝对误差")
    w.card("CDI 定量结果", [f"HIO 最佳测量残差：{r.metrics['hio_fourier_amplitude_residual']:.4g}",
                            f"歧义对齐后物体误差：{r.metrics['hio_aligned_complex_relative_error']:.4g}",
                            "先验：已知支集 + 实数非负振幅"], overview=False)
    return ["由已知靶标计算 FFT 强度，模拟探测器只记录强度。", "恢复算法只获取强度、支集及实数非负先验，不获取真值相位。",
            "分别运行 ER 与 HIO；输出恢复场及测量残差。", "结束后才与真值对齐比较，保留未对齐原始解。"]


def _talbot_stages(w):
    r, a, c = w.result, w.result.arrays, w.case.config
    x = r.planes["object"].x
    zt = r.metrics["talbot_distance_m"]
    if w.case.variant in ("talbot", "fractional"):
        w.card("多平面传播", [f"zT = 2d²/λ = {zt*1e3:.6g} mm", "FFT 光栅复场 × 各距离的 H(z)", "分别 IFFT 得到各平面复场", "计算强度、振幅和相位"], "03_计算过程", "多平面传播公式")
        selected = (("half", "半距"), ("full", "整距")) if w.case.variant=="talbot" else (("eighth", "八分之一距"), ("quarter", "四分之一距"), ("curved_quarter", "曲率入射四分之一距"))
        for key, label in selected:
            w.plane(r.planes[key], "04_输出结果", label, label, modes=("intensity", "amplitude", "phase"), overview=False)
            w.scalar(r.planes[key].intensity, x, "04_输出结果", label+"_流程强度", label+"强度", overview=True, limit=c["period"]*4)
        fig, ax = plt.subplots(figsize=(7, 4.8), constrained_layout=True)
        roi = np.abs(x)<=c["period"]*4
        ax.imshow(a["carpet_intensity"][:, roi], origin="lower", aspect="auto", cmap="magma",
                  extent=[x[roi][0]*1e6, x[roi][-1]*1e6, 0, c["z_periods"]])
        ax.set(title="x–z 泰伯演化", xlabel="x (μm)", ylabel="z/zT")
        w.figure(fig, "03_计算过程", "强度演化", "多个平面的 x–z 强度演化")
        fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
        ax.stem(a["orders"], a["order_power_numeric"], basefmt=" ", label="数值级次")
        if "order_power_analytic" in a:
            ax.plot(a["orders"], a["order_power_analytic"], "x", label="连续光栅解析值")
        ax.set(title="光栅远场级次验证", xlabel="级次 m", ylabel="级次功率")
        ax.legend()
        w.figure(fig, "05_验证", "衍射级次", "衍射级次及包络验证", overview=w.case.variant=="talbot")
        w.card("泰伯自成像验证", [f"zT = {zt*1e3:.6g} mm", f"整距自像强度相似度：{r.metrics['zT_intensity_similarity']:.8f}",
                                  f"整距复场相对误差：{r.metrics['zT_complex_relative_error']:.4g}",
                                  f"半距平移复场误差：{r.metrics['half_shift_complex_relative_error']:.4g}"])
        return ["构造周期光栅，采用整数周期窗口作为理想无限光栅模型。", "以 zT=2d²/λ 为距离标尺，在多个平面独立传播。",
                "保存各平面的强度、振幅和相位，绘制 x–z 演化。", "整距比较原像；半距比较沿周期方向平移 d/2 的像。相位光栅同时验证复场。"]
    if w.case.variant=="tli":
        g = Grid(c["n"], c["dx"], c["wavelength"])
        second = grating(g, c["period"]*(1+c["analyzer_mismatch"]), "binary", c["duty"], dimensions=c["dimensions"])
        w.scalar(np.abs(second), x, "03_计算过程", "第二片分析光栅", "第二片分析光栅的透过率")
        w.plane(r.planes["full"], "03_计算过程", "第一光栅传播", "第一光栅传播至 zT", modes=("intensity",))
        for key, label in (("tli_flat_intensity", "平面波双光栅"), ("tli_curved_intensity", "曲率波前双光栅")):
            w.scalar(a[key], x, "04_输出结果", label, label+"透射强度")
        fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
        ax.plot(x*1e6, a["tli_flat_envelope"], label="平面波")
        ax.plot(x*1e6, a["tli_curved_envelope"], label="曲率波前")
        ax.set(title="双光栅莫尔包络响应", xlabel="x (μm)", ylabel="空间平滑后的强度")
        ax.legend()
        w.figure(fig, "05_验证", "莫尔包络对比", "平面波与曲率波前的莫尔包络")
        w.card("双光栅参数", [f"第一光栅周期：{c['period']*1e6:g} μm", f"第二光栅周期：{c['period']*(1+c['analyzer_mismatch'])*1e6:g} μm",
                                f"间距 zT：{zt*1e3:.6g} mm", f"曲率半径：{c['curvature_radius']:g} m"])
        return ["平面波/曲率波前照明第一片光栅。", "传播 zT 后乘第二片光栅透过率，记录透射强度。",
                "第二片周期默认略失配4%，产生可见莫尔包络。", "空间平滑提取包络并比较波前曲率响应；这一步不是再次自由空间传播。"]
    # An incoherent mixture has intensity, not a single amplitude/phase wavefront.
    w.card("部分相干传播计算", ["各输入模态独立通过光栅", "每个模态单独进行菲涅耳传播", r"$I_z=\sum_m w_m |U_{z,m}|^2$", "只做强度求和，不做复振幅求和"], "03_计算过程", "非相干叠加公式")
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    ax.plot(np.arange(len(a["coherence_tilts_per_m"])), a["coherence_tilts_per_m"]*1e-3, "o")
    ax.set(title="部分相干输入 · 高斯角谱模态", xlabel="模态编号", ylabel=r"倾斜空间频率 ($\mathrm{mm}^{-1}$)")
    w.figure(fig, "03_计算过程", "照明模态", "多个独立照明模态及等权重")
    w.scalar(r.planes["full"].intensity, x, "04_输出结果", "完全相干自像", "完全相干照明 · 整距强度")
    w.scalar(r.planes["partial_coherence_full"].intensity, x, "04_输出结果", "部分相干自像", "各模态强度平均 · 整距强度")
    fig, ax = plt.subplots(figsize=(6.5, 4.8), constrained_layout=True)
    roi = np.abs(x)<c["period"]*4
    ax.plot(x[roi]*1e6, r.planes["full"].intensity[len(x)//2, roi], label="完全相干")
    ax.plot(x[roi]*1e6, r.planes["partial_coherence_full"].intensity[len(x)//2, roi], label="部分相干")
    ax.set(title="自像衬度变化", xlabel="x (μm)", ylabel="强度")
    ax.legend()
    w.figure(fig, "05_验证", "自像衬度对比", "相干性变化后的中心截面")
    w.card("部分相干模型", [f"模态数：{c['coherence_modes']}，各模态等权", f"相干长度：{c['coherence_length']*1e6:g} μm",
                              "输出 = 各模态强度平均", "不存在单一总场相位；输入图显示第一模态。"])
    return ["将照明建模为高斯倾斜分布的多个互不相干平面波。", "分别乘光栅并传播，最后求强度的加权和。",
            "比较完全相干与部分相干自像衬度。", "输入幅相图只代表第一模态；不对非相干总强度编造相位。"]


def _holography_stages(w):
    r, a, c = w.result, w.result.arrays, w.case.config
    x = r.planes["object"].x
    if w.case.variant=="off_axis":
        w.plane(r.planes["object_wave"], "03_计算过程", "记录面物光", "传播到记录面的物光", modes=("intensity",), overview=False)
        w.scalar(np.angle(a["reference_wave"]), x, "03_计算过程", "参考光相位", "离轴参考光的倾斜相位", phase=True, overview=False)
        w.scalar(a["hologram_intensity_only"], x, "03_计算过程", "强度全息图", "探测器记录的强度全息图")
        w.scalar(a["hologram_spectrum"], a["frequency_per_m"], "03_计算过程", "全息图频谱", "全息图 0级 / ±1级频谱", log=True, frequency=True)
        w.plane(r.planes["virtual_image"], "04_输出结果", "重建虚像", "滤出物光后反向传播重建")
        w.plane(r.planes["twin_image"], "04_输出结果", "孪生实像", "共轭一级重建", overview=False)
        w.plane(r.planes["raw_virtual"], "04_输出结果", "原参考光未分级重现", "原参考光照明原始全息图", modes=("intensity",), overview=False)
        w.plane(r.planes["raw_real"], "04_输出结果", "共轭参考光未分级重现", "共轭参考光照明原始全息图", modes=("intensity",), overview=False)
        w.card("离轴全息定量恢复", [f"强度相似度：{r.metrics['virtual_intensity_similarity']:.8f}",
                                      f"复场相对误差：{r.metrics['virtual_complex_relative_error']:.4g}",
                                      f"相位 RMSE：{r.metrics['virtual_phase_rmse_rad']:.4g} rad", "同一输入经过明确的有限 NA 带宽 B。"])
        return ["透明片复场先通过有限NA带宽B，随后传播到记录面。", "与倾斜参考光相干叠加，探测器只保存 |Uobj+R|²。",
                "以已知参考光解调并低通分离一级物光，反向传播得到虚像。", "另分离共轭一级得到孪生实像；与带限输入真值比较幅相误差。"]
    if w.case.variant=="phase_steps":
        fig, ax = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
        for axis, frame, label in zip(ax.flat, a["phase_step_intensities"], ("0", "π/2", "π", "3π/2")):
            image(axis, frame, x, "参考相位 "+label)
        w.figure(fig, "03_计算过程", "四幅强度曝光", "同轴参考 · 四步相移强度图")
        w.card("四步直接解调", ["R = A exp(iφ)，参考角为0", "实部来自 I0 − Iπ", "虚部来自 Iπ/2 − I3π/2", "复场 = (实部 + i·虚部)/(4A)"], "03_计算过程", "相移解调公式")
        w.plane(r.planes["phase_shift_image"], "04_输出结果", "相移重建", "相移解调后反向传播")
        w.card("相移全息误差", [f"记录面复场误差：{r.metrics['phase_shift_field_relative_error']:.4g}",
                                f"重建物面复场误差：{r.metrics['phase_shift_image_relative_error']:.4g}",
                                "四幅强度直接给出振幅及相位，无需迭代。"])
        return ["带限输入物场传播至记录面。", "同轴参考相位依次为0、π/2、π、3π/2，记录四幅强度。",
                "以四步公式直接恢复记录面复场，再反向传播。", "此流程的测量参考角为0；不使用离轴全息图作为四步测量数据。"]
    w.scalar(a["screen_phase_rad"], x, "03_计算过程", "薄相位屏", "记录面位置的薄相位屏 W(x,y)", phase=True)
    w.plane(r.planes["aberrated_image"], "03_计算过程", "畸变基线", "未补偿重建", modes=("intensity",))
    for key, label in (("common_path_image", "共光路补偿"), ("corrected_image", "已知屏逆补偿"), ("reverse_image", "逆参考共轭补偿")):
        w.plane(r.planes[key], "04_输出结果", label, label, modes=("intensity",))
        w.plane(r.planes[key], "04_输出结果", label, label, modes=("amplitude", "phase"), overview=False)
    w.card("薄相位屏补偿误差", [f"畸变基线误差：{r.metrics['aberrated_image_relative_error']:.4g}",
                                 f"共光路误差：{r.metrics['common_path_image_relative_error']:.4g}",
                                 f"已知屏逆补偿误差：{r.metrics['known_screen_corrected_relative_error']:.4g}",
                                 f"逆参考共轭误差：{r.metrics['reverse_reference_conjugate_error']:.4g}"])
    return ["物光传播到记录面后乘薄相位屏 exp(iW)，形成畸变基线。", "共光路参考穿过同一屏，以四步强度测量消去公共相位。",
            "另外演示已知相位屏数值逆补偿及共轭波经同屏相消。", "逆参考路径恢复共轭物场；本例限定为记录面薄屏，不代表未知体散射介质。"]


def _abbe_layout(w):
    from matplotlib.patches import Ellipse
    c=w.case.config
    f1,f2=c["f1"],c["f2"]
    positions=np.array([0,f1,2*f1,2*f1+f2,2*(f1+f2)])
    fig,ax=plt.subplots(figsize=(11,4.4),constrained_layout=True)
    names=["物面 U0","第一透镜 L1","频谱面 H=P·M","第二透镜 L2","倒像面 Ui"]
    height=.25
    for slope,color in zip((0,-height/f1,height/f1),("#db754d","#3c9ca8","#a475c7")):
        y1=height+slope*f1
        slope1=slope-y1/f1
        yf=y1+slope1*f1
        y2=yf+slope1*f2
        yi=y2+(slope1-y2/f2)*f2
        ax.plot(positions*1e3,[height,y1,yf,y2,yi],color=color,lw=1.5)
    ymax=.72*max(1,f2/f1)
    for i,(z,name) in enumerate(zip(positions,names)):
        if i in (1,3):
            ax.add_patch(Ellipse((z*1e3,0),positions[-1]*1e3*.025,ymax*1.55,facecolor="#c6e6f0",alpha=.65))
        else:
            ax.axvline(z*1e3,color="#4c6780",lw=2,ls="--" if i==2 else "-")
        ax.text(z*1e3,ymax*.95,name,ha="center",fontsize=10)
    for left,right,label in zip(positions[:-1],positions[1:],["f1","f1","f2","f2"]):
        ax.annotate("",(left*1e3,-ymax*.8),(right*1e3,-ymax*.8),arrowprops={"arrowstyle":"<->","color":"#52718b"})
        ax.text((left+right)*.5e3,-ymax*.73,label,ha="center")
    ax.axhline(0,color="#bec9d1",lw=.7)
    ax.set(xlim=(-positions[-1]*60,positions[-1]*1060),ylim=(-ymax,ymax*1.15),xlabel="光轴位置 z (mm)",yticks=[],
           title=f"阿贝4f相干成像 · f1={f1*1e3:g} mm，f2={f2*1e3:g} mm，横向放大率={-f2/f1:g}")
    return fig


def _abbe_stages(w):
    r,c=w.result,w.case.config
    freq=r.arrays["frequency_per_m"]
    w.figure(_abbe_layout(w),"03_计算过程","4f光路","物面 → L1 → 频谱滤波 → L2 → 倒像面")
    w.card("本例复场计算与滤波函数",model_for(r).propagation,"03_计算过程","4f计算公式",overview=False)
    for key,prefix in (("fourier","第一透镜频谱"),("filtered_fourier","滤后频谱")):
        field=r.planes[key].field
        for mode in ("振幅","相位","强度"):
            values=np.abs(field) if mode=="振幅" else (np.angle(field) if mode=="相位" else np.abs(field)**2)
            if mode=="相位":
                values=np.ma.masked_where(np.abs(field)<np.abs(field).max()*1e-5,values)
            w.scalar(values,freq,"03_计算过程",prefix+"_"+mode,prefix+" · "+mode,
                     phase=mode=="相位",log=mode=="强度",frequency=True,overview=mode=="强度",
                     text="频率ν对应频谱面实际位置ξ=λf1ν；原始复场与ξ坐标另存。")
    for key,prefix,overview in (("pupil","有限NA孔径",False),("filter_mask","附加滤波器",True),("effective_transfer","有效复透过率",False)):
        mask=r.arrays[key]
        w.scalar(np.abs(mask),freq,"03_计算过程",prefix+"_振幅",prefix+" · 振幅透过率",frequency=True,overview=overview)
        if key!="pupil":
            phase=np.ma.masked_where(np.abs(mask)<1e-12,np.angle(mask))
            w.scalar(phase,freq,"03_计算过程",prefix+"_相位",prefix+" · 相位延迟",phase=True,frequency=True,overview=overview)
    w.plane(r.planes["baseline_image"],"04_输出结果","无附加滤波倒像","相同NA · 无附加滤波基准",modes=("intensity",))
    w.plane(r.planes["baseline_image"],"04_输出结果","无附加滤波倒像","相同NA · 无附加滤波基准",modes=("amplitude","phase"),overview=False)
    w.plane(r.planes["image"],"04_输出结果","滤波后倒像",FILTER_LABELS[c["filter_kind"]]+" · 实际像面")
    if c["filter_kind"]=="phase_contrast":
        baseline=r.planes["baseline_image"].intensity
        after=r.planes["image"].intensity
        position=r.planes["image"].x*1e6
        step=(position[1]-position[0])/2
        extent=[position[0]-step,position[-1]+step]*2
        fig,axes=plt.subplots(1,3,figsize=(13,4.3),constrained_layout=True)
        vmin,vmax=min(baseline.min(),after.min()),max(baseline.max(),after.max())
        for ax,values,title in zip(axes,(baseline,after,after-baseline),("无附加滤波 · 线性强度","中心相位片 · 线性强度","强度差 ΔI")):
            if title.endswith("ΔI"):
                bound=max(float(np.abs(values).max()),1e-12)
                im=ax.imshow(values,origin="lower",extent=extent,cmap="RdBu_r",vmin=-bound,vmax=bound)
            else:
                im=ax.imshow(values,origin="lower",extent=extent,cmap="gray",vmin=vmin,vmax=vmax)
            limit=_view_limit(r)*(c["f2"]/c["f1"])
            ax.set(title=title,xlim=(-limit,limit),ylim=(-limit,limit),xlabel="x (μm)",ylabel="y (μm)")
            plt.colorbar(im,ax=ax,fraction=.046,pad=.04,label="a.u.")
        w.figure(fig,"05_验证","相衬线性强度对比","弱相位 → 可见强度对比（统一线性标度）")
    x=r.planes["image"].x
    limit=_view_limit(r)*(c["f2"]/c["f1"])
    fig,ax=plt.subplots(figsize=(6.5,4.7),constrained_layout=True)
    mid=len(x)//2
    ax.plot(x*1e6,r.planes["baseline_image"].intensity[mid],label="相同NA，无附加滤波")
    ax.plot(x*1e6,r.planes["image"].intensity[mid],label="滤波后")
    ax.set(xlim=(-limit,limit),xlabel="像面 x (μm)",ylabel="强度 (a.u.)",title="像面中心截面 · 统一强度标度")
    ax.legend()
    w.figure(fig,"05_验证","像面截面对比","滤波前后同一强度标度",overview=False)
    m=r.metrics
    w.card("阿贝成像定量验证",[f"横向放大率：{m['magnification_signed']:g}；像面采样：{m['image_dx_m']*1e6:g} μm",
        f"有限NA保留功率：{m['pupil_transmitted_power_fraction']:.6f}",
        f"滤波后 / 无附加滤波功率：{m['filter_power_fraction_of_baseline']:.6f}",
        f"频谱与像面功率相对差：{m['spectral_power_balance_relative_error']:.3g}"])
    return ["按具体透过率函数构造物体，在单位平面波相干照明下形成输入复场。",
            f"第一透镜将复场变换到频谱面，ξ=λf1ν；f1={c['f1']*1e3:g} mm。",
            "频谱面以有限NA孔径P和所选复滤波器M相乘，H=P·M；遮光改变振幅，相位片改变相位。",
            f"第二透镜再次作正向傅里叶变换，输出物理倒像，横向放大率-f2/f1={m['magnification_signed']:g}。",
            "与相同NA的无附加滤波倒像比较，并核对频谱功率和像面功率；图像强度为复场模平方。"]


def _overview(w):
    # Images are arranged in a snake so every neighbouring stage has a visible
    # direction arrow; full-resolution individual images are retained alongside.
    entries = w.overview
    columns, rows = 3, (len(entries)+2)//3
    fig = plt.figure(figsize=(19, rows*5.2+1.4))
    fig.patch.set_facecolor("#f3f6fa")
    fig.suptitle(w.case.title+" · 从输入到输出的全流程", y=.99, fontsize=24, weight="bold", color="#164e73")
    boxes = []
    for i, entry in enumerate(entries):
        row, position = divmod(i, columns)
        col = position if row%2==0 else columns-1-position
        left, bottom = .035+col*.322, .07+(rows-1-row)*(.88/rows)
        width, height = .285, .77/rows
        ax = fig.add_axes([left, bottom, width, height])
        with Image.open(w.directory/entry["path"]) as picture:
            ax.imshow(np.asarray(picture.convert("RGB")))
        ax.axis("off")
        ax.set_title(f"{i+1:02d}  "+entry["title"], fontsize=13, pad=10, color="#203b51")
        boxes.append((left, bottom, width, height))
    from matplotlib.patches import FancyArrowPatch
    for (left, bottom, width, height), (nl, nb, nw, nh) in zip(boxes, boxes[1:]):
        if np.isclose(bottom, nb):
            start = (left+width+.003, bottom+height/2) if nl>left else (left-.003, bottom+height/2)
            end = (nl-.007, nb+nh/2) if nl>left else (nl+nw+.007, nb+nh/2)
        else:
            start, end = (left+width/2, bottom+.002), (nl+nw/2, nb+nh+.012)
        fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=20, lw=2, color="#4c7e9a"))
    fig.text(.04, .024, "按编号与箭头阅读；振幅、相位和强度是同一复场的不同视图。每一步原图、参数和数据均另存。", fontsize=13, color="#52657a")
    fig.savefig(w.directory/"00_全流程图.png", dpi=160)
    fig.savefig(w.directory/"00_全流程图.svg")
    plt.close(fig)
    fig.clear()
    gc.collect()


def export_case(case, result, directory, reproduction_command=None):
    w = Writer(directory, case, result)
    for key,label in (("filter_amplitude_path","滤波器振幅原图"),("filter_phase_path","滤波器相位原图")):
        if result.parameters.get(key):
            source=Path(result.parameters[key])
            capture=Path(directory)/"03_计算过程"/(label+source.suffix.lower())
            capture.parent.mkdir(parents=True,exist_ok=True)
            if source.resolve()!=capture.resolve():
                shutil.copyfile(source,capture)
            result.parameters[key]=str(capture.resolve())
    input_paths = export_input_visuals(result, directory)
    if result.parameters.get("image_path"):
        source = Path(result.parameters["image_path"])
        destination = Path(directory)/"01_输入图"/("原始输入图片"+source.suffix.lower())
        if source.resolve()!=destination.resolve():
            shutil.copyfile(source, destination)
    model = model_for(result)
    model_error = float(np.linalg.norm(model.input_numeric-result.planes["object"].field)/max(np.linalg.norm(result.planes["object"].field), 1e-30))
    if model_error>1e-12:
        raise ValueError(f"显示公式与实际输入场不一致：{case.id}，相对误差={model_error:.6g}")
    result.metrics["displayed_input_model_relative_error"] = model_error
    display_paths = [input_paths[0], input_paths[4], input_paths[1], input_paths[2]]
    for path, title in zip(display_paths, ("物理输入结构与照明", "本例透过率及复振幅表达式", "实际输入复场 · 振幅", "实际输入复场 · 相位")):
        entry = {"path": path, "title": title, "description": "零振幅处相位未定义。" if title.endswith("相位") else ""}
        w.stages.append(entry)
        w.overview.append(entry)
    steps = {"diffraction": _diffraction_stages, "cdi": _cdi_stages,
             "talbot": _talbot_stages, "holography": _holography_stages,"abbe":_abbe_stages}[case.family](w)
    w.stages.sort(key=lambda entry: entry["path"].split("/")[0])
    w.overview.sort(key=lambda entry: entry["path"].split("/")[0])
    _overview(w)
    result.parameters = _portable_parameters(result.parameters)
    data_dir = result.save(Path(directory)/"06_数据")
    config = {"case_id": case.id, "title": case.title, "family": case.family, "variant": case.variant, "config": _portable_parameters(case.config)}
    (data_dir/"配置.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {"case": config, "stages": w.stages, "overview_steps": w.overview,
                "input_field_meaning": result.parameters.get("input_field_meaning", "single coherent input field")}
    (Path(directory)/"流程清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    descriptions = "\n".join(f"{i}. {text}" for i, text in enumerate(steps, 1))
    notes = "\n".join("- "+note for note in result.notes)
    inventory = "\n".join(f"- `{e['path']}`：{e['title']}" for e in w.stages)
    command = reproduction_command or f".\\.venv\\Scripts\\python.exe -m optics_sim.workflow_catalog --case {case.id}"
    readme = f"# {case.title}：全流程\n\n先看 `00_全流程图.png` 或 `index.html`，然后按编号查看独立图片。\n\n{descriptions}\n\n## 文件顺序\n\n01_输入图 → 02_输入复振幅 → 03_计算过程 → 04_输出结果 → 05_验证 → 06_数据。\n\n## 图表\n\n{inventory}\n\n## 模型约定\n\n{notes}\n\n## 重现本例\n\n在项目根目录运行：\n\n```powershell\n{command}\n```\n\n完整参数见 `06_数据/配置.json`，所有物理长度以 m 保存。\n"
    (Path(directory)/"流程说明.md").write_text(readme, encoding="utf-8")
    figures = "".join(f'<section><h2>{i:02d}. {html.escape(e["title"])}</h2><p>{html.escape(e["description"])}</p><a href="{e["path"]}"><img src="{e["path"]}" loading="lazy"></a></section>' for i, e in enumerate(w.stages, 1))
    page = '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>'+html.escape(case.title)+' · 全流程</title><style>body{max-width:1240px;margin:30px auto;padding:0 22px;font:16px/1.7 system-ui;color:#203047;background:#f4f7fa}img{max-width:100%;background:white}section{margin:36px 0}a{color:#17639a}pre{background:white;padding:20px;overflow:auto}</style>'
    page += '<h1>'+html.escape(case.title)+' · 全流程</h1><p><a href="../index.html">全部模拟</a> · <a href="00_全流程图.png">完整流程大图</a> · <a href="06_数据/data.npz">原始复场数据</a> · <a href="06_数据/metrics.json">定量指标</a></p>'
    page += '<ol>'+"".join('<li>'+html.escape(step)+'</li>' for step in steps)+'</ol>'
    page += '<a href="00_全流程图.png"><img src="00_全流程图.png"></a>'+figures
    page += '<details><summary>模型约定</summary><ul>'+"".join('<li>'+html.escape(note)+'</li>' for note in result.notes)+'</ul></details>'
    page += '<details><summary>本例定量指标</summary><pre>'+html.escape(json.dumps(result.metrics, ensure_ascii=False, indent=2))+'</pre></details>'
    (Path(directory)/"index.html").write_text(page, encoding="utf-8")
    return manifest


def _portable_parameters(parameters):
    """Store project-local source images without machine-specific absolute paths."""
    parameters = dict(parameters)
    for key in ("image_path","filter_amplitude_path","filter_phase_path"):
        if not parameters.get(key):
            continue
        source = Path(parameters[key])
        if source.is_absolute():
            try:
                parameters[key] = source.relative_to(PROJECT_ROOT).as_posix()
            except ValueError:
                pass
    return parameters


def _load_saved(directory):
    metadata = json.loads((directory/"06_数据"/"metrics.json").read_text(encoding="utf-8"))
    planes, arrays = {}, {}
    with np.load(directory/"06_数据"/"data.npz") as archive:
        for name in archive.files:
            if name.endswith("_field") and name[:-6]+"_x_m" in archive.files:
                key = name[:-6]
                planes[key] = Plane(archive[name], archive[key+"_x_m"], key)
            elif not (name.endswith("_x_m") and name[:-4]+"_field" in archive.files):
                arrays[name] = archive[name]
    parameters = dict(metadata["parameters"])
    if parameters.get("image_path"):
        # Re-render from the captured source, even if examples/ has since changed.
        captured = sorted((directory/"01_输入图").glob("原始输入图片.*"))
        if captured:
            parameters["image_path"] = str(captured[0].resolve())
        elif not Path(parameters["image_path"]).is_absolute():
            parameters["image_path"] = str(PROJECT_ROOT/parameters["image_path"])
    for key,label in (("filter_amplitude_path","滤波器振幅原图"),("filter_phase_path","滤波器相位原图")):
        if parameters.get(key):
            captured=sorted((directory/"03_计算过程").glob(label+".*"))
            parameters[key]=str(captured[0].resolve()) if captured else str(PROJECT_ROOT/parameters[key])
    return Result(metadata["module"], parameters, metadata["metrics"], planes, arrays, metadata["notes"])


def generate(directory, selected=None, render_only=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    chosen = [case for case in cases() if selected is None or case.id in selected]
    if not chosen:
        raise ValueError("没有对应编号的模拟。孔径与光栅编号范围01至15；照片全息与阿贝使用 image_workflows 入口。")
    cache = {}
    for case in chosen:
        print("计算并整理："+case.folder, flush=True)
        result = _load_saved(directory/case.folder) if render_only else _compute(case, cache)
        export_case(case, result, directory/case.folder)
        print("完成："+case.folder, flush=True)
        # Distinct Abbe masks have distinct configurations and no shared scans.
        # Retaining all their full complex fields would grow memory per case.
        if case.family=="abbe":
            cache.clear()
        del result
        gc.collect()
    present = [case for case in cases() if (directory/case.folder/"index.html").is_file()]
    write_indexes(directory,present)
    print("汇总入口："+str(directory.resolve()/"index.html"), flush=True)


def write_indexes(directory,present):
    directory=Path(directory)
    links = "".join(f'<li><a href="{case.folder}/index.html">{case.folder}</a> · <a href="{case.folder}/00_全流程图.png">全流程图</a></li>' for case in present)
    abbe_cases=[case for case in present if case.family=="abbe"]
    photo_abbe = (directory.parent/'image_simulations/red_panda/03_照片阿贝成像_圆形低通/index.html').is_file()
    extra='<p><a href="abbe.html">阿贝成像 · 真实照片频谱面滤波</a></p>' if abbe_cases or photo_abbe else ''
    if (directory.parent/'image_simulations/index.html').is_file():
        extra+='<p><a href="../image_simulations/index.html">自定义图片模拟 · 小狗 / 小熊猫 / 猫</a></p>'
    (directory/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>每一种模拟的全流程</title><style>body{max-width:1050px;margin:40px auto;padding:0 24px;font:18px/1.9 system-ui;color:#203047}a{color:#17639a}</style><h1>每一种模拟的全流程</h1><p>每例一个独立文件夹：物理输入 → 输入振幅与相位 → 计算过程 → 输出结果 → 定量验证。</p>'+extra+'<ol>'+links+'</ol>', encoding="utf-8")
    (directory/"目录清单.json").write_text(json.dumps([{"id": case.id, "title": case.title, "folder": case.folder} for case in present], ensure_ascii=False, indent=2), encoding="utf-8")
    if abbe_cases:
        links=''.join(f'<li><a href="{case.folder}/index.html">{case.title}</a></li>' for case in abbe_cases)
        page='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>阿贝成像 · 频谱面调制</title><style>body{max-width:1200px;margin:32px auto;padding:0 22px;font:18px/1.8 system-ui;color:#203047}img{max-width:100%}a{color:#17639a}</style><h1>阿贝成像 · 频谱面调制</h1><p>物体复场 → 第一透镜傅里叶变换 → 有限NA孔径与复滤波器 → 第二透镜 → 物理倒像。</p><p><a href="index.html">全部模拟</a>；桌面界面选择“阿贝成像”，支持参数调整及自定义振幅/相位遮罩。</p><ol>'+links+'</ol>'
        if (directory.parent/'abbe_comparison/dog/index.html').is_file():
            page+='<h2>dog照片滤波对比</h2><p><a href="../abbe_comparison/dog/index.html">详细参数、功率及原始复场</a>。对比图按各图峰值归一化，并转回物方方向；原始物理倒像与强度标度另存。</p><img src="../abbe_comparison/dog/滤波效果对比.png">'
        (directory/'abbe.html').write_text(page,encoding='utf-8')


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="按每一种模拟单独生成从输入到输出的全流程文件夹。")
    parser.add_argument("--output", type=Path, default=WORKFLOWS_DIR, help="默认 outputs/workflows")
    parser.add_argument("--case", nargs="+", help="仅生成指定编号，示例：--case 01 02")
    parser.add_argument("--render-only", action="store_true", help="从每例已保存的复场数据重绘，不重新运行传播扫描")
    args = parser.parse_args()
    selected = {i.zfill(2) for i in args.case} if args.case else None
    generate(args.output, selected, args.render_only)


if __name__=="__main__":
    main()
