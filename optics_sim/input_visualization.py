"""Deterministic physical input drawings followed by actual complex-field maps.

The physical scene is never fed back into the solver as a screenshot. Geometry
and the field are both constructed from the same SI parameters.
"""
from pathlib import Path
import gc
import numpy as np
from PIL import Image
from matplotlib.patches import Circle, Rectangle
from .input_models import model_for, model_figure


def input_maps(plane):
    amplitude = np.abs(plane.field)
    phase = np.ma.masked_where(amplitude <= amplitude.max()*1e-12, np.angle(plane.field))
    return amplitude, phase


def input_description(result):
    p = result.parameters
    if result.module=="cdi":
        label = "非对称振幅靶标（CDI）"
    elif p.get("image_path"):
        label = "灰度相位透明片" if p.get("image_mode")=="phase" else "灰度振幅透明片"
    elif result.module=="diffraction":
        label = {"circle": "不透光屏上的圆孔", "square": "不透光屏上的方孔", "slit": "不透光屏上的单缝"}[p["kind"]]
    elif result.module=="talbot":
        label = f'{p["dimensions"]}维'+("相位光栅" if p["kind"].startswith("phase") else "振幅光栅")
    elif result.module=="abbe":
        label = "阿贝弱相位物体" if p["object_kind"]=="phase_target" else "阿贝二维振幅靶标"
    else:
        label = "振幅及相位透明片（合成靶标）"
    return label


def _view_limit(result):
    p, plane = result.parameters, result.planes["object"]
    window = (plane.x[-1]-plane.x[0]+plane.x[1]-plane.x[0])*1e6/2
    if result.module=="talbot":
        return min(window, p["period"]*4e6)
    if result.module=="abbe" and not p.get("image_path"):
        return min(window,p["object_width"]*.85e6)
    if result.module=="diffraction" and not p.get("image_path"):
        height = p.get("height") or (p["width"]*5 if p["kind"]=="slit" else p["width"])
        return min(window, max(p["width"], height)*.85e6)
    source = result.planes.get("raw_object", plane)
    mask = np.abs(source.field)>np.abs(source.field).max()*.02
    row, col = np.nonzero(mask)
    if not len(row):
        return window
    return min(window, max(abs(source.x[row]).max(), abs(source.x[col]).max())*1.4e6)


def _label_axes(ax, limit):
    ax.set(xlim=(-limit, limit), ylim=(-limit, limit), xlabel="x (μm)", ylabel="y (μm)", aspect="equal")


def physical_input_figure(result):
    from .reporting import plt
    p, plane = result.parameters, result.planes["object"]
    title = input_description(result)
    fig = plt.figure(figsize=(11.5, 6.8), constrained_layout=True)
    spec = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    front, side = fig.add_subplot(spec[0]), fig.add_subplot(spec[1])
    limit = _view_limit(result)
    front.set_facecolor("#f0f3f6")
    geometric = result.module=="diffraction" and not p.get("image_path")
    if geometric:
        front.add_patch(Rectangle((-limit, -limit), 2*limit, 2*limit, facecolor="#404b58", edgecolor="#26313d"))
        width = p["width"]*1e6
        height = (p.get("height") or (p["width"]*5 if p["kind"]=="slit" else p["width"]))*1e6
        if p["kind"]=="circle":
            opening = Circle((0, 0), width/2, facecolor="white", edgecolor="#88a5b9", lw=2)
            size = f"圆孔直径 D = {width:g} μm"
        else:
            opening = Rectangle((-width/2, -height/2), width, height, facecolor="white", edgecolor="#88a5b9", lw=2)
            size = f"开口宽 {width:g} μm × 高 {height:g} μm"
        front.add_patch(opening)
        dim_y = -height*.5-limit*.12
        front.annotate("", (-width/2, dim_y), (width/2, dim_y), arrowprops={"arrowstyle": "<->", "color": "#bee6ff", "lw": 1.6})
        front.text(0, dim_y-limit*.07, f"{width:g} μm", ha="center", va="top", color="white")
        front.set_title(size, pad=12)
        description = "深灰：不透光屏\n白色：真实开口，单位平面波透过"
    else:
        source = result.planes.get("raw_object", plane)
        x = source.x*1e6
        step = (x[1]-x[0])/2
        extent = [x[0]-step, x[-1]+step]*2
        if p.get("image_path"):
            with Image.open(p["image_path"]) as original:
                gray = np.asarray(original.convert("L"), dtype=float)/255
            samples = p["image_samples"]
            with Image.open(p["image_path"]) as original:
                original.thumbnail((samples, samples), Image.Resampling.LANCZOS)
                w, h = original.size
            dx = p["dx"]*1e6
            # Same center placement as image_object, including half-cell endpoints.
            row, col = p["n"]//2-h//2, p["n"]//2-w//2
            extent = [(col-p["n"]//2-.5)*dx, (col+w-p["n"]//2-.5)*dx,
                      (row-p["n"]//2-.5)*dx, (row+h-p["n"]//2-.5)*dx]
            front.imshow(gray, origin="lower", extent=extent, cmap="Blues" if p["image_mode"]=="phase" else "gray", vmin=0, vmax=1)
            description = ("透明片原始灰度图\n灰度 → 相位延迟，片内振幅为1" if p["image_mode"]=="phase" else "透明片原始灰度图\n灰度 → 振幅透过率")
        elif result.module=="abbe" and p["object_kind"]=="phase_target":
            front.imshow(np.angle(source.field),origin="lower",extent=extent,cmap="Blues",vmin=0,vmax=max(p["phase_depth"],1e-12))
            description = "蓝色表示指定的相位延迟\n物体振幅处处为1"
        elif result.module=="talbot" and p["kind"].startswith("phase"):
            X, Y = np.meshgrid(plane.x, plane.x)
            def profile(a):
                cycles = np.round(a/p["period"], 12)
                return ((cycles+.5)%1<p["duty"]).astype(float) if p["kind"]=="phase" else .5*(1+np.cos(2*np.pi*cycles))
            relief = profile(X)+(profile(Y) if p["dimensions"]==2 else 0)
            front.imshow(relief, origin="lower", extent=extent, cmap="Blues", vmin=0, vmax=p["dimensions"])
            description = f"蓝色表示相位延迟层\n调制深度 = {p['phase_depth']:g} rad\n透光区域振幅均为1"
        else:
            front.imshow(np.abs(source.field), origin="lower", extent=extent, cmap="gray", vmin=0, vmax=1)
            description = "白色透光，深色遮光/衰减" if result.module=="talbot" else ("非对称振幅透过率靶标\n支集内非负，支集外遮光" if result.module=="cdi" else "透明片的透光外观\n同时具有非均匀振幅和相位延迟")
        if result.module=="talbot":
            period = p["period"]*1e6
            front.set_title(f"{title} · 周期 d = {period:g} μm", pad=12)
            front.annotate("", (-period/2, limit*.83), (period/2, limit*.83), arrowprops={"arrowstyle": "<->", "color": "#db5b32", "lw": 1.8})
            front.text(0, limit*.9, f"d = {period:g} μm", ha="center", color="#b73f22")
        else:
            front.set_title(title, pad=12)
        front.add_patch(Rectangle((-limit, -limit), 2*limit, 2*limit, fill=False, edgecolor="#42576c", lw=1.4))
    _label_axes(front, limit)
    if p.get("display_image_y_down"):
        front.invert_yaxis()
    # Side view is a schematic, while the front view has physical coordinates.
    side.set(xlim=(0, 1), ylim=(0, 1))
    side.axis("off")
    side.text(.5, .96, "平面波照明 · 输入面 z = 0", ha="center", weight="bold", fontsize=12)
    color = "#404b58" if geometric else "#a6d2e6"
    if geometric:
        for y, h in ((.33, .13), (.58, .13)):
            side.add_patch(Rectangle((.49, y), .045, h, facecolor=color))
    else:
        side.add_patch(Rectangle((.49, .33), .045, .38, facecolor=color, edgecolor="#42576c"))
    for y in (.4, .52, .64):
        side.annotate("", (.475, y), (.06, y), arrowprops={"arrowstyle": "->", "color": "#dc744d", "lw": 1.8})
        if not geometric or y==.52:
            side.annotate("", (.91, y), (.55, y), arrowprops={"arrowstyle": "->", "color": "#3d9aac", "lw": 1.8})
    incident_label = "第一照明模态\n"+r"$U_{\mathrm{inc},0}=e^{i2\pi f_0x}$" if result.parameters.get("workflow_variant")=="coherence" else "入射场\n振幅 = 1，相位 = 0"
    side.text(.26, .27, incident_label, ha="center", fontsize=10)
    model = model_for(result)
    side.text(.75, .27, "屏后 / 实际输入复场", ha="center", fontsize=10)
    side.text(.5, .205, model.short_field, ha="center", fontsize=8)
    side.text(.5, .11, description, ha="center", va="center", fontsize=10, color="#354b61", linespacing=1.6)
    if result.module=="holography":
        side.text(.5, .015, "透明片后另施加有限 NA 带宽 B；\n下一组振幅/相位图显示带限后的实际输入场。", ha="center", fontsize=9, color="#52657a")
    fig.suptitle("① 物理输入示意图 · "+title, fontsize=16)
    return fig


def field_figure(result, mode="both"):
    from .reporting import plt, image
    plane = result.planes["object"]
    amplitude, phase = input_maps(plane)
    count = 2 if mode=="both" else 1
    fig, axes = plt.subplots(1, count, figsize=(12, 6.6) if count==2 else (6.8, 6.5), constrained_layout=True, squeeze=False)
    items = [(amplitude, r"输入复振幅的振幅 $|U_0|$", False), (phase, r"输入复振幅的相位 $\arg(U_0)$", True)]
    chosen = items if mode=="both" else [items[0 if mode=="amplitude" else 1]]
    for ax, (data, title, is_phase) in zip(axes.flat, chosen):
        image(ax, data, plane.x, title, phase=is_phase)
        _label_axes(ax, _view_limit(result))
        if result.parameters.get("display_image_y_down"):
            ax.invert_yaxis()
    fig.suptitle("② 输入复场 · "+input_description(result), fontsize=14)
    model = model_for(result)
    fig.supxlabel(model.short_field+"\n"+f"实际网格：{len(plane.x)} × {len(plane.x)}，dx = {(plane.x[1]-plane.x[0])*1e6:g} μm；完整函数见输入表达式图。", fontsize=9)
    return fig


def export_input_visuals(result, directory):
    from .reporting import plt
    directory = Path(directory)
    scene_dir, field_dir = directory/"01_输入图", directory/"02_输入复振幅"
    scene_dir.mkdir(parents=True, exist_ok=True)
    field_dir.mkdir(parents=True, exist_ok=True)
    scene = physical_input_figure(result)
    scene.savefig(scene_dir/"输入示意图.png")
    scene.savefig(scene_dir/"输入示意图.svg")
    plt.close(scene)
    scene.clear()
    pictures = ["01_输入图/输入示意图.png"]
    formulas = model_figure(result)
    formulas.savefig(scene_dir/"透过率与复振幅表达式.png", bbox_inches="tight", pad_inches=.15)
    formulas.savefig(scene_dir/"透过率与复振幅表达式.svg", bbox_inches="tight", pad_inches=.15)
    plt.close(formulas)
    formulas.clear()
    for mode, name in (("amplitude", "振幅"), ("phase", "相位"), ("both", "振幅与相位")):
        fig = field_figure(result, mode)
        fig.savefig(field_dir/(name+".png"))
        plt.close(fig)
        fig.clear()
        gc.collect()
        pictures.append("02_输入复振幅/"+name+".png")
    amplitude, phase = input_maps(result.planes["object"])
    np.savez_compressed(field_dir/"输入复场.npz", field=result.planes["object"].field,
                        x_m=result.planes["object"].x, amplitude=amplitude,
                        phase_rad=np.angle(result.planes["object"].field), phase_defined=~np.ma.getmaskarray(phase))
    pictures.append("01_输入图/透过率与复振幅表达式.png")
    return pictures
