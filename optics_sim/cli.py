import argparse
from dataclasses import fields
from pathlib import Path
from datetime import datetime
import json
import sys
from .diffraction import DiffractionConfig, run as diffraction
from .talbot import TalbotConfig, run as talbot
from .holography import HolographyConfig, run as holography
from .abbe import AbbeConfig,FILTER_LABELS,run as abbe
from .paths import OUTPUTS_DIR, VALIDATION_DIR

MODULES = {"diffraction": (DiffractionConfig, diffraction, "衍射计算"),
           "talbot": (TalbotConfig, talbot, "泰伯效应"),
           "holography": (HolographyConfig, holography, "离轴全息"),
           "abbe":(AbbeConfig,abbe,"阿贝成像")}


def load_config(path=None):
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or set(data)-{"shared", *MODULES}:
        raise ValueError("配置顶层只允许 shared / diffraction / talbot / holography / abbe。")
    shared = data.get("shared", {})
    if not isinstance(shared, dict) or set(shared)-{"n", "wavelength", "backend", "seed"}:
        raise ValueError("shared 只允许 n / wavelength / backend / seed。")
    return data


def config_for(name, data, overrides=None):
    klass = MODULES[name][0]
    local = data.get(name, {})
    if not isinstance(local, dict):
        raise ValueError(f"{name} 配置必须为对象。")
    values = {**data.get("shared", {}), **local, **(overrides or {})}
    unknown = set(values)-{f.name for f in fields(klass)}
    if unknown:
        raise ValueError(f"{name} 存在未知参数：{', '.join(sorted(unknown))}")
    if "image_path" in values and values["image_path"] is not None:
        values["image_path"] = str(values["image_path"])
    return klass(**values)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="衍射/泰伯/离轴全息/阿贝成像模拟；配置文件采用SI单位，阿贝4f变换使用SciPy FFT。")
    p.add_argument("module", nargs="?", default="gui", choices=["gui", "all", "validate", *MODULES])
    p.add_argument("--config", type=Path, help="读取分模块 JSON 配置")
    p.add_argument("--output", type=Path, help="输出目录（默认项目 outputs/时间戳；validate 默认 outputs/validation）")
    p.add_argument("--n", type=int)
    p.add_argument("--backend", choices=["lightpipes", "scipy"])
    p.add_argument("--wavelength-nm", type=float)
    p.add_argument("--dx-um", type=float)
    p.add_argument("--z-m", type=float)
    p.add_argument("--width-um", type=float)
    p.add_argument("--object", choices=["circle", "square", "slit"])
    p.add_argument("--method", choices=["fresnel", "angular", "fraunhofer", "fresnel_one_fft"])
    p.add_argument("--image", type=Path)
    p.add_argument("--image-mode", choices=["amplitude", "phase"])
    p.add_argument("--cdi-iterations", type=int)
    p.add_argument("--period-um", type=float)
    p.add_argument("--z-steps", type=int)
    p.add_argument("--theta-deg", type=float)
    p.add_argument("--bandwidth", type=float, help="物光圆形带宽 B，单位 m⁻¹")
    p.add_argument("--filter",dest="filter_kind",choices=list(FILTER_LABELS))
    p.add_argument("--f1-m",type=float)
    p.add_argument("--f2-m",type=float)
    p.add_argument("--na",dest="numerical_aperture",type=float)
    p.add_argument("--cutoff",type=float,help="阿贝截止频率，m⁻¹；高斯滤波时为σ")
    p.add_argument("--inner-cutoff",type=float,help="环形带通内截止频率，m⁻¹")
    p.add_argument("--slit-width",type=float,help="方向狭缝的频率宽度，m⁻¹")
    p.add_argument("--angle-deg",type=float)
    p.add_argument("--phase-radius",type=float,help="中心相位片频率半径，m⁻¹")
    p.add_argument("--phase-shift",type=float,help="中心相位片延迟，rad")
    p.add_argument("--filter-amplitude",type=Path)
    p.add_argument("--filter-phase",type=Path)
    p.add_argument("--image-samples",type=int)
    args = p.parse_args(argv)
    try:
        data = load_config(args.config)
        if args.module == "gui":
            from .gui import launch
            launch(data)
            return
        default_output = VALIDATION_DIR if args.module=="validate" else OUTPUTS_DIR/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = (args.output or default_output).resolve()
        if args.module == "validate":
            from .validation import validate
            report = validate(output)
            for check in report["checks"]:
                print(f"{'PASS' if check['passed'] else 'FAIL'} {check['name']}: {check['value']:.6g}", flush=True)
            print(f"验证报告：{output/'validation.json'}")
            if not report["passed"]:
                raise SystemExit(1)
            return
        from .reporting import export_result, export_index
        names = list(MODULES) if args.module=="all" else [args.module]
        if args.module=="all" and args.dx_um is not None:
            raise ValueError("all 模式下请在配置中为各模块分别设 dx（泰伯采样不同）。")
        reports = []
        for name in names:
            klass, run, label = MODULES[name]
            overrides = {}
            mapping = {"n": ("n", 1), "backend": ("backend", 1), "wavelength_nm": ("wavelength", 1e-9), "dx_um": ("dx", 1e-6)}
            per_module = {
                "diffraction": {"z_m": ("z", 1), "width_um": ("width", 1e-6), "object": ("kind", 1),
                                "method": ("method", 1), "cdi_iterations": ("cdi_iterations", 1),
                                "image": ("image_path", 1), "image_mode": ("image_mode", 1)},
                "talbot": {"period_um": ("period", 1e-6), "z_steps": ("z_steps", 1)},
                "holography": {"z_m": ("z", 1), "theta_deg": ("theta_deg", 1), "bandwidth": ("bandwidth", 1),
                               "image": ("image_path", 1), "image_mode": ("image_mode", 1)},
                "abbe":{"image":("image_path",1),"image_mode":("image_mode",1),"image_samples":("image_samples",1),
                        "filter_kind":("filter_kind",1),"f1_m":("f1",1),"f2_m":("f2",1),
                        "numerical_aperture":("numerical_aperture",1),"cutoff":("cutoff",1),"inner_cutoff":("inner_cutoff",1),
                        "slit_width":("slit_width",1),"angle_deg":("angle_deg",1),"phase_radius":("phase_radius",1),
                        "phase_shift":("phase_shift",1),"filter_amplitude":("filter_amplitude_path",1),"filter_phase":("filter_phase_path",1)}}
            mapping.update(per_module[name])
            for flag, (key, scale) in mapping.items():
                value = getattr(args, flag)
                if value is not None:
                    overrides[key] = value*scale if isinstance(value, (int, float)) else str(value)
            # Single-module invocations should not silently accept irrelevant flags.
            if args.module != "all":
                applicable = set(mapping)|{"module", "config", "output"}
                invalid = [k for k, v in vars(args).items() if v is not None and k not in applicable]
                if invalid:
                    raise ValueError(f"{name} 不支持参数：{', '.join(invalid)}")
            config = config_for(name, data, overrides)
            engine="SciPy FFT" if name=="abbe" else config.backend
            print(f"计算 {label} · {config.n}×{config.n} · {engine}", flush=True)
            result = run(config)
            report = export_result(result, output/name)
            reports.append((name, label))
            print(f"报告：{report}", flush=True)
        export_index(output, reports)
        print(f"实验入口：{output/'index.html'}", flush=True)
    except (ValueError, OSError, RuntimeError) as exc:
        p.error(str(exc))
