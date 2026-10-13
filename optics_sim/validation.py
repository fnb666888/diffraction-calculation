"""Independent physical checks, including continuous references and grid convergence."""
from pathlib import Path
import json
import numpy as np
from .core import Grid, propagate, far_field, power
from .objects import aperture, demo_object, grating
from .diffraction import square_fresnel, airy_intensity, cdi_demo
from .holography import HolographyConfig, run as run_holography, record_phase_steps, reconstruct_phase_steps
from .metrics import relative_error, similarity, align_global_phase


def validate(directory):
    checks = []
    def check(name, value, limit, direction="max"):
        passed = value <= limit if direction=="max" else value >= limit
        checks.append({"name": name, "value": float(value), "limit": limit, "direction": direction, "passed": bool(passed)})
    g = Grid(256, 5e-6)
    obj = demo_object(g)
    for z in (.01, -.01):
        lp = propagate(obj, g, z)
        sp = propagate(obj, g, z, backend="scipy")
        check(f"LightPipes/Scipy 复场一致性 z={z}m", relative_error(align_global_phase(lp, sp), sp), 1e-7)
    out = propagate(obj, g, .02)
    check("菲涅耳往返复场误差", relative_error(propagate(out, g, -.02), obj), 1e-10)
    check("同采样传播功率守恒", abs(power(out, g.dx)/power(obj, g.dx)-1), 1e-10)
    f = far_field(obj, g, .1)
    check("远场坐标缩放后功率守恒", abs(power(f.field, f.x[1]-f.x[0])/power(obj, g.dx)-1), 1e-10)
    disk = aperture(Grid(), "circle", 200e-6)
    far = far_field(disk, Grid(), .1)
    line = far.intensity[256]
    line /= line.max()
    ref = airy_intensity(far.x, 200e-6, g.wavelength, .1)
    roi = np.abs(far.x)<5*g.wavelength*.1/200e-6
    check("圆孔 Airy 解析截面", relative_error(line[roi], ref[roi]), .015)
    # Fixed physical window, refine object pixels, compare to a continuous integral.
    convergence = []
    for n in (256, 512, 1024):
        fine = Grid(n, 2.56e-3/n)
        u0 = aperture(fine, "square", 200e-6)
        numeric = propagate(u0, fine, .02)
        exact = square_fresnel(fine.x, 200e-6, fine.wavelength, .02)
        region = np.abs(fine.x)<300e-6
        ix = np.ix_(region, region)
        err = relative_error(np.abs(numeric[ix])**2, np.abs(exact[ix])**2)
        convergence.append({"n": n, "dx_m": fine.dx, "square_intensity_relative_error": err})
    check("方孔连续 Fresnel 积分（细网格）", convergence[-1]["square_intensity_relative_error"], .005)
    check("方孔细化后误差/粗网格误差", convergence[-1]["square_intensity_relative_error"]/convergence[0]["square_intensity_relative_error"], .25)
    # Explicit zero-padding study, fixed dx and physical object; compare against a
    # larger grid over exactly the same central physical crop.
    padding_fields = []
    for n in (256, 512, 1024):
        padded = Grid(n, 5e-6)
        u = propagate(aperture(padded, "square", 200e-6), padded, .02)
        center = n//2
        padding_fields.append(u[center-60:center+61, center-60:center+61])
    padding_errors = [relative_error(u, padding_fields[-1]) for u in padding_fields[:-1]]
    check("补零512与1024中心复场一致性", padding_errors[1], .002)
    check("增加补零后中心误差下降", padding_errors[1]/max(padding_errors[0], 1e-30), .8)
    tg = Grid(256, 1.25e-6)
    gr = grating(tg)
    zt = 2*(20e-6)**2/tg.wavelength
    full = propagate(gr, tg, zt)
    half = propagate(gr, tg, zt/2)
    shifted = np.roll(gr, 8, axis=1)
    check("泰伯整距强度相关性", similarity(np.abs(full)**2, np.abs(gr)**2), .99, "min")
    check("泰伯半距平移强度相关性", similarity(np.abs(half)**2, np.abs(shifted)**2), .99, "min")
    check("泰伯整距复场误差", relative_error(full, gr), 1e-7)
    rng = np.random.default_rng(17)
    arbitrary = rng.normal(size=(48, 48))+1j*rng.normal(size=(48, 48))
    recovered = reconstruct_phase_steps(record_phase_steps(arbitrary, 1.3), 1.3)
    check("四步相移任意复场恢复（含相位符号）", relative_error(recovered, arbitrary), 1e-12)
    holo = run_holography(HolographyConfig(n=256))
    for key, name, limit in (("virtual_intensity_similarity", "离轴全息虚像相关性", .99),):
        check(name, holo.metrics[key], limit, "min")
    for key in ("virtual_complex_relative_error", "twin_conjugate_field_relative_error", "phase_shift_image_relative_error",
                "common_path_image_relative_error", "reverse_reference_conjugate_error"):
        check(key, holo.metrics[key], 1e-8)
    cm, _ = cdi_demo(iterations=250)
    check("HIO 实测强度振幅残差", cm["hio_fourier_amplitude_residual"], 1e-4)
    check("HIO 歧义对齐后物体误差", cm["hio_aligned_complex_relative_error"], 1e-3)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"passed": all(c["passed"] for c in checks), "checks": checks,
              "sampling_convergence": convergence,
              "padding_convergence": {"n": [256, 512, 1024], "relative_to_1024": padding_errors+[0.]},
              "note": "阈值针对这里声明的固定基准参数；任意用户参数需要重新检查。"}
    (directory/"validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    from .reporting import plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    ax[0].loglog([c["dx_m"]*1e6 for c in convergence], [c["square_intensity_relative_error"] for c in convergence], "o-")
    ax[0].set(title="方孔连续解析解 · 采样收敛", xlabel="dx (μm)", ylabel="强度相对误差")
    ax[1].semilogy([256, 512], padding_errors, "o-")
    ax[1].set(title="固定 dx · 补零收敛", xlabel="每边像素 n", ylabel="相对1024中心复场误差")
    fig.savefig(directory/"convergence.png", dpi=160)
    plt.close(fig)
    return report
