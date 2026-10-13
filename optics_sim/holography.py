"""Off-axis recording/reconstruction, four-step phase shifting and screen compensation."""
from dataclasses import dataclass, asdict
import numpy as np
from .core import Grid, Plane, propagate, fft2c, ifft2c
from .objects import demo_object, image_object, phase_screen
from .metrics import similarity, relative_error, phase_rmse
from .results import Result


@dataclass
class HolographyConfig:
    n: int = 512
    dx: float = 5e-6
    wavelength: float = 632.8e-9
    z: float = .1
    theta_deg: float = 1.5
    bandwidth: float = 8000.
    reference_amplitude: float = 1.5
    snap_carrier: bool = True
    strict_sampling: bool = True
    backend: str = "lightpipes"
    image_path: str | None = None
    image_mode: str = "amplitude"
    image_samples: int = 128
    screen_rms: float = 1.2
    seed: int = 7


def bandlimit(field, grid, bandwidth):
    fx, fy = np.meshgrid(grid.frequency, grid.frequency)
    mask = fx**2+fy**2 <= bandwidth**2
    return ifft2c(fft2c(field)*mask), mask


def carrier_conditions(grid, carrier, bandwidth):
    nyquist = 1/(2*grid.dx)
    # DC autocorrelation radius is 2B; the object sideband radius is B.
    return {"carrier_per_m": float(carrier), "object_bandwidth_per_m": float(bandwidth),
            "nyquist_per_m": float(nyquist),
            "dc_separation_margin_per_m": float(abs(carrier)-3*bandwidth),
            "nyquist_margin_per_m": float(nyquist-abs(carrier)-bandwidth),
            "dc_fits": bool(2*bandwidth < nyquist),
            "separated": bool(abs(carrier)>3*bandwidth and abs(carrier)+bandwidth<nyquist and 2*bandwidth<nyquist),
            "fringe_samples": float(1/(abs(carrier)*grid.dx)) if carrier else None}


def record(object_wave, reference):
    return np.abs(object_wave+reference)**2


def reconstruct_off_axis(intensity, reference, grid, bandwidth, amplitude=1.):
    """Receives only real recorded intensity and known reference/calibration."""
    if amplitude <= 0:
        raise ValueError("参考振幅必须为正。")
    return bandlimit(np.asarray(intensity)*reference/amplitude**2, grid, bandwidth)[0]


def record_phase_steps(object_wave, amplitude=1., common_phase=0.):
    if amplitude <= 0:
        raise ValueError("参考振幅必须为正。")
    return np.asarray([record(object_wave, amplitude*np.exp(1j*(p+common_phase)))
                       for p in (0, np.pi/2, np.pi, 3*np.pi/2)])


def reconstruct_phase_steps(frames, amplitude=1.):
    frames = np.asarray(frames)
    if frames.ndim != 3 or frames.shape[0] != 4 or amplitude <= 0:
        raise ValueError("四步相移需要4幅二维强度图和正参考振幅。")
    # R=A exp(+i phi); this sign is fixed by the recording convention.
    return ((frames[0]-frames[2])+1j*(frames[1]-frames[3]))/(4*amplitude)


def run(config=None):
    c = config or HolographyConfig()
    g = Grid(c.n, c.dx, c.wavelength)
    if not np.isfinite(c.z) or c.z <= 0 or not 0 < c.bandwidth < 1/(2*g.dx) or c.reference_amplitude <= 0 or abs(c.theta_deg)>=90:
        raise ValueError("距离/参考振幅必须为正，带宽须在Nyquist内，参考角须在 (-90°,90°)。")
    raw = image_object(g, c.image_path, c.image_samples, c.image_mode) if c.image_path else demo_object(g)
    if not np.any(np.abs(raw)>0):
        raise ValueError("输入物体振幅全为0，请选择非全黑图片或改用相位模式。")
    # Explicit aperture in frequency: ideal finite-NA model, not a hidden filter
    # inserted after validation. The unfiltered target and retained energy are saved.
    obj, pupil = bandlimit(raw, g, c.bandwidth)
    requested = np.sin(np.radians(c.theta_deg))/g.wavelength
    carrier = round(requested*g.size)/g.size if c.snap_carrier else requested
    conditions = carrier_conditions(g, carrier, c.bandwidth)
    if c.strict_sampling and not conditions["separated"]:
        raise ValueError(f"离轴频谱未分离或超Nyquist：需要 3B < |fc| < fNyquist-B。当前 fc={carrier:.1f}/m，B={c.bandwidth:.1f}/m。可调整角度、dx或B。")
    object_wave = propagate(obj, g, c.z, backend=c.backend)
    X, Y = g.xy
    reference = c.reference_amplitude*np.exp(2j*np.pi*carrier*X)
    hologram = record(object_wave, reference)  # the only measurement passed below
    recovered_wave = reconstruct_off_axis(hologram, reference, g, c.bandwidth, c.reference_amplitude)
    virtual = propagate(recovered_wave, g, -c.z, backend=c.backend)
    # Isolate the conjugate order by illuminating with conjugate reference.
    conjugate_wave = reconstruct_off_axis(hologram, np.conj(reference), g, c.bandwidth, c.reference_amplitude)
    twin = propagate(conjugate_wave, g, c.z, backend=c.backend)
    transmission = hologram/hologram.max()
    raw_virtual = propagate(reference*transmission, g, -c.z, backend=c.backend)
    raw_real = propagate(np.conj(reference)*transmission, g, c.z, backend=c.backend)
    frames = record_phase_steps(object_wave, c.reference_amplitude)
    phase_wave = reconstruct_phase_steps(frames, c.reference_amplitude)
    phase_image = propagate(phase_wave, g, -c.z, backend=c.backend)
    # Known thin phase screen exactly at the recording plane. This does not claim
    # arbitrary volume/unknown-screen compensation from a single hologram.
    w = phase_screen(g, c.seed, c.screen_rms)
    distorted = object_wave*np.exp(1j*w)
    baseline = propagate(distorted, g, -c.z, backend=c.backend)
    common_frames = record_phase_steps(distorted, c.reference_amplitude, common_phase=w)
    common_wave = reconstruct_phase_steps(common_frames, c.reference_amplitude)
    common_image = propagate(common_wave, g, -c.z, backend=c.backend)
    distorted_frames = record_phase_steps(distorted, c.reference_amplitude)
    measured_distorted = reconstruct_phase_steps(distorted_frames, c.reference_amplitude)
    corrected = propagate(measured_distorted*np.exp(-1j*w), g, -c.z, backend=c.backend)
    reverse = propagate(np.conj(measured_distorted)*np.exp(1j*w), g, c.z, backend=c.backend)
    metrics = {"sampling": conditions,
               "requested_theta_deg": c.theta_deg,
               "actual_theta_deg": float(np.degrees(np.arcsin(carrier*g.wavelength))),
               "optical_pupil_retained_power": float(np.sum(np.abs(obj)**2)/np.sum(np.abs(raw)**2)),
               "virtual_intensity_similarity": similarity(np.abs(virtual)**2, np.abs(obj)**2),
               "virtual_complex_relative_error": relative_error(virtual, obj),
               "virtual_phase_rmse_rad": phase_rmse(virtual, obj),
               "twin_intensity_similarity": similarity(np.abs(twin)**2, np.abs(obj)**2),
               "twin_conjugate_field_relative_error": relative_error(twin, np.conj(obj)),
               "phase_shift_field_relative_error": relative_error(phase_wave, object_wave),
               "phase_shift_image_relative_error": relative_error(phase_image, obj),
               "aberrated_image_relative_error": relative_error(baseline, obj),
               "common_path_image_relative_error": relative_error(common_image, obj),
               "known_screen_corrected_relative_error": relative_error(corrected, obj),
               "reverse_reference_conjugate_error": relative_error(reverse, np.conj(obj)),
               "common_path_frame_invariance_error": relative_error(common_frames, frames)}
    min_fc = 3*c.bandwidth
    max_fc = 1/(2*g.dx)-c.bandwidth
    metrics["theoretical_carrier_interval_per_m"] = [min_fc, max_fc]
    metrics["theoretical_min_separation_angle_deg"] = float(np.degrees(np.arcsin(min(1., min_fc*g.wavelength))))
    # Quantitative angle and pixel-size scans with a fixed physical window. All
    # low-resolution inputs obey the same finite-NA bandwidth model.
    scan_carriers = np.arange(0, int((1/(2*g.dx))*g.size), max(1, int((1/(2*g.dx))*g.size)//16))/g.size
    angle_errors, angle_valid = [], []
    for fc in scan_carriers:
        r = c.reference_amplitude*np.exp(2j*np.pi*fc*X)
        u = reconstruct_off_axis(record(object_wave, r), r, g, c.bandwidth, c.reference_amplitude)
        angle_errors.append(relative_error(u, object_wave))
        angle_valid.append(carrier_conditions(g, fc, c.bandwidth)["separated"])
    pixel_counts = sorted(set([max(32, g.n//4), max(32, g.n//2), g.n]))
    pixel_errors, pixel_valid = [], []
    for count in pixel_counts:
        small = Grid(count, g.size/count, g.wavelength)
        source, _ = bandlimit(demo_object(small), small, c.bandwidth)
        propagated = propagate(source, small, c.z, backend=c.backend)
        sx, _ = small.xy
        r = c.reference_amplitude*np.exp(2j*np.pi*carrier*sx)
        rec = reconstruct_off_axis(record(propagated, r), r, small, c.bandwidth, c.reference_amplitude)
        pixel_errors.append(relative_error(rec, propagated))
        pixel_valid.append(carrier_conditions(small, carrier, c.bandwidth)["separated"])
    planes = {name: Plane(value, g.x, name) for name, value in
              (("raw_object", raw), ("object", obj), ("object_wave", object_wave),
               ("recovered_wave", recovered_wave), ("virtual_image", virtual), ("twin_image", twin),
               ("raw_virtual", raw_virtual), ("raw_real", raw_real), ("phase_shift_image", phase_image),
               ("aberrated_image", baseline), ("common_path_image", common_image),
               ("corrected_image", corrected), ("reverse_image", reverse))}
    arrays = {"hologram_intensity_only": hologram, "hologram_transmission": transmission,
              "reference_wave": reference, "pupil": pupil, "hologram_spectrum": np.abs(fft2c(hologram))**2,
              "frequency_per_m": g.frequency, "phase_step_intensities": frames,
              "screen_phase_rad": w, "common_path_intensities": common_frames,
              "angle_scan_deg": np.degrees(np.arcsin(scan_carriers*g.wavelength)),
              "angle_scan_errors": np.asarray(angle_errors), "angle_scan_separated": np.asarray(angle_valid),
              "pixel_scan_n": np.asarray(pixel_counts), "pixel_scan_errors": np.asarray(pixel_errors),
              "pixel_scan_separated": np.asarray(pixel_valid)}
    notes = ["记录后重建仅接收实数强度、已知参考波及带宽；真值不参与重建。",
             "虚像定量结果来自分离后的数字一级衍射波；原始光学照明重建另存，包含0级和其它级。",
             "对二维透明片，共轭实像和原像可有相同强度，相位符号相反。凹凸/视差反转需要三维物体；不能用低强度相关性作为孪生像必要条件。",
             "光学有限NA模型将物体限制到B带宽，raw_object为滤波前真值；保留功率和滤波后真值分别导出。",
             "默认载频取最近FFT格点，避免有限周期窗口的非整周期泄漏；关闭snap_carrier可研究泄漏。",
             "四步记录约定 R=A exp(+iφ)，虚部取 Iπ/2-I3π/2，四幅强度即可同时恢复振幅和相位。",
             "像差演示限定为记录面处的薄相位屏；共光路相消、已知屏逆补偿、同屏共轭传播分别计算。",
             "像素扫描采用同窗口的合成靶标；分离条件依赖带宽/像素间隔，4N是带宽预算而非普遍固定网格倍率。"]
    return Result("holography", asdict(c), metrics, planes, arrays, notes)
