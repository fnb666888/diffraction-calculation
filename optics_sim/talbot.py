"""Talbot module: periodic/finite gratings, fractional planes, TLI and coherence."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.ndimage import gaussian_filter1d
from .core import Grid, Plane, propagate, fft2c, edge_metrics
from .objects import grating, grating_line
from .metrics import similarity, relative_error
from .results import Result


@dataclass
class TalbotConfig:
    n: int = 512
    dx: float = 1.25e-6
    wavelength: float = 632.8e-9
    period: float = 20e-6
    duty: float = .5
    kind: str = "binary"
    phase_depth: float = np.pi
    dimensions: int = 1
    finite_fraction: float = 1.
    z_steps: int = 100
    z_periods: float = 2.
    backend: str = "lightpipes"
    curvature_radius: float = .03
    analyzer_mismatch: float = .04
    coherence_modes: int = 15
    coherence_length: float = 15e-6
    seed: int = 7


def run(config=None):
    c = config or TalbotConfig()
    g = Grid(c.n, c.dx, c.wavelength)
    if c.z_steps < 2 or c.z_periods <= 0 or c.curvature_radius == 0 or c.coherence_modes < 1 or c.coherence_length <= 0:
        raise ValueError("扫描至少2个平面，距离倍数/相干长度须为正，曲率半径不能为零，相干模态数须≥1。")
    if not 0 < 1+c.analyzer_mismatch:
        raise ValueError("分析光栅周期必须为正。")
    zt = 2*c.period**2/g.wavelength
    period_samples = c.period/g.dx
    if not np.isclose(period_samples, round(period_samples), atol=1e-8) or not np.isclose(g.size/c.period, round(g.size/c.period), atol=1e-8):
        raise ValueError("周期校核要求每周期为整数像素，且窗口包含整数周期；请调整 n/dx/period。")
    # The periodic ideal has no padding; finite illumination is computed separately.
    ideal = grating(g, c.period, c.kind, c.duty, c.phase_depth, c.dimensions)
    u0 = grating(g, c.period, c.kind, c.duty, c.phase_depth, c.dimensions, c.finite_fraction)
    input_i = np.abs(ideal)**2
    full = propagate(ideal, g, zt, backend=c.backend)
    twice = propagate(ideal, g, 2*zt, backend=c.backend)
    half = propagate(ideal, g, zt/2, backend=c.backend)
    # Exact Fourier shift also handles an odd number of samples per grating period.
    fx, fy = np.meshgrid(g.frequency, g.frequency)
    shift_phase = np.exp(-2j*np.pi*(fx + (fy if c.dimensions==2 else 0))*c.period/2)
    from .core import ifft2c
    shifted = ifft2c(fft2c(ideal)*shift_phase)
    metrics = {"talbot_distance_m": zt, "period_samples": period_samples,
               "periods_in_window": g.size/c.period,
               "zT_intensity_similarity": similarity(np.abs(full)**2, input_i),
               "2zT_intensity_similarity": similarity(np.abs(twice)**2, input_i),
               "half_shift_intensity_similarity": similarity(np.abs(half)**2, np.abs(shifted)**2),
               "half_unshifted_similarity": similarity(np.abs(half)**2, input_i),
               "fixed_coordinate_half_period_relative_error": relative_error(np.abs(half)**2, input_i)}
    notes = ["zT=2d²/λ；zT/2 自像横移 d/2（二维光栅沿 x、y 各移 d/2）。固定坐标强度通常以 zT 为周期。",
             "周期网格的 FFT 周期边界是理想无限光栅模型；有限照明结果另算，边缘强度不适用于无限光栅判据。",
             "纯相位光栅输入强度为常数，强度相关性不能独立证明复场自像。"]
    metrics["zT_complex_relative_error"] = relative_error(full, ideal)
    metrics["half_shift_complex_relative_error"] = relative_error(half, shifted)
    planes = {"object": Plane(u0, g.x, "grating")}
    for name, ratio in (("eighth", .125), ("quarter", .25), ("half", .5), ("three_quarters", .75), ("full", 1.), ("twice", 2.)):
        planes[name] = Plane(propagate(u0, g, ratio*zt, backend=c.backend), g.x, f"z={ratio}zT")
    zs = np.linspace(0, c.z_periods*zt, c.z_steps)
    carpet = np.asarray([np.abs(propagate(u0, g, z, backend=c.backend)[g.n//2])**2 for z in zs])
    # Far-field order check uses a 1D x slice (not the constant y column).
    line = grating_line(g, c.period, c.kind, c.duty, c.phase_depth)
    from scipy import fft
    spectrum = fft.fftshift(fft.fft(fft.ifftshift(line)))/g.n
    orders = np.arange(-min(7, int(period_samples/2)-1), min(7, int(period_samples/2)-1)+1)
    positions = orders/c.period
    indices = np.array([np.argmin(np.abs(g.frequency-f)) for f in positions])
    order_power = np.abs(spectrum[indices])**2
    coeff = c.duty*np.sinc(orders*c.duty)
    if c.kind == "sine":
        coeff = np.where(orders==0, .5, np.where(np.abs(orders)==1, .25, 0))
    elif c.kind == "phase":
        coeff = coeff*(np.exp(1j*c.phase_depth)-1) + (orders==0)
    analytic_power = np.abs(coeff)**2 if c.kind != "phase_sine" else None
    metrics["order_frequency_max_error_per_m"] = float(np.max(np.abs(g.frequency[indices]-positions)))
    propagating = np.abs(orders*g.wavelength/c.period) <= 1
    theta = np.full(orders.shape, 0., dtype=float)
    theta[propagating] = np.degrees(np.arcsin(orders[propagating]*g.wavelength/c.period))
    metrics["order_angles_deg"] = theta.tolist()
    if c.kind == "binary" and c.duty == .5:
        even = (orders!=0) & (orders%2==0)
        metrics["even_orders_relative_to_zero"] = float(order_power[even].max()/max(order_power[orders==0][0], 1e-30)) if even.any() else 0.
    if analytic_power is not None:
        metrics["order_power_relative_error_vs_continuous_grating"] = relative_error(order_power, analytic_power)
    finite = grating(g, c.period, c.kind, c.duty, c.phase_depth, c.dimensions, .75)
    finite_full = propagate(finite, g, zt, backend=c.backend)
    roi = np.abs(g.x)<g.size*.18
    select = np.ix_(roi, roi)
    metrics["finite_central_zT_similarity"] = similarity(np.abs(finite_full[select])**2, np.abs(finite[select])**2)
    metrics["finite_boundary"] = edge_metrics(finite_full)
    planes["finite_full"] = Plane(finite_full, g.x, "finite grating zT")
    X, Y = g.xy
    curved = u0*np.exp(1j*np.pi/(g.wavelength*c.curvature_radius)*(X**2+Y**2))
    planes["curved_quarter"] = Plane(propagate(curved, g, zt/4, backend=c.backend), g.x, "curved zT/4")
    # Slight analyzer-period mismatch produces a resolvable Moire envelope.
    analyzer = grating(g, c.period*(1+c.analyzer_mismatch), "binary", c.duty, dimensions=c.dimensions)
    flat_tli = np.abs(propagate(u0, g, zt, backend=c.backend)*analyzer)**2
    curved_tli = np.abs(propagate(curved, g, zt, backend=c.backend)*analyzer)**2
    blur = max(1, c.period/g.dx*.65)
    flat_profile = gaussian_filter1d(flat_tli[g.n//2], blur)
    curved_profile = gaussian_filter1d(curved_tli[g.n//2], blur)
    # Gaussian angular mixture: <exp(i k Δx)> = exp[-Δx²/(2 lc²)].
    rng = np.random.default_rng(c.seed)
    tilts = rng.normal(0, 1/(2*np.pi*c.coherence_length), c.coherence_modes)
    incoherent = np.zeros_like(input_i)
    for frequency in tilts:
        incoherent += np.abs(propagate(u0*np.exp(2j*np.pi*frequency*X), g, zt, backend=c.backend))**2/c.coherence_modes
    planes["partial_coherence_full"] = Plane(np.sqrt(incoherent).astype(complex), g.x, "incoherent intensity ensemble; no physical phase")
    metrics["coherence_full_image_similarity"] = similarity(incoherent[select], np.abs(u0[select])**2)
    arrays = {"carpet_intensity": carpet, "carpet_z_m": zs, "carpet_x_m": g.x,
              "far_frequency_per_m": g.frequency, "far_order_spectrum": np.abs(spectrum)**2,
              "orders": orders, "order_frequency_per_m": positions, "order_power_numeric": order_power,
              "tli_flat_intensity": flat_tli, "tli_curved_intensity": curved_tli,
              "tli_flat_envelope": flat_profile, "tli_curved_envelope": curved_profile,
              "coherence_tilts_per_m": tilts}
    if analytic_power is not None:
        arrays["order_power_analytic"] = analytic_power
    return Result("talbot", asdict(c), metrics, planes, arrays, notes)
