"""Diffraction module: four propagation methods, analytic checks and ER/HIO CDI."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.special import fresnel, j1
from .core import Grid, Plane, propagate, far_field, fresnel_one_fft, sampling_report, edge_metrics, power, fft2c, ifft2c
from .objects import aperture, image_object
from .metrics import similarity, relative_error, align_cdi, phase_rmse
from .results import Result


@dataclass
class DiffractionConfig:
    n: int = 512
    dx: float = 5e-6
    wavelength: float = 632.8e-9
    z: float = .02
    width: float = 200e-6
    kind: str = "circle"
    height: float | None = None
    method: str = "fresnel"
    backend: str = "lightpipes"
    image_path: str | None = None
    image_mode: str = "amplitude"
    image_samples: int = 128
    cdi_iterations: int = 400
    seed: int = 7


def square_fresnel(x, width, wavelength, z, height=None):
    """Continuous rectangle integral, no global exp(ikz); coordinates are centred."""
    if z <= 0:
        raise ValueError("解析菲涅耳积分要求 z > 0。")
    def integral(w):
        u1 = np.sqrt(2/(wavelength*z))*(x-w/2)
        u2 = np.sqrt(2/(wavelength*z))*(x+w/2)
        s1, c1 = fresnel(u1)
        s2, c2 = fresnel(u2)
        return (c2-c1)+1j*(s2-s1)
    return np.outer(integral(width if height is None else height), integral(width))/(2j)


def airy_intensity(x, diameter, wavelength, z):
    v = np.pi*diameter*np.abs(x)/(wavelength*z)
    out = np.ones_like(v)
    nonzero = v != 0
    out[nonzero] = (2*j1(v[nonzero])/v[nonzero])**2
    return out


def recover_cdi(intensity, support, iterations=400, algorithm="hio", beta=.9,
                seed=7, positive_real=True):
    """Recovery receives only far-field intensity and prior constraints, never truth.

    Best residual is selected across iterates. HIO finishes with 50 ER polishing
    steps included in the requested iteration budget. General complex objects can
    disable the real/nonnegative prior; convergence then is not guaranteed.
    """
    intensity, support = np.asarray(intensity, float), np.asarray(support, bool)
    if intensity.shape != support.shape or intensity.ndim != 2 or (intensity < 0).any() or not np.all(np.isfinite(intensity)):
        raise ValueError("强度与支集须为同形二维数组，强度必须有限非负。")
    if not support.any() or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("支集不能为空，迭代次数必须为正整数。")
    if algorithm not in ("er", "hio") or not 0 < beta <= 1:
        raise ValueError("算法必须是 er / hio，反馈系数位于 (0,1]。")
    amplitude = np.sqrt(intensity)
    rng = np.random.default_rng(seed)
    u = ifft2c(amplitude*np.exp(1j*rng.uniform(-np.pi, np.pi, amplitude.shape)))
    best, best_residual = u.copy(), np.inf
    history = []
    for step in range(iterations):
        spectrum = fft2c(u)
        projected = ifft2c(amplitude*np.exp(1j*np.angle(spectrum)))
        valid = support & (projected.real >= 0) if positive_real else support
        constrained = projected.real.astype(complex) if positive_real else projected
        er = algorithm == "er" or step >= iterations-min(50, iterations//4)
        u = np.where(valid, constrained, 0 if er else u-beta*projected)
        candidate = np.where(support, np.maximum(u.real, 0) if positive_real else u, 0)
        residual = relative_error(np.abs(fft2c(candidate)), amplitude)
        history.append(residual)
        if residual < best_residual:
            best_residual, best = residual, candidate.copy()
    return best, np.asarray(history)


def cdi_demo(n=128, iterations=400, seed=7):
    # A compact, asymmetric nonnegative object with a deliberately loose support.
    y, x = np.mgrid[:n, :n]
    X, Y = (x-n//2)/n, (y-n//2)/n
    support = (np.abs(X)<.155) & (np.abs(Y)<.14)
    truth = np.zeros((n, n), dtype=complex)
    truth[(X>-.13)&(X<-.05)&(Y>-.12)&(Y<.10)] = .8
    truth[(X>-.07)&(X<.13)&(Y>-.11)&(Y<-.04)] = 1.
    truth[(X>.01)&(X<.11)&(Y>.015)&(Y<.10)] = .5
    intensity = np.abs(fft2c(truth))**2
    er, er_history = recover_cdi(intensity, support, iterations, "er", seed=seed)
    hio, hio_history = recover_cdi(intensity, support, iterations, "hio", seed=seed)
    aligned, error, ambiguity = align_cdi(hio, truth)
    metrics = {"hio_fourier_amplitude_residual": float(hio_history.min()),
               "er_fourier_amplitude_residual": float(er_history.min()),
               "hio_aligned_complex_relative_error": error,
               "hio_intensity_similarity": similarity(np.abs(aligned)**2, np.abs(truth)**2),
               "hio_phase_rmse_rad": phase_rmse(aligned, truth, np.abs(truth)>0),
               "evaluation_alignment": ambiguity, "iterations": iterations,
               "prior": "loose rectangular support + real nonnegative object"}
    arrays = {"cdi_truth": truth, "cdi_intensity_only": intensity, "cdi_support": support,
              "cdi_hio_raw": hio, "cdi_hio_aligned": aligned, "cdi_er": er,
              "cdi_hio_residual": hio_history, "cdi_er_residual": er_history}
    return metrics, arrays


def run(config=None):
    c = config or DiffractionConfig()
    if not np.isfinite(c.z) or c.z <= 0 or c.cdi_iterations < 0:
        raise ValueError("距离须为有限正数，CDI 次数须非负（0关闭）。")
    g = Grid(c.n, c.dx, c.wavelength)
    if c.image_path:
        u0 = image_object(g, c.image_path, c.image_samples, c.image_mode)
        effective_width = c.image_samples*g.dx
    else:
        height = c.height if c.height else (c.width*5 if c.kind=="slit" else c.width)
        u0 = aperture(g, c.kind, c.width, height)
        effective_width = max(c.width, height)
    if not np.any(np.abs(u0)>0):
        raise ValueError("输入物体振幅全为0，请选择非全黑图片或改用相位模式。")
    if c.method == "fraunhofer":
        plane = far_field(u0, g, c.z)
    elif c.method == "fresnel_one_fft":
        plane = fresnel_one_fft(u0, g, c.z)
    else:
        plane = Plane(propagate(u0, g, c.z, c.method, c.backend), g.x, c.method)
    far = far_field(u0, g, c.z)
    dx_out = float(plane.x[1]-plane.x[0])
    metrics = {"sampling": sampling_report(g, effective_width, c.z),
               "boundary": edge_metrics(plane.field),
               "power_ratio": power(plane.field, dx_out)/power(u0, g.dx)}
    if c.method=="fraunhofer" and metrics["sampling"]["fresnel_number"]>=.25:
        metrics["sampling"]["warnings"].append("当前菲涅耳数不够小，自由传播的夫琅禾费近似可能不足；也可将 z 解释为透镜焦距，用作焦面图样。")
    arrays = {}
    # Independent continuous Airy reference, regardless of selected input aperture.
    disk = aperture(g, "circle", c.width)
    far_disk = far_field(disk, g, c.z)
    numerical = far_disk.intensity[g.n//2]
    numerical /= numerical.max()
    analytic = airy_intensity(far_disk.x, c.width, g.wavelength, c.z)
    roi = np.abs(far_disk.x) <= 5*g.wavelength*c.z/c.width
    metrics["airy_profile_relative_error"] = relative_error(numerical[roi], analytic[roi])
    arrays.update(airy_x_m=far_disk.x, airy_numeric=numerical, airy_analytic=analytic)
    # Use scaled single FFT to avoid a circular-window TF result as analytic truth.
    sq = aperture(g, "square", c.width)
    sq_plane = fresnel_one_fft(sq, g, c.z)
    exact = square_fresnel(sq_plane.x, c.width, g.wavelength, c.z)
    roi2 = np.abs(sq_plane.x) <= max(c.width, 3*np.sqrt(g.wavelength*c.z))
    numeric_i, exact_i = sq_plane.intensity, np.abs(exact)**2
    select = np.ix_(roi2, roi2)
    metrics["square_fresnel_intensity_relative_error"] = relative_error(numeric_i[select], exact_i[select])
    arrays.update(square_x_m=sq_plane.x, square_numeric=numeric_i[g.n//2], square_analytic=exact_i[g.n//2])
    if c.cdi_iterations:
        cm, ca = cdi_demo(min(128, c.n), c.cdi_iterations, c.seed)
        metrics["cdi"], arrays = cm, {**arrays, **ca}
    return Result("diffraction", asdict(c), metrics,
                  {"object": Plane(u0, g.x, "input"), "propagated": plane, "far_field": far,
                   "airy_far_field": far_disk}, arrays,
                  ["单FFT输出坐标 x=λzf，和同采样传播的坐标不同，导出数据分别保存。",
                   "far_field 是夫琅禾费近似（或将 z 视为透镜焦距的焦面图样），默认近场 z 不表示实际已经进入远场。",
                   "CDI为单独的非负振幅靶标闭环；真值仅用于迭代结束后的评价。单幅强度存在平移、全局相位及共轭反转歧义。",
                   "边缘指标只是诊断量，不能单独证明无混叠；需要结合补零与采样收敛。"])
