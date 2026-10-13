"""SI units; arrays indexed [y, x]; centered spatial and frequency coordinates.

LightPipes.Forvard is the default Fresnel engine. Its common carrier phase is
removed so all backends use the slowly varying complex envelope convention.
"""
from dataclasses import dataclass
import numpy as np
from scipy import fft


@dataclass(frozen=True)
class Grid:
    n: int = 512
    dx: float = 5e-6
    wavelength: float = 632.8e-9

    def __post_init__(self):
        if not isinstance(self.n, (int, np.integer)) or self.n < 32 or self.n % 2:
            raise ValueError("n 必须是 ≥32 的偶数（LightPipes FFT 约定）。")
        if not np.isfinite(self.dx) or not np.isfinite(self.wavelength) or min(self.dx, self.wavelength) <= 0:
            raise ValueError("采样间隔和波长必须是有限正数，单位 m。")

    @property
    def size(self):
        return self.n * self.dx

    @property
    def x(self):
        return (np.arange(self.n) - self.n // 2) * self.dx

    @property
    def frequency(self):
        return fft.fftshift(fft.fftfreq(self.n, self.dx))

    @property
    def xy(self):
        return np.meshgrid(self.x, self.x)

    def check_field(self, field):
        a = np.asarray(field, dtype=np.complex128)
        if a.shape != (self.n, self.n) or not np.all(np.isfinite(a)):
            raise ValueError(f"复振幅必须是有限的 {self.n}×{self.n} 数组。")
        return a


@dataclass
class Plane:
    field: np.ndarray
    x: np.ndarray
    method: str

    @property
    def intensity(self):
        return np.abs(self.field) ** 2


def fft2c(a):
    return fft.fftshift(fft.fft2(fft.ifftshift(a), norm="ortho"))


def ifft2c(a):
    return fft.fftshift(fft.ifft2(fft.ifftshift(a), norm="ortho"))


def transfer_function(grid, z, method="fresnel"):
    fx, fy = np.meshgrid(fft.fftfreq(grid.n, grid.dx), fft.fftfreq(grid.n, grid.dx))
    if method == "fresnel":
        return np.exp(-1j * np.pi * grid.wavelength * z * (fx**2 + fy**2))
    if method != "angular":
        raise ValueError("传递函数仅支持 fresnel / angular。")
    q = 1 - grid.wavelength**2 * (fx**2 + fy**2)
    # Project out evanescent waves for a bounded backward operator. For forward
    # propagation retain their physical exponential decay, never a growing branch.
    h = np.zeros_like(q, dtype=complex)
    mask = q >= 0
    # sqrt(q)-1 evaluated without cancellation; removes the common exp(ikz).
    h[mask] = np.exp(1j * 2*np.pi/grid.wavelength * z *
                     ((q[mask]-1) / (np.sqrt(q[mask])+1)))
    if z >= 0:
        h[~mask] = np.exp(-2*np.pi/grid.wavelength * z * np.sqrt(-q[~mask])) * np.exp(-1j * 2*np.pi/grid.wavelength * z)
    return h


def propagate(field, grid, z, method="fresnel", backend="lightpipes"):
    a = grid.check_field(field)
    if not np.isfinite(z):
        raise ValueError("传播距离必须是有限数。")
    if method not in ("fresnel", "angular"):
        raise ValueError("同采样传播仅支持 fresnel / angular；远场使用 far_field。")
    if backend not in ("lightpipes", "scipy"):
        raise ValueError("backend 必须是 lightpipes / scipy。")
    if z == 0:
        return a.copy()
    if method == "fresnel" and backend == "lightpipes":
        try:
            from LightPipes import Begin, Forvard
        except ImportError as exc:
            raise RuntimeError("请安装 requirements.txt，或显式选择 backend='scipy'。") from exc
        wave = Begin(grid.size, grid.wavelength, grid.n)
        wave.field = a.copy()
        out = Forvard(wave, float(z)).field
        # LP 2.1.5 uses this legacy constant and abs(z) for its common phase.
        return out * np.exp(-1j * (2 * 3.141592654 / grid.wavelength * abs(z)))
    return fft.ifft2(fft.fft2(a) * transfer_function(grid, z, method))


def far_field(field, grid, z):
    if not np.isfinite(z) or z <= 0:
        raise ValueError("夫琅禾费距离必须是有限正数。")
    a = grid.check_field(field)
    x = grid.frequency * grid.wavelength * z
    X, Y = np.meshgrid(x, x)
    # The global exp(ikz) is omitted, but dimensional normalization and curvature
    # are retained. Output pixel size differs from the source pixel size.
    u = fft2c(a) * grid.n * grid.dx**2 / (1j * grid.wavelength * z)
    u *= np.exp(1j*np.pi/(grid.wavelength*z) * (X**2+Y**2))
    return Plane(u, x, "fraunhofer")


def far_field_window(field, grid, z, limit, samples=601):
    """Evaluate a dense far-field window with SciPy Zoom FFT, without interpolation."""
    from scipy.signal import zoom_fft
    if not np.isfinite(z) or z <= 0 or not np.isfinite(limit) or limit <= 0:
        raise ValueError("远场距离和观察范围必须为有限正数。")
    if not isinstance(samples, int) or samples < 3:
        raise ValueError("远场观察窗口至少需要3个样本。")
    frequency_limit = limit/(grid.wavelength*z)
    if frequency_limit > 1/(2*grid.dx):
        raise ValueError("远场观察窗口超出输入场的Nyquist频率。")
    a = grid.check_field(field)
    f = np.linspace(-frequency_limit, frequency_limit, samples)
    spectrum = zoom_fft(a, [-frequency_limit, frequency_limit], samples,
                        fs=1/grid.dx, endpoint=True, axis=1)
    spectrum = zoom_fft(spectrum, [-frequency_limit, frequency_limit], samples,
                        fs=1/grid.dx, endpoint=True, axis=0)
    # Zoom FFT indexes the source from zero; our physical origin is at n/2.
    spectrum *= np.exp(2j*np.pi*(grid.n//2)*grid.dx*(f[:, None]+f[None, :]))
    x = f*grid.wavelength*z
    u = spectrum*grid.dx**2/(1j*grid.wavelength*z)
    u *= np.exp(1j*np.pi/(grid.wavelength*z)*(x[:, None]**2+x[None, :]**2))
    return Plane(u, x, "fraunhofer zoom FFT")


def fresnel_one_fft(field, grid, z):
    if not np.isfinite(z) or z <= 0:
        raise ValueError("单 FFT 菲涅耳传播距离必须是有限正数。")
    X, Y = grid.xy
    chirp = np.exp(1j*np.pi/(grid.wavelength*z) * (X**2+Y**2))
    plane = far_field(grid.check_field(field)*chirp, grid, z)
    plane.method = "fresnel_one_fft"
    return plane


def power(field, dx):
    return float(np.sum(np.abs(field)**2) * dx**2)


def edge_metrics(field, fraction=0.05):
    intensity = np.abs(field)**2
    w = max(1, round(intensity.shape[0]*fraction))
    mask = np.ones_like(intensity, dtype=bool)
    mask[w:-w, w:-w] = False
    peak = float(intensity.max())
    return {"edge_peak_ratio": float(intensity[mask].max()/max(peak, 1e-30)),
            "edge_power_fraction": float(intensity[mask].sum()/max(intensity.sum(), 1e-30))}


def sampling_report(grid, width, z):
    if width <= 0 or z == 0:
        raise ValueError("孔径宽度必须为正，传播距离不能为零。")
    nf = (width/2)**2 / (grid.wavelength*abs(z))
    samples = width/grid.dx
    messages = []
    if samples < 16:
        messages.append("孔径内少于16个样本，边缘几何误差可能较大。")
    if samples <= 4*nf:
        messages.append("孔径样本数未满足方案 M > 4 N_F 的经验判据。")
    if width >= grid.size*0.8:
        messages.append("孔径接近窗口边界，请增大网格留出补零。")
    if grid.wavelength*abs(z) > grid.n*grid.dx**2:
        messages.append("传递函数高频相位抽样可能不足；应结合边缘指标、有效带宽及补零收敛检查。")
    return {"fresnel_number": float(nf), "aperture_samples": float(samples),
            "padding_ratio": float(grid.size/width), "warnings": messages}
