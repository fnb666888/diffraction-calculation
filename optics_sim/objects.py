"""Object and grating constructors. All physical lengths are in metres."""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def aperture(grid, kind="circle", width=200e-6, height=None):
    height = width if height is None else height
    if not np.isfinite(width) or not np.isfinite(height) or min(width, height) <= 0:
        raise ValueError("孔径宽高必须是有限正数。")
    if max(width, height) >= grid.size:
        raise ValueError("孔径必须小于网格窗口。")
    X, Y = grid.xy
    if kind == "circle":
        mask = X**2+Y**2 <= (width/2)**2
    elif kind in ("square", "slit"):
        # Cell-averaged boundary transmission prevents a systematic one-pixel
        # aperture-width deficit when a continuous edge lies on a grid node.
        wx = np.clip((width/2-np.abs(X))/grid.dx+.5, 0, 1)
        wy = np.clip((height/2-np.abs(Y))/grid.dx+.5, 0, 1)
        mask = wx*wy
    else:
        raise ValueError("孔径类型必须是 circle / square / slit。")
    if not mask.any():
        raise ValueError("孔径小于一个像素，无法解析。")
    return mask.astype(complex)


def image_object(grid, path, samples=128, mode="amplitude", phase_depth=2*np.pi):
    if mode not in ("amplitude", "phase"):
        raise ValueError("图片物体模式必须是 amplitude / phase。")
    if not isinstance(samples, int) or not 2 <= samples <= grid.n-2:
        raise ValueError("图片有效样本数必须位于 [2,n-2]。")
    with Image.open(path) as image:
        image = image.convert("L")
        image.thumbnail((samples, samples), Image.Resampling.LANCZOS)
        gray = np.asarray(image, dtype=float)/255
    patch = gray if mode == "amplitude" else np.exp(1j*phase_depth*gray)
    u = np.zeros((grid.n, grid.n), dtype=complex)
    h, w = patch.shape
    cy, cx = grid.n//2-h//2, grid.n//2-w//2
    u[cy:cy+h, cx:cx+w] = patch
    return u


def demo_object(grid, width=400e-6, complex_phase=True):
    """Smooth asymmetric transparent target with genuine, nonconstant phase."""
    if not 0 < width < grid.size:
        raise ValueError("物体宽度必须小于窗口。")
    X, Y = grid.xy
    w = width/5
    a = (np.exp(-((X+width*.22)**2+(Y+width*.15)**2)/(2*w**2)) +
         .65*np.exp(-((X-width*.23)**2+(Y-width*.18)**2)/(2*(w*.65)**2)) +
         .25*np.exp(-((X-width*.1)**2+(Y+width*.3)**2)/(2*(w*.45)**2)))
    a /= a.max()
    phase = .8*X/width + 1.1*np.exp(-((X+width*.12)**2+Y**2)/(2*w**2))
    return a * np.exp(1j*phase) if complex_phase else a.astype(complex)


def grating_line(grid, period=20e-6, kind="binary", duty=.5, phase_depth=np.pi, shift=0.):
    if not np.isfinite(period) or period <= 0 or not 0 < duty < 1:
        raise ValueError("周期必须为正，占空比必须位于 (0,1)。")
    if period/grid.dx < 8:
        raise ValueError("光栅每周期至少需要8个样本，请减小 dx。")
    # Rounded sample coordinates avoid floating-point changes at binary edges.
    cycles = np.round((grid.x-shift)/period, 12)
    binary = ((cycles+.5) % 1 < duty).astype(float)
    sine = .5*(1+np.cos(2*np.pi*cycles))
    if kind == "binary":
        return binary.astype(complex)
    if kind == "sine":
        return sine.astype(complex)
    if kind == "phase":
        return np.exp(1j*phase_depth*binary)
    if kind == "phase_sine":
        return np.exp(1j*phase_depth*sine)
    raise ValueError("光栅类型必须是 binary / sine / phase / phase_sine。")


def grating(grid, period=20e-6, kind="binary", duty=.5, phase_depth=np.pi,
            dimensions=1, finite_fraction=1., shift=0.):
    if dimensions not in (1, 2) or not 0 < finite_fraction <= 1:
        raise ValueError("维度必须为1或2，有效窗口比例必须位于 (0,1]。")
    g = grating_line(grid, period, kind, duty, phase_depth, shift)
    u = np.tile(g, (grid.n, 1)) if dimensions == 1 else np.outer(g, g)
    if finite_fraction < 1:
        X, Y = grid.xy
        u *= (np.abs(X)<grid.size*finite_fraction/2) & (np.abs(Y)<grid.size*finite_fraction/2)
    return u


def phase_screen(grid, seed=7, rms=.8, smooth_pixels=12):
    w = gaussian_filter(np.random.default_rng(seed).normal(size=(grid.n, grid.n)), smooth_pixels)
    w -= w.mean()
    return w * rms/max(float(w.std()), 1e-30)
