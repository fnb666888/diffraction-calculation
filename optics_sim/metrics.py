import numpy as np
from scipy.signal import fftconvolve


def similarity(a, b):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.real(np.vdot(a, b))/denom) if denom else 0.


def relative_error(a, b):
    return float(np.linalg.norm(np.asarray(a)-b)/max(np.linalg.norm(b), 1e-30))


def align_global_phase(a, truth):
    product = np.vdot(truth, a)
    return a*np.exp(-1j*np.angle(product))


def phase_rmse(a, truth, mask=None):
    a = align_global_phase(a, truth)
    if mask is None:
        mask = np.abs(truth) > np.abs(truth).max()*.05
    return float(np.sqrt(np.mean(np.angle(a[mask]*np.conj(truth[mask]))**2))) if mask.any() else 0.


def align_cdi(a, truth):
    """Evaluate only: remove Fourier-intensity translation/twin/global-phase ambiguities."""
    best, best_error, best_details = None, np.inf, None
    twin = np.conj(np.roll(a[::-1, ::-1], (1, 1), axis=(0, 1)))
    for label, candidate in (("direct", a), ("conjugate_inversion", twin)):
        corr = fftconvolve(np.abs(truth), np.abs(candidate)[::-1, ::-1], mode="full")
        shift = np.array(np.unravel_index(np.argmax(corr), corr.shape)) - np.array(candidate.shape) + 1
        shifted = np.roll(candidate, tuple(shift), axis=(0, 1))
        shifted = align_global_phase(shifted, truth)
        err = relative_error(shifted, truth)
        if err < best_error:
            best, best_error = shifted, err
            best_details = {"ambiguity": label, "shift_yx": shift.tolist()}
    return best, best_error, best_details
