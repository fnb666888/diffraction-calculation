"""Regression checks for conventions where visually plausible mistakes are easy."""
import numpy as np
import pytest
from PIL import Image
from optics_sim.core import Grid, propagate, transfer_function, far_field, far_field_window, power
from optics_sim.objects import demo_object, grating, image_object
from optics_sim.holography import (record, reconstruct_off_axis, bandlimit,
    carrier_conditions, record_phase_steps, reconstruct_phase_steps, HolographyConfig, run)
from optics_sim.metrics import relative_error, align_global_phase


@pytest.mark.parametrize("z", [.012, -.012])
def test_lightpipes_matches_fresnel_envelope_for_asymmetric_complex_field(z):
    g = Grid(128, 5e-6)
    obj = demo_object(g, 200e-6)
    lp = propagate(obj, g, z)
    scipy = propagate(obj, g, z, backend="scipy")
    assert relative_error(align_global_phase(lp, scipy), scipy) < 1e-7
    assert relative_error(propagate(lp, g, -z), obj) < 1e-10


def test_far_field_physical_scaling_conserves_power():
    g = Grid(128, 5e-6)
    obj = demo_object(g, 200e-6)
    plane = far_field(obj, g, .2)
    dx_out = plane.x[1]-plane.x[0]
    assert power(plane.field, dx_out) == pytest.approx(power(obj, g.dx), rel=1e-12)
    assert dx_out == pytest.approx(g.wavelength*.2/(g.n*g.dx))


def test_zoom_far_field_matches_direct_complex_diffraction_sum():
    g = Grid(32, 5e-6)
    rng = np.random.default_rng(23)
    field = rng.normal(size=(g.n,g.n))+1j*rng.normal(size=(g.n,g.n))
    z = .3
    plane = far_field_window(field, g, z, 3.7/g.size*g.wavelength*z, samples=11)
    frequencies = plane.x/(g.wavelength*z)
    kernel = np.exp(-2j*np.pi*frequencies[:,None]*g.x[None,:])
    expected = (kernel@field@kernel.T)*g.dx**2/(1j*g.wavelength*z)
    expected *= np.exp(1j*np.pi/(g.wavelength*z)*(plane.x[:,None]**2+plane.x[None,:]**2))
    assert relative_error(plane.field, expected)<1e-11


@pytest.mark.parametrize("dimensions", [1, 2])
@pytest.mark.parametrize("kind", ["binary", "sine", "phase", "phase_sine"])
def test_talbot_reproduces_complex_field_and_half_period_translation(kind, dimensions):
    g = Grid(128, 1.25e-6)
    obj = grating(g, kind=kind, dimensions=dimensions, phase_depth=1.1)
    zt = 2*(20e-6)**2/g.wavelength
    assert relative_error(propagate(obj, g, zt), obj) < 1e-7
    shifted = np.roll(obj, 8, axis=1)
    if dimensions==2:
        shifted = np.roll(shifted, 8, axis=0)
    assert relative_error(propagate(obj, g, zt/2), shifted) < 1e-7


def test_four_phase_steps_preserve_sign_and_amplitude_of_general_complex_wave():
    rng = np.random.default_rng(9)
    u = rng.normal(size=(32, 32))+1j*rng.normal(size=(32, 32))
    rec = reconstruct_phase_steps(record_phase_steps(u, 2.), 2.)
    assert relative_error(rec, u) < 1e-12
    w = rng.uniform(-np.pi, np.pi, u.shape)
    common = reconstruct_phase_steps(record_phase_steps(u*np.exp(1j*w), 2., w), 2.)
    assert relative_error(common, u) < 1e-12


def test_off_axis_reconstruction_needs_only_intensity_and_known_reference():
    g = Grid(256, 5e-6)
    u, _ = bandlimit(demo_object(g, 200e-6), g, 8000)
    fc = 53/g.size
    X, _ = g.xy
    r = 1.4*np.exp(2j*np.pi*fc*X)
    measured = record(u, r)
    assert np.isrealobj(measured)
    assert carrier_conditions(g, fc, 8000)["separated"]
    assert relative_error(reconstruct_off_axis(measured, r, g, 8000, 1.4), u)<1e-11


def test_invalid_holographic_sampling_fails_explicitly():
    with pytest.raises(ValueError, match="Nyquist"):
        run(HolographyConfig(n=128, theta_deg=5))
    with pytest.raises(ValueError, match="分离"):
        run(HolographyConfig(n=128, theta_deg=.2))


def test_evanescent_branch_decays_and_backward_operator_is_bounded():
    g = Grid(64, .15e-6, .6328e-6)
    forward = transfer_function(g, 1e-6, "angular")
    backward = transfer_function(g, -1e-6, "angular")
    assert np.isfinite(forward).all() and np.isfinite(backward).all()
    assert np.abs(forward).max() <= 1+1e-15
    assert np.abs(backward).max() <= 1+1e-15
    assert (backward==0).any()


def test_image_amplitude_phase_and_non_square_centering(tmp_path):
    path = tmp_path/"target.png"
    Image.fromarray(np.full((20, 40), 128, dtype=np.uint8)).save(path)
    g = Grid(64)
    a = image_object(g, path, 32, "amplitude")
    p = image_object(g, path, 32, "phase", np.pi)
    mask = np.abs(p)>0
    assert mask.sum()==16*32
    assert np.abs(p[mask]) == pytest.approx(np.ones(mask.sum()))
    assert a[mask].real == pytest.approx(np.full(mask.sum(), 128/255))
    assert np.angle(p[mask]) == pytest.approx(np.full(mask.sum(), np.pi*128/255))


def test_bad_grid_or_negative_intensity_is_rejected():
    from optics_sim.diffraction import recover_cdi
    with pytest.raises(ValueError):
        Grid(63)
    with pytest.raises(ValueError):
        recover_cdi(-np.ones((32, 32)), np.ones((32, 32), bool))
