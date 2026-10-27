"""Independent coherent-imaging checks: signs, magnification and Fourier orders."""
from dataclasses import replace
import numpy as np
import pytest
from PIL import Image
from optics_sim.abbe import AbbeConfig,run,invert,make_filter
from optics_sim.core import Grid,fft2c
from optics_sim.metrics import relative_error


def test_4f_magnification_inversion_and_power_with_unlimited_discrete_pupil():
    c=AbbeConfig(n=128,dx=5e-6,numerical_aperture=.2,filter_kind="none",f2=.17)
    r=run(c)
    expected=-invert(r.planes['object'].field)/(c.f2/c.f1)
    assert relative_error(r.planes['image'].field,expected)<1e-12
    assert r.planes['image'].x[65]-r.planes['image'].x[64]==pytest.approx(c.dx*c.f2/c.f1)
    assert r.metrics['power_fraction_of_input']==pytest.approx(1.,rel=1e-12)
    # The bump makes the object asymmetric; a missing inversion must be detectable.
    assert relative_error(r.planes['image'].field,-r.planes['object'].field/(c.f2/c.f1))>1e-3


@pytest.mark.parametrize('kind',['none','lowpass','highpass','bandpass','gaussian','rectangle','slit','notch','phase_contrast','vortex','phase_ramp'])
def test_passive_masks_power_matches_independent_spectral_budget(kind):
    c=AbbeConfig(n=128,dx=5e-6,filter_kind=kind)
    r=run(c)
    F=fft2c(r.planes['object'].field)
    predicted=float(np.sum(np.abs(F*r.arrays['effective_transfer'])**2)*c.dx**2)
    assert r.metrics['output_power']==pytest.approx(predicted,rel=1e-12,abs=1e-25)
    assert r.metrics['output_power']<=r.metrics['baseline_power']*(1+1e-12)
    if kind in ('none','phase_contrast','phase_ramp'):
        assert r.metrics['output_power']==pytest.approx(r.metrics['baseline_power'],rel=1e-12)


def test_notch_removes_selected_symmetric_orders_and_slit_orientation():
    g=Grid(128,5e-6)
    f=8/g.size
    X,Y=g.xy
    u=1+.2*np.cos(2*np.pi*f*X)+.3*np.cos(2*np.pi*f*Y)
    F=fft2c(u)
    c=AbbeConfig(n=g.n,dx=g.dx,filter_kind='notch',notch_x=f,notch_y=0,notch_radius=.4/g.size)
    filtered=F*make_filter(g,c)
    assert abs(filtered[64,72])<1e-12 and abs(filtered[64,56])<1e-12
    assert abs(filtered[72,64])>1 and abs(filtered[64,64])>1
    slit=make_filter(g,replace(c,filter_kind='slit',angle_deg=0,slit_width=.8/g.size))
    assert slit[64,72]==1 and slit[72,64]==0


def test_phase_ramp_gives_positive_final_image_displacement():
    c=AbbeConfig(n=128,dx=5e-6,filter_kind='none',f2=.15)
    baseline=run(c)
    ramp=run(replace(c,filter_kind='phase_ramp',shift_x=3*c.dx*c.f2/c.f1,shift_y=-2*c.dx*c.f2/c.f1))
    expected=np.roll(baseline.planes['image'].field,(-2,3),axis=(0,1))
    assert relative_error(ramp.planes['image'].field,expected)<1e-12


def test_phase_plate_converts_weak_phase_to_intensity_contrast():
    r=run(AbbeConfig(object_kind='phase_target',filter_kind='phase_contrast'))
    source=r.planes['object'].intensity
    assert np.max(source)-np.min(source)<1e-12
    assert np.std(r.planes['image'].intensity)>5*np.std(r.planes['baseline_image'].intensity)


def test_custom_masks_gray_amplitude_phase_and_dc_coordinate(tmp_path):
    amplitude=np.full((32,32),128,np.uint8)
    phase=np.zeros((32,32),np.uint8)
    phase[16,16]=64
    a,p=tmp_path/'a.png',tmp_path/'p.png'
    Image.fromarray(amplitude).save(a)
    Image.fromarray(phase).save(p)
    c=AbbeConfig(n=32,dx=5e-6,filter_kind='custom',filter_amplitude_path=str(a),filter_phase_path=str(p))
    mask=make_filter(Grid(32,5e-6),c)
    assert mask[16,16]==pytest.approx(128/255*np.exp(2j*np.pi*64/255))
    assert abs(mask[0,0])==pytest.approx(128/255)
    with pytest.raises(ValueError,match='至少'):
        make_filter(Grid(32,5e-6),replace(c,filter_amplitude_path=None,filter_phase_path=None))
