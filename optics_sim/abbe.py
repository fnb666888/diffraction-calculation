"""Coherent Abbe imaging: two physical Fourier transforms with a complex pupil."""
from dataclasses import asdict, dataclass
import numpy as np
from PIL import Image
from .core import Grid, Plane, fft2c, power
from .objects import image_object
from .metrics import relative_error, similarity
from .results import Result


FILTER_LABELS = {
    "none": "无附加滤波", "lowpass": "圆形低通", "highpass": "圆形高通",
    "bandpass": "环形带通", "gaussian": "高斯低通", "rectangle": "矩形低通",
    "slit": "方向狭缝", "notch": "对称陷波", "phase_contrast": "中心相位片",
    "vortex": "螺旋相位", "phase_ramp": "相位斜坡", "custom": "自定义复振幅遮罩"}
OBJECT_LABELS = {"amplitude_target": "二维振幅靶标", "phase_target": "弱相位物体"}


@dataclass
class AbbeConfig:
    n: int = 512
    dx: float = 5e-6
    wavelength: float = 632.8e-9
    backend: str = "scipy"
    seed: int = 7
    f1: float = .1
    f2: float = .1
    numerical_aperture: float = .03
    object_kind: str = "amplitude_target"
    object_width: float = 400e-6
    phase_depth: float = .35
    image_path: str | None = None
    image_mode: str = "amplitude"
    image_samples: int = 256
    filter_kind: str = "lowpass"
    cutoff: float = 12000.
    inner_cutoff: float = 6000.
    rectangle_width: float = 32000.
    rectangle_height: float = 12000.
    slit_width: float = 4000.
    angle_deg: float = 0.
    notch_x: float = 25000.
    notch_y: float = 0.
    notch_radius: float = 3000.
    phase_radius: float = 1800.
    phase_shift: float = np.pi/2
    vortex_charge: int = 1
    shift_x: float = 80e-6
    shift_y: float = 0.
    filter_amplitude_path: str | None = None
    filter_phase_path: str | None = None


def invert(field):
    """Periodic inversion around the sample at x=y=0, not around a pixel edge."""
    return np.roll(np.asarray(field)[::-1, ::-1], (1, 1), axis=(0, 1))


def target(grid, config):
    if config.image_path:
        return image_object(grid, config.image_path, config.image_samples, config.image_mode)
    x, y = grid.xy
    w = config.object_width
    gx = np.cos(2*np.pi*x/(w/10))
    gy = np.cos(2*np.pi*y/(w*.15))
    if config.object_kind == "amplitude_target":
        bump = np.exp(-((x-.15*w)**2+(y+.1*w)**2)/(2*(w/16)**2))
        return ((np.abs(x)<=w/2)&(np.abs(y)<=w/2))*(.55+.2*gx+.15*gy+.08*bump)+0j
    if config.object_kind == "phase_target":
        phase = config.phase_depth*np.exp(-(x*x+y*y)/(2*(w/4)**2))*(.55+.45*gx*gy)
        return np.exp(1j*phase)
    raise ValueError("阿贝物体必须是 amplitude_target / phase_target。")


def _mask_image(path, n):
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((n,n), Image.Resampling.BILINEAR), float)/255
    # Display-image y is downwards; a periodic reflection keeps the DC pixel fixed.
    return np.roll(gray[::-1], 1, axis=0)


def make_filter(grid, c):
    fx, fy = np.meshgrid(grid.frequency, grid.frequency)
    radius = np.hypot(fx, fy)
    k = c.filter_kind
    if k not in FILTER_LABELS:
        raise ValueError("未知频谱滤波器："+str(k))
    if k == "none":
        mask = np.ones_like(radius)
    elif k == "lowpass":
        mask = radius <= c.cutoff
    elif k == "highpass":
        mask = radius >= c.cutoff
    elif k == "bandpass":
        if not 0 <= c.inner_cutoff < c.cutoff:
            raise ValueError("带通内截止频率须小于外截止频率。")
        mask = (radius >= c.inner_cutoff)&(radius <= c.cutoff)
    elif k == "gaussian":
        mask = np.exp(-radius**2/(2*c.cutoff**2))
    elif k == "rectangle":
        mask = (np.abs(fx)<=c.rectangle_width/2)&(np.abs(fy)<=c.rectangle_height/2)
    elif k == "slit":
        theta = np.deg2rad(c.angle_deg)
        mask = np.abs(-fx*np.sin(theta)+fy*np.cos(theta)) <= c.slit_width/2
    elif k == "notch":
        mask = (np.hypot(fx-c.notch_x, fy-c.notch_y)>c.notch_radius)&(
                np.hypot(fx+c.notch_x, fy+c.notch_y)>c.notch_radius)
    elif k == "phase_contrast":
        mask = np.exp(1j*c.phase_shift*(radius<=c.phase_radius))
    elif k == "vortex":
        mask = np.exp(1j*c.vortex_charge*np.arctan2(fy,fx))
        if c.vortex_charge:
            mask[radius==0] = 0  # A blocked central sample regularizes the singularity.
    elif k == "phase_ramp":
        mask = np.exp(2j*np.pi*(fx*c.shift_x+fy*c.shift_y)/(c.f2/c.f1))
    else:
        if not (c.filter_amplitude_path or c.filter_phase_path):
            raise ValueError("自定义滤波至少需要一张振幅或相位遮罩图。")
        amplitude = _mask_image(c.filter_amplitude_path, grid.n) if c.filter_amplitude_path else 1.
        phase = 2*np.pi*_mask_image(c.filter_phase_path, grid.n) if c.filter_phase_path else 0.
        mask = amplitude*np.exp(1j*phase)*np.ones_like(radius)
    return np.asarray(mask, complex)


def filter_expressions(c):
    """Concrete displayed mask expression; frequencies use mm^-1."""
    v = lambda name: f"{getattr(c,name)*1e-3:.7g}"
    formulas = {
        "none": r"M=1",
        "lowpass": r"M=\mathbf{1}_{r\leq "+v("cutoff")+"}",
        "highpass": r"M=\mathbf{1}_{r\geq "+v("cutoff")+"}",
        "bandpass": r"M=\mathbf{1}_{"+v("inner_cutoff")+r"\leq r\leq "+v("cutoff")+"}",
        "gaussian": r"M=\exp[-r^2/(2\cdot"+v("cutoff")+"^2)]",
        "rectangle": r"M=\mathbf{1}_{|\nu_x|\leq "+f"{c.rectangle_width*.5e-3:.7g}"+r"}\mathbf{1}_{|\nu_y|\leq "+f"{c.rectangle_height*.5e-3:.7g}"+"}",
        "slit": r"M=\mathbf{1}_{|-\nu_x\sin\theta+\nu_y\cos\theta|\leq "+f"{c.slit_width*.5e-3:.7g}"+r"},\quad\theta="+f"{c.angle_deg:g}"+r"^\circ",
        "notch": r"M=\mathbf{1}_{\sqrt{(\nu_x-"+v("notch_x")+r")^2+(\nu_y-"+v("notch_y")+r")^2}>"+v("notch_radius")+r"}\mathbf{1}_{\sqrt{(\nu_x+"+v("notch_x")+r")^2+(\nu_y+"+v("notch_y")+r")^2}>"+v("notch_radius")+"}",
        "phase_contrast": r"M=\exp[i("+f"{c.phase_shift:.7g}"+r")\mathbf{1}_{r\leq "+v("phase_radius")+"}]",
        "vortex": r"M=\exp[i("+str(c.vortex_charge)+r")\operatorname{atan2}(\nu_y,\nu_x)]"+(r",\quad M(0,0)=0" if c.vortex_charge else ""),
        "phase_ramp": r"M=\exp[i2\pi(\nu_x\cdot"+f"{c.shift_x*1e3:.7g}"+r"+\nu_y\cdot"+f"{c.shift_y*1e3:.7g}"+r")/"+f"{c.f2/c.f1:.7g}"+"]",
        "custom": r"M[k,j]=A_M[k,j]\exp(i2\pi G_\phi[k,j])",
    }
    return [r"r=\sqrt{\nu_x^2+\nu_y^2},\quad\nu_x,\nu_y\;[\mathrm{mm}^{-1}]",
            formulas[c.filter_kind],
            r"P=\mathbf{1}_{r\leq "+f"{c.numerical_aperture/c.wavelength*1e-3:.7g}"+r"},\quad H=P\,M"]


def run(config=None):
    c = config or AbbeConfig()
    grid = Grid(c.n,c.dx,c.wavelength)
    scalars = [c.f1,c.f2,c.numerical_aperture,c.object_width,c.cutoff,c.inner_cutoff,
               c.rectangle_width,c.rectangle_height,c.slit_width,c.angle_deg,c.notch_x,c.notch_y,
               c.notch_radius,c.phase_radius,c.phase_shift,c.phase_depth,c.shift_x,c.shift_y]
    if not np.all(np.isfinite(scalars)):
        raise ValueError("阿贝参数必须为有限数值。")
    if min(c.f1,c.f2,c.cutoff,c.rectangle_width,c.rectangle_height,c.slit_width,c.notch_radius,c.phase_radius)<=0:
        raise ValueError("焦距、截止频率及滤波器尺寸必须为正。")
    if not 0 < c.numerical_aperture <= 1 or not isinstance(c.vortex_charge,int):
        raise ValueError("NA须在(0,1]，螺旋拓扑荷须为整数。")
    if not c.image_path and not 0 < c.object_width < grid.size:
        raise ValueError("物体尺寸必须为正并小于计算窗口。")
    u0 = target(grid,c)
    if not np.any(np.abs(u0)>0):
        raise ValueError("输入物体不能为全零场。")
    fx,fy = np.meshgrid(grid.frequency,grid.frequency)
    pupil = (fx**2+fy**2 <= (c.numerical_aperture/c.wavelength)**2).astype(float)
    mask = make_filter(grid,c)
    transfer = pupil*mask
    xi = c.wavelength*c.f1*grid.frequency
    dxi = c.wavelength*c.f1/(grid.n*grid.dx)
    spectrum = fft2c(u0)*grid.n*grid.dx**2/(1j*c.wavelength*c.f1)
    filtered = spectrum*transfer
    magnification = c.f2/c.f1
    second = grid.n*dxi**2/(1j*c.wavelength*c.f2)
    baseline = fft2c(spectrum*pupil)*second
    output = fft2c(filtered)*second
    ideal = -invert(u0)/magnification
    dx_out = grid.dx*magnification
    p0 = power(u0,grid.dx)
    pb = power(baseline,dx_out)
    po = power(output,dx_out)
    prediction = power(filtered,dxi)
    metrics = {"magnification_signed": -magnification, "image_dx_m": dx_out,
               "fourier_plane_dx_m": dxi, "pupil_cutoff_per_m": c.numerical_aperture/c.wavelength,
               "fourier_plane_pupil_radius_m": c.f1*c.numerical_aperture,
               "input_power":p0,"baseline_power":pb,"output_power":po,
               "pupil_transmitted_power_fraction":pb/p0,"filter_power_fraction_of_baseline":po/pb if pb else 0.,
               "power_fraction_of_input":po/p0,
               "spectral_power_balance_relative_error":abs(po-prediction)/max(p0,1e-30),
               "unfiltered_vs_ideal_complex_error":relative_error(baseline,ideal),
               "filtered_vs_baseline_intensity_similarity":similarity(np.abs(output)**2,np.abs(baseline)**2),
               "filter_passive":bool(np.max(np.abs(transfer))<=1+1e-12),
               "pupil_resolved_by_grid":bool(c.numerical_aperture/c.wavelength<1/(2*c.dx))}
    planes = {"object":Plane(u0,grid.x,"object plane"),
              "fourier":Plane(spectrum,xi,"first lens Fourier plane"),
              "filtered_fourier":Plane(filtered,xi,"filtered Fourier plane"),
              "baseline_image":Plane(baseline,grid.x*magnification,"finite NA unfiltered inverted image"),
              "image":Plane(output,grid.x*magnification,"4f filtered inverted image"),
              "ideal_image":Plane(ideal,grid.x*magnification,"unlimited pupil inverted image")}
    notes = ["相干、标量、近轴4f模型；两次正向FFT保留物理倒像，横向放大率为-f2/f1。省略共同轴向载波相位。",
             "频谱面坐标ξ=λf1νx；物方相干孔径P=1[r≤NA/λ]，额外滤波器M与其相乘。NA只表示等效圆形通带，不逐面建模镜片边缘。",
             "输出振幅包含f1/f2标度，像面采样Δxi=(f2/f1)Δx；按各平面采样面积计算功率。",
             "输入和频谱为复场；频谱滤波作用于复振幅，不对强度图直接做频谱乘法。",
             "离散FFT采用周期窗口；无滤波基准仍受NA限制，理想无限孔径倒像另存。",
             "低通保留低频，高通/带通抑制背景，方向狭缝选择沿其长轴的频率；硬边界可产生振铃。",
             "中心相位片只改变中心频带相位；螺旋相位的奇点样本遮挡。相位斜坡参数为最终像面位移。",
             "自定义遮罩全幅对应当前频谱视场，中心像素为DC；灰度分别映射到振幅和2π相位，图上方为正νy。"]
    if c.image_path:
        notes.append("照片灰度被指定为振幅透过率或相位延迟；未测量真实相位。空间图采用y向下的图像坐标，物理倒像仍保留。")
    params=asdict(c)
    params["backend"]="scipy"
    params["display_image_y_down"]=bool(c.image_path)
    return Result("abbe",params,metrics,planes,
                  {"pupil":pupil,"filter_mask":mask,"effective_transfer":transfer,
                   "frequency_per_m":grid.frequency},notes)
