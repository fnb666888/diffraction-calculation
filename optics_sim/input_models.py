"""Object-specific optical expressions and independent evaluation on the grid.

Displayed x/y lengths use micrometres, frequency uses inverse micrometres.
Numeric evaluation uses SI and matches the sampling convention of objects.py.
"""
from dataclasses import dataclass, field
import numpy as np
from PIL import Image
from .core import fft2c, ifft2c


def number(value):
    text = f"{float(value):.10g}"
    if "e" in text:
        coefficient, exponent = text.split("e")
        return coefficient+r"\times10^{"+str(int(exponent))+"}"
    return text


def math(expression):
    return "$"+expression+"$"


@dataclass
class InputModel:
    name: str
    transmission: list[str]
    incident: list[str]
    input_field: list[str]
    sampling: list[str]
    t_numeric: np.ndarray
    input_numeric: np.ndarray
    source_symbol: str
    short_field: str
    propagation: list[str] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)


def _image_model(result):
    p = result.parameters
    n, dx = p["n"], p["dx"]
    with Image.open(p["image_path"]) as img:
        img = img.convert("L")
        img.thumbnail((p["image_samples"], p["image_samples"]), Image.Resampling.LANCZOS)
        gray = np.asarray(img, float)/255
    h, w = gray.shape
    k0, j0 = n//2-h//2, n//2-w//2
    bounds = (f"{j0}\\leq j<{j0+w},\\quad {k0}\\leq k<{k0+h}")
    t = np.zeros((n, n), complex)
    phase = p["image_mode"]=="phase"
    patch = np.exp(2j*np.pi*gray) if phase else gray
    t[k0:k0+h, j0:j0+w] = patch
    symbol = "t_G"
    expr = r"\exp(i2\pi G_{k-k_0,j-j_0})" if phase else r"G_{k-k_0,j-j_0}"
    transmission = [math(r"G_{k,j}=\mathrm{gray}_{k,j}/255,\quad 0\leq G_{k,j}\leq1"),
                    math(r"t_G[k,j]="+expr+r",\quad (j,k)\in\Omega_G"),
                    math(r"t_G[k,j]=0,\quad (j,k)\notin\Omega_G"),
                    math(r"\Omega_G:\quad "+bounds),
                    math(f"j_0={j0},\\quad k_0={k0},\\quad (W,H)=({w},{h})")]
    sampling = [math(r"x_j=(j-N/2)\Delta x,\quad y_k=(k-N/2)\Delta x"),
                math(f"N={n},\\quad \\Delta x={number(dx*1e6)}\\,\\mu\\mathrm{{m}}"),
                "G 为实际读取、灰度转换及缩放后的像素矩阵；支集外透过率为0。"]
    short = r"U_0[k,j]=\exp(i2\pi G_{k-k_0,j-j_0})\,\mathbf{1}_{\Omega_G}(k,j)" if phase else r"U_0[k,j]=G_{k-k_0,j-j_0}\,\mathbf{1}_{\Omega_G}(k,j)"
    return symbol, transmission, sampling, t, short


def _synthetic_model(result):
    x, y = np.meshgrid(result.planes["object"].x, result.planes["object"].x)
    width, s = 400e-6, 80e-6
    g1 = np.exp(-((x+width*.22)**2+(y+width*.15)**2)/(2*s**2))
    g2 = np.exp(-((x-width*.23)**2+(y-width*.18)**2)/(2*(s*.65)**2))
    g3 = np.exp(-((x-width*.1)**2+(y+width*.3)**2)/(2*(s*.45)**2))
    gp = np.exp(-((x+width*.12)**2+y**2)/(2*s**2))
    a = g1+.65*g2+.25*g3
    normalization = float(a.max())
    t = a/normalization*np.exp(1j*(.8*x/width+1.1*gp))
    lines = [math(r"g_1=\exp[-((x+88)^2+(y+60)^2)/(2\cdot80^2)]"),
             math(r"g_2=\exp[-((x-92)^2+(y-72)^2)/(2\cdot52^2)]"),
             math(r"g_3=\exp[-((x-40)^2+(y+120)^2)/(2\cdot36^2)]"),
             math(r"g_\varphi=\exp[-((x+48)^2+y^2)/(2\cdot80^2)]"),
             math(r"t_h(x,y)=\frac{g_1+0.65g_2+0.25g_3}{C}\exp[i(0.002x+1.1g_\varphi)]"),
             math(r"C=\max_{k,j}(g_1+0.65g_2+0.25g_3)\simeq"+number(normalization))]
    return "t_h", lines, ["高斯分量的坐标和宽度均以 μm 表示；C 使用实际采样网格最大值。"], t


def model_for(result):
    p = result.parameters
    variant = p.get("workflow_variant", "")
    x, y = np.meshgrid(result.planes["object"].x, result.planes["object"].x)
    n, dx, lam = p["n"], p["dx"], p["wavelength"]
    incident = [math(r"U_{\mathrm{inc}}(x,y)=1+0i")]
    sampling, extra = [], []
    if result.module=="cdi":
        X, Y = x/(n*dx), y/(n*dx)
        q1 = (X>-.13)&(X<-.05)&(Y>-.12)&(Y<.10)
        q2 = (X>-.07)&(X<.13)&(Y>-.11)&(Y<-.04)
        q3 = (X>.01)&(X<.11)&(Y>.015)&(Y<.10)
        t = (.8*q1*(1-q2)+q2+.5*q3).astype(complex)
        symbol = r"t_{\mathrm{CDI}}"
        transmission = [math(f"X=x/{number(n*dx*1e6)},\\quad Y=y/{number(n*dx*1e6)}"),
                        math(r"Q_1:\ -0.13<X<-0.05,\ -0.12<Y<0.10"),
                        math(r"Q_2:\ -0.07<X<0.13,\ -0.11<Y<-0.04"),
                        math(r"Q_3:\ 0.01<X<0.11,\ 0.015<Y<0.10"),
                        math(r"t_{\mathrm{CDI}}=0.8\mathbf{1}_{Q_1}(1-\mathbf{1}_{Q_2})+\mathbf{1}_{Q_2}+0.5\mathbf{1}_{Q_3}")]
        short = r"U_{0,\mathrm{CDI}}=0.8\mathbf{1}_{Q_1}(1-\mathbf{1}_{Q_2})+\mathbf{1}_{Q_2}+0.5\mathbf{1}_{Q_3}"
        sampling = ["Q2 区域覆盖与 Q1 的重叠处，和靶标构造中的赋值顺序一致。"]
        extra = [math(r"I_{p,q}=|\mathrm{DFT}_{2,c}(t_{\mathrm{CDI}})|^2"),
                 "真值相位仅作评价；算法输入是强度、支集及非负振幅先验。"]
    elif p.get("image_path"):
        symbol, transmission, sampling, t, short = _image_model(result)
    elif result.module=="abbe":
        w = p["object_width"]
        gx,gy=np.cos(2*np.pi*x/(w/10)),np.cos(2*np.pi*y/(w*.15))
        W,S,PX,PY=map(number,(w*1e6,w*.25e6,w*.1e6,w*.15e6))
        if p["object_kind"]=="phase_target":
            phi=p["phase_depth"]*np.exp(-(x*x+y*y)/(2*(w/4)**2))*(.55+.45*gx*gy)
            t=np.exp(1j*phi)
            symbol=r"t_{\varphi}"
            transmission=[math(r"\varphi_A="+number(p["phase_depth"])+r"\exp[-(x^2+y^2)/(2\cdot"+S+r"^2)]\,[0.55+0.45\cos(2\pi x/"+PX+r")\cos(2\pi y/"+PY+r")]"),
                          math(r"t_\varphi=\exp(i\varphi_A),\quad |t_\varphi|=1")]
            short=r"U_0=\exp(i\varphi_A)"
        else:
            bump=np.exp(-((x-.15*w)**2+(y+.1*w)**2)/(2*(w/16)**2))
            t=(((np.abs(x)<=w/2)&(np.abs(y)<=w/2))*(.55+.2*gx+.15*gy+.08*bump)).astype(complex)
            symbol="t_A"
            transmission=[math(r"b=\exp[-((x-"+number(w*.15e6)+r")^2+(y+"+number(w*.1e6)+r")^2)/(2\cdot"+number(w/16*1e6)+r"^2)]"),
                          math(r"t_A=\mathbf{1}_{|x|\leq "+number(w*.5e6)+r"}\mathbf{1}_{|y|\leq "+number(w*.5e6)+r"}[0.55+0.2\cos(2\pi x/"+PX+r")+0.15\cos(2\pi y/"+PY+r")+0.08b]")]
            short="U_0=t_A"
        sampling=[math(r"N="+str(n)+r",\quad\Delta x="+number(dx*1e6)+r"\,\mu\mathrm{m}"),"显示坐标x、y以μm计；按实际网格点取值。"]
    elif result.module=="diffraction":
        a = p["width"]
        b = p.get("height") or (a*5 if p["kind"]=="slit" else a)
        if p["kind"]=="circle":
            radius = number(a*.5e6)
            t = (x*x+y*y<=(a/2)**2).astype(complex)
            symbol = "t_c"
            transmission = [math(f"t_c(x,y)=1,\\quad x^2+y^2\\leq {radius}^2"),
                            math(f"t_c(x,y)=0,\\quad x^2+y^2>{radius}^2"),
                            math(f"a={radius}\\,\\mu\\mathrm{{m}}\\quad(D={number(a*1e6)}\\,\\mu\\mathrm{{m}})")]
            short = f"U_{{0,c}}(x,y)=\\mathbf{{1}}_{{x^2+y^2\\leq {radius}^2}}"
            sampling = [math(r"U_{0,c}[k,j]=\mathbf{1}_{x_j^2+y_k^2\leq a^2}"), "圆孔按网格点坐标采样，孔外振幅0、相位未定义。"]
        else:
            symbol = "t_s" if p["kind"]=="square" else "t_l"
            t = (np.clip((a/2-np.abs(x))/dx+.5, 0, 1)*np.clip((b/2-np.abs(y))/dx+.5, 0, 1)).astype(complex)
            shape = f"\\mathrm{{rect}}(x/{number(a*1e6)})\\mathrm{{rect}}(y/{number(b*1e6)})"
            transmission = [math(symbol+"(x,y)="+shape),
                            math(f"a_x={number(a*1e6)}\\,\\mu\\mathrm{{m}},\\quad a_y={number(b*1e6)}\\,\\mu\\mathrm{{m}}"),
                            math(r"\mathrm{rect}(u)=1\ (|u|<1/2),\ 0\ (|u|>1/2)")]
            short = "U_0[k,j]=w_{"+number(a*1e6)+"}(x_j)w_{"+number(b*1e6)+"}(y_k)"
            sampling = [math(r"w_a(u)=\mathrm{clip}[(a/2-|u|)/\Delta x+1/2,0,1]"),
                        math(r"U_0[k,j]=w_{a_x}(x_j)w_{a_y}(y_k)"),
                        f"边界像素用单元平均透过率；dx={number(dx*1e6)} μm。连续矩形边界取1/2。"]
    elif result.module=="talbot":
        d, duty, dims = p["period"], p["duty"], p["dimensions"]
        bx = ((np.round(x/d, 12)+.5)%1<duty).astype(float)
        by = ((np.round(y/d, 12)+.5)%1<duty).astype(float)
        sx, sy = .5*(1+np.cos(2*np.pi*x/d)), .5*(1+np.cos(2*np.pi*y/d))
        binary = p["kind"] in ("binary", "phase")
        profile_x, profile_y = (bx, by) if binary else (sx, sy)
        profile = "b" if binary else "s"
        phase = p["kind"].startswith("phase")
        depth = p["phase_depth"]
        if phase:
            argument = profile+"(x)"+("+"+profile+"(y)" if dims==2 else "")
            depth_text = r"\pi" if np.isclose(depth, np.pi, atol=1e-14, rtol=0) else number(depth)
            expression = r"\exp[i"+depth_text+"("+argument+")]"
            t = np.exp(1j*depth*(profile_x+(profile_y if dims==2 else 0)))
        else:
            expression = profile+"(x)"+(profile+"(y)" if dims==2 else "")
            t = (profile_x*(profile_y if dims==2 else 1)).astype(complex)
        symbol = "t_g"
        transmission = [math(f"d={number(d*1e6)}\\,\\mu\\mathrm{{m}},\\quad \\eta={number(duty)}"),
                        math(r"b(u)=\mathbf{1}_{\operatorname{frac}(u/"+number(d*1e6)+r"+1/2)<"+number(duty)+"}") if binary else math(f"s(u)=[1+\\cos(2\\pi u/{number(d*1e6)})]/2"),
                        math("t_g(x,y)="+expression)]
        short = "U_0(x,y)="+expression
        sampling = ["图中周期函数的原点与实际程序一致；二值周期坐标舍入至12位避免边界浮点误差。"]
        if p["finite_fraction"]<1:
            half = n*dx*p["finite_fraction"]/2
            win = (np.abs(x)<half)&(np.abs(y)<half)
            t *= win
            window = r"\mathbf{1}_{|x|<"+number(half*1e6)+r",\ |y|<"+number(half*1e6)+"}"
            transmission.append(math(r"t_{g,\mathrm{finite}}=t_g"+window))
            symbol = r"t_{g,\mathrm{finite}}"
            short += window
        if variant in ("fractional", "tli"):
            R = number(p["curvature_radius"]*1e6)
            extra += [math(r"U_{\mathrm{inc},R}=\exp[i\pi(x^2+y^2)/("+number(lam*1e6)+r"\cdot"+R+r")]")]
            extra.append(math(r"U_{0,R}=t_g\exp[i\pi(x^2+y^2)/("+number(lam*1e6)+r"\cdot"+R+r")]") )
        if variant=="tli":
            d2 = d*(1+p["analyzer_mismatch"])
            extra += [math(r"t_2(x)=\mathbf{1}_{\operatorname{frac}(x/"+number(d2*1e6)+r"+1/2)<"+number(duty)+"}"),
                      math(r"U_{\mathrm{after}\,2}(x,y)=t_2(x)\mathcal{P}_{z_T}[t_g](x,y)")]
    else:
        symbol, transmission, sampling, t = _synthetic_model(result)
        short = r"U_0=\mathcal{F}^{-1}[P_B\mathcal{F}(t_h)]"

    actual = t.copy()
    input_field = [math(r"U_{\mathrm{screen}}="+symbol+r"U_{\mathrm{inc}}="+symbol), math(short)]
    if result.module=="talbot" and variant=="coherence":
        modes = result.arrays["coherence_tilts_per_m"]
        first = float(modes[0])
        actual = t*np.exp(2j*np.pi*first*x)
        incident = [math(r"U_{\mathrm{inc},m}=\exp(i2\pi f_mx),\quad w_m=1/"+str(len(modes))),
                    math(r"f_0\simeq"+number(first*1e-6)+r"\,\mu\mathrm{m}^{-1}")]
        short = r"U_{0,0}(x,y)=t_g(x,y)\exp[i2\pi("+number(first*1e-6)+r")x]"
        input_field = [math(r"U_{0,m}=t_g\exp(i2\pi f_mx)"), math(short)]
        extra = [math(r"I_z=\sum_m w_m|\mathcal{P}_z[t_g\exp(i2\pi f_mx)]|^2"), "本输入幅相图为第一个模态；总强度没有单一相位。"]
    if result.module=="holography":
        fx, fy = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(n, dx)), np.fft.fftshift(np.fft.fftfreq(n, dx)))
        B = p["bandwidth"]
        pupil = (fx*fx+fy*fy)<=B*B
        actual = ifft2c(fft2c(t)*pupil)
        short = r"U_0=\mathcal{F}^{-1}[P_B\mathcal{F}("+symbol+r")]"
        input_field = [math(r"U_{\mathrm{screen}}="+symbol),
                       math(r"P_B=\mathbf{1}_{f_x^2+f_y^2\leq("+number(B*1e-6)+r")^2}"), math(short)]
        sampling.append("透明片透过率与带限输入场分开定义；频率单位 1/μm。")
        amplitude, z = p["reference_amplitude"], p["z"]
        extra.append(math(r"U_{\mathrm{obj}}=\mathcal{P}_{"+number(z*1e6)+r"}[\mathcal{F}^{-1}(P_B\mathcal{F}("+symbol+r"))]"))
        if variant=="phase_steps":
            extra += [math(r"R_k="+number(amplitude)+r"\exp(i\phi_k),\quad\phi_k=0,\pi/2,\pi,3\pi/2"),
                      math(r"U_{\mathrm{obj}}=\frac{I_0-I_\pi+i(I_{\pi/2}-I_{3\pi/2})}{"+number(4*amplitude)+r"}")]
        elif variant=="aberration":
            extra += [math(r"t_W[k,j]=\exp(iW[k,j])"),
                      math(r"W="+number(p["screen_rms"])+r"(V-\langle V\rangle)/\sigma_V"),
                      "V=GaussianFilter(ξ, σ=12像素)，ξ为标准正态场；seed="+str(p["seed"]),
                      math(r"U_d=\exp(iW)\mathcal{P}_{"+number(z*1e6)+"}[U_0]"),
                      math(r"R_k="+number(amplitude)+r"\exp[i(W+\phi_k)]")]
        else:
            carrier = p.get("actual_carrier_per_m", result.metrics.get("sampling", {}).get("carrier_per_m"))
            if carrier is None:
                requested = np.sin(np.deg2rad(p["theta_deg"]))/lam
                carrier = round(requested*n*dx)/(n*dx) if p["snap_carrier"] else requested
            extra += [math(r"R="+number(amplitude)+r"\exp[i2\pi("+number(carrier*1e-6)+r")x]"),
                      math(r"I_H=|U_{\mathrm{obj}}+"+number(amplitude)+r"\exp[i2\pi("+number(carrier*1e-6)+r")x]|^2"),
                      math(r"t_H=I_H/\max(I_H)")]

    if result.module=="abbe":
        from .abbe import AbbeConfig, filter_expressions
        c=AbbeConfig(**{key:p[key] for key in AbbeConfig.__dataclass_fields__ if key in p})
        extra += [math(line) for line in filter_expressions(c)]
        extra += [math(r"\xi=\lambda f_1\nu_x,\quad\eta=\lambda f_1\nu_y"),
                  math(r"f_1="+number(c.f1*1e3)+r"\,\mathrm{mm},\quad f_2="+number(c.f2*1e3)+r"\,\mathrm{mm},\quad\mathrm{NA}="+number(c.numerical_aperture))]
        if c.filter_kind=="custom":
            extra.append("自定义振幅图灰度→AM，相位图灰度→Gφ；直接映射全频谱网格。空缺通道分别取1和0，原图随案例保存。")
        propagation=[math(short),math(r"U_F=\frac{N\Delta x^2}{i\lambda f_1}\mathrm{DFT}_{2,c}[U_0]"),
                     math(r"U_i=\frac{N\Delta\xi^2}{i\lambda f_2}\mathrm{DFT}_{2,c}[H U_F],\quad\Delta\xi=\frac{\lambda f_1}{N\Delta x}"),
                     math(r"m=f_2/f_1="+number(c.f2/c.f1)+r",\quad x_i=m x_j,\quad M_{\mathrm{lateral}}=-m")]
    elif result.module=="diffraction":
        z = p["z"]
        # lambda and z both converted to micrometres, preserving dimensionless phase.
        discrete_source = r"w_{a_x}(x_j)w_{a_y}(y_k)" if not p.get("image_path") and p["kind"] in ("square", "slit") else short.split("=", 1)[1]
        propagation = [math(short),
                       math(r"U_z=\mathrm{IDFT}_{2,c}[\mathrm{DFT}_{2,c}("+discrete_source+r")H_z]"),
                       math(r"H_z=\exp[-i\pi("+number(lam*1e6)+r")("+number(z*1e6)+r")(f_x^2+f_y^2)]")]
        method = p.get("method", "fresnel")
        if method=="angular":
            propagation[-1] = math(r"H_z=\exp\{i2\pi("+number(z/lam)+r")[\sqrt{1-("+number(lam*1e6)+r")^2(f_x^2+f_y^2)}-1]\}")
            propagation.append("正向倏逝波取衰减分支；反向投影掉倏逝分量。")
        elif method in ("fraunhofer", "fresnel_one_fft"):
            prefactor = number(n*(dx*1e6)**2/(lam*z*1e12))
            propagation = [math(short),
                           math(r"X_p=\lambda zf_p,\quad Y_q=\lambda zf_q"),
                           math(r"U_z[q,p]=\frac{"+prefactor+r"}{i}\exp[i\pi(X_p^2+Y_q^2)/(\lambda z)]\mathrm{DFT}_{2,c}(v)[q,p]")]
            propagation.append(math(r"v=U_0") if method=="fraunhofer" else math(r"v=U_0\exp[i\pi(x^2+y^2)/(\lambda z)]"))
            propagation.append(math(r"\lambda="+number(lam*1e6)+r",\quad z="+number(z*1e6)))
    else:
        propagation = [math(r"\mathcal{P}_z[v]=\mathcal{F}^{-1}\{\mathcal{F}(v)\exp[-i\pi\lambda z(f_x^2+f_y^2)]\}"),
                       math(r"\lambda="+number(lam*1e6)+r"\,\mu\mathrm{m}")]
        if result.module=="talbot":
            propagation.append(math(r"U_z=\mathcal{P}_z["+short.split("=",1)[1]+r"]"))
            propagation.append(math(r"z_T=2d^2/\lambda\simeq"+number(2*p["period"]**2/lam*1e6)+r"\,\mu\mathrm{m}"))
    return InputModel(result.module, transmission, incident, input_field, sampling, t, actual, symbol, math(short), propagation, extra)


def model_figure(result, include_propagation=True):
    from .reporting import plt
    model = model_for(result)
    groups = [("1. 本例衍射屏 / 透明片的具体透过率", model.transmission),
              ("2. 入射场与屏后 / 实际输入复振幅", model.incident+model.input_field),
              ("3. 与实际采样矩阵的对应", model.sampling)]
    if model.extra:
        groups.append(("4. 本例的其它光场与光学元件", model.extra))
    if include_propagation:
        groups.append(("传播计算表达式", model.propagation))
    count = sum(len(lines)+1.4 for _, lines in groups)
    height = max(6.8, count*.41+1.5)
    fig, ax = plt.subplots(figsize=(12, height))
    fig.subplots_adjust(left=.04, right=.98, top=.93, bottom=.03)
    ax.axis("off")
    ax.text(0, 1.02, "本例的透过率函数与复振幅表达式", fontsize=19, weight="bold", color="#164e73", transform=ax.transAxes)
    y = .96
    step = .91/count
    ax.text(0, y, "x、y：μm；fx、fy：1/μm。指示函数在条件成立时为1，否则为0；省略公共轴向载波。", fontsize=10, color="#52657a", transform=ax.transAxes)
    y -= step*1.25
    for label, lines in groups:
        ax.text(0, y, label, fontsize=13, weight="bold", color="#164e73", va="top", transform=ax.transAxes)
        y -= step*1.15
        for line in lines:
            ax.text(.02, y, line, fontsize=12, va="top", transform=ax.transAxes)
            y -= step
        y -= step*.35
    return fig
