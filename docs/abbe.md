# 阿贝成像与频谱面调制

第四个独立模块采用相干平面波照明的4f系统。物面位于第一透镜前焦面，公共频谱面位于两镜之间，第二透镜的后焦面输出倒像。适用于透明片/SLM复振幅调制；不会从普通照片推断真实相位。

## 使用

桌面：双击`启动模拟.cmd`，选择“阿贝成像”。用右侧滚动条查看参数。图片留空时使用二维振幅靶标或弱相位物体，也可输入自己的图片；频谱滤波器可直接选择中文名称。

```powershell
.\.venv\Scripts\python.exe -m optics_sim abbe --config configs/abbe.json
.\.venv\Scripts\python.exe -m optics_sim abbe --filter highpass --cutoff 8000
.\.venv\Scripts\python.exe -m optics_sim abbe --filter bandpass --inner-cutoff 6000 --cutoff 20000
.\.venv\Scripts\python.exe -m optics_sim abbe --filter slit --slit-width 4000 --angle-deg 90
.\.venv\Scripts\python.exe -m optics_sim abbe --image "D:\下载\dog.jpg" --image-mode amplitude --filter lowpass --cutoff 12000
.\.venv\Scripts\python.exe -m optics_sim abbe --filter custom --filter-amplitude examples/abbe_filter_amplitude.png --filter-phase examples/abbe_filter_phase.png
```

JSON采用SI单位。CLI的截止/狭缝频率以m⁻¹计，相移以rad计；GUI频率为mm⁻¹，焦距为mm，位移为μm。NA为物方数值孔径。图片采样由`image_samples`或`--image-samples`设置。

## 滤波器

所有附加滤波器M与相同有限NA孔径P相乘；无附加滤波也受NA限制。r=√(νx²+νy²)。

| 类型 | 参数与作用 |
|---|---|
| `none` | M=1，无附加滤波基准 |
| `lowpass` | r≤cutoff，保留低频；硬边会振铃 |
| `highpass` | r≥cutoff，遮挡低频，常突出边缘 |
| `bandpass` | inner_cutoff≤r≤cutoff，选择环形频带 |
| `gaussian` | exp[-r²/(2 cutoff²)]，cutoff为高斯σ |
| `rectangle` | rectangle_width×rectangle_height矩形通带 |
| `slit` | slit_width频率宽，angle_deg为长轴角度；0°允许沿νx的频率 |
| `notch` | ±(notch_x,notch_y)处遮挡notch_radius圆盘 |
| `phase_contrast` | 中心phase_radius频带增加phase_shift相位；弱相位物体转为强度对比 |
| `vortex` | exp[i·l·atan2(νy,νx)]，l=vortex_charge；l≠0时遮挡零频样本 |
| `phase_ramp` | 频谱线性相位；shift_x/shift_y为最终像面位移 |
| `custom` | 可独立选择滤波器振幅图和相位图，也可同时使用 |

全部参数可在GUI或JSON里设置。自定义遮罩全幅重采样到n×n频谱视场，中心像素为DC，上方为正νy。振幅灰度→[0,1]，相位灰度→[0,2π]；空缺通道分别取1和0。偶数网格纵轴用保持中心不动的周期反射，最外行位于Nyquist接缝。它不同于物体图片的中央嵌入规则。

运行`examples/create_abbe_filters.py`可重建两张示例复滤波器图；原图随案例保存。

## 结果与模型

当前完整示例使用真实照片，位于`outputs/image_simulations/图片名/03_照片阿贝成像_圆形低通/`。其内部`滤波对比/`保存六种频谱滤波与原始复场；小熊猫另有`04_照片阿贝成像_自定义复遮罩/`。浏览入口为`outputs/workflows/abbe.html`。原21至32的合成物体结果已清理。

专题入口为`outputs/workflows/abbe.html`。可运行`.\.venv\Scripts\python.exe -m optics_sim.abbe_comparison --image "D:\下载\dog.jpg"`，生成照片在六种滤波器下的对比和原始数据，保存于`outputs/abbe_comparison/dog/`。对比图独立按峰值归一化并转回物方方向，原始物理倒像、真实强度和功率仍保留在NPZ和JSON中。

频谱物理坐标ξ=λf1νx、η=λf1νy；近轴等效孔径半径为f1·NA。图表使用空间频率坐标，NPZ保存实际ξ坐标。两次正向FFT产生物理倒像，横向放大率-f2/f1，像面采样(f2/f1)dx，振幅保留f1/f2标度。功率按各平面的像素面积计算。

强度图的对数显示不改写原始数值；中心截面使用统一线性强度标度。有限NA截掉的频率不能靠附加滤波恢复。SciPy负责理想4f变换，复用Grid、Plane、Result；LightPipes继续用于其他模块的菲涅耳传播。模型采用周期FFT窗口，未逐面建模实际镜片像差、非相干照明或SLM器件误差。

参考：[TU Delft相干成像](https://qiweb.tudelft.nl/aoi/coherentimaging/coherentimaging/)、[MIT 4f空间滤波课程](https://web.mit.edu/2.710/Fall06/2.710-wk10-b-ho.pdf)。
