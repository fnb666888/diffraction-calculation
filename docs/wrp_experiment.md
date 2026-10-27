# WRP二维照片试算

本次在项目真实小熊猫照片上试验WRP，保留常规离轴全息算法，并将试算作为独立模块保存。当前输入没有真实深度数据，本实验为单深度平面照片的WRP退化情形，不声称实现通用RGB-D三维点云WRP。

## 传播与恢复

- 常规全息物光生成采用LightPipes `Forvard` 对应的菲涅耳传递函数。不是逐像素直接积分，也没有选择代码中可选的精确角谱传播。
- 物光传播20 mm，带宽44/mm，波长632.8 nm，计算网格1024×1024，像素间距2.5 μm。输入照片区域512×341，与CDI、阿贝相同。
- 复场恢复采用已知参考光解调、圆形低通分离一级、逆传播。振幅为复数模，相位为atan2得到的包裹相位，光强为模平方。
- 当前重建距离由配置给出，没有自聚焦搜索。新增10、20、30 mm重建图仅展示不同距离的结果，没有自动选择焦面。
- CDI另用HIO/ER，不与离轴全息频谱分离混为一体。文档中的GS和双相位编码涉及纯相位SLM全息编码，未用于本次离轴复场恢复。

## WRP方法

取WRP距物面0.1 mm，再传播19.9 mm到全息面。保持相同带限物场、复数精度、采样、周期边界和菲涅耳核。

1. 完整两段传播为 `P(19.9 mm) P(0.1 mm)`，与单步 `P(20 mm)` 等价，用来检查分段本身。
2. 局部WRP先对近传播的**离散冲激响应**截方形窗口，再对输入复振幅局部累加，最后整面传播19.9 mm。平面菲涅耳核可分离，两个一维复卷积与二维方形窗口点源累加数学等价，节省重复运算。不对截断核重新做幅值归一。
3. 窗口分别为13×13、25×25、49×49，检查扩大窗口时的误差变化。未用第一菲涅耳区半径直接代替准确性检查。
4. 各方法产生自己的物光，与同一倾斜参考光干涉。随后使用相同的离轴算法重建，因此重建图包含了前向近似引入的误差。

## 测速口径和结果

CPU为Intel Core i7-12700H，SciPy FFT单线程，complex128。预热后零/正弦相位各9次，轮换方法顺序，合并18次计时取中位数。传播核预先缓存，排除文件读写、绘图、光瞳、参考光记录和逆重建。不同方法使用同一软件后端进行比较。

| 方法 | 物光传播中位耗时 | 物光复场最大相对误差 |
| --- | ---: | ---: |
| 单步FFT传播 | 30.0 ms | 基准 |
| 完整两段传播 | 59.9 ms | 8.36×10⁻¹⁶ |
| 局部WRP 13×13 | 144.0 ms | 13.28% |
| 局部WRP 25×25 | 161.9 ms | 2.31% |
| 局部WRP 49×49 | 205.6 ms | 0.51% |

误差均对零、正弦两个条件取较大值。49×49窗口的重建光强相对误差最大为0.912%。单步SciPy基准与现有LightPipes物光相对差小于3.5×10⁻¹⁰。

本次二维密集照片输入下，WRP没有加速。原算法已经通过FFT利用整张平面图的卷积结构，增加中间平面及局部累加带来额外开销。结论限于此模型、精度、窗口和实现，不能外推到稀疏三维点云、其他硬件或GPU版本。常规全息计算继续采用原单步FFT传播。

## 复现

在项目根目录运行：

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& '.venv/Scripts/python.exe' -m optics_sim.wrp_experiment --repeats 9
& '.venv/Scripts/python.exe' -m optics_sim.method_figures
```

原始结果保存在 `outputs/wrp_experiment/red_panda/data.npz`，逐次耗时、误差和参数在同目录 `metrics.json`。讲解图在 `outputs/method_explanation/`。图像只作显示增强，原始复数数据保留。

## 来源

- 用户提供：`D:/桌面/22063101 朱熙莱 全息课设 第三次大作业.docx`，WRP方法与图1。
- Shimobaba, Masuda, Ito, *Simple and fast calculation algorithm for computer-generated hologram with wavefront recording plane*, Optics Letters 34, 3133–3135 (2009), <https://doi.org/10.1364/OL.34.003133>。
- LightPipes传播说明：<https://opticspy.github.io/lightpipes/manual.html>。
- Takeda, Ina, Kobayashi, *Fourier-transform method of fringe-pattern analysis for computer-based topography and interferometry*, JOSA 72, 156–160 (1982), <https://doi.org/10.1364/JOSA.72.000156>。
- Fienup, *Phase retrieval algorithms: a comparison*, Applied Optics 21, 2758–2769 (1982), <https://doi.org/10.1364/AO.21.002758>。
