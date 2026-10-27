# Optics Sim Lab · 波动光学模拟实验室

用 Python 模拟衍射、泰伯自成像、全息记录与再现，以及阿贝相干成像。项目提供桌面界面、命令行和独立案例脚本，保存计算参数、复振幅数组、图表及 HTML 报告，适合课程实验、算法学习和数值复现。

## 功能

| 模块 | 主要内容 |
| --- | --- |
| 衍射计算 | 圆孔、方孔、有限长单缝与图片输入；菲涅耳、角谱、夫琅禾费及单 FFT 菲涅耳传播 |
| 泰伯效应 | 一维、二维振幅或相位光栅；整距、半距、沿程演化与部分相干对比 |
| 数字全息 | 单幅离轴干涉记录、频谱选级、复振幅恢复与数值逆传播；另保留相移案例 |
| 光学全息 | 模拟振幅全息片记录、参考光照明、光学选级及实像／虚像再现 |
| 计算全息（CGH） | 二维目标的 GS 傅里叶纯相位编码；三维点云的点源求和与复振幅编码 |
| 相干衍射成像（CDI） | 由衍射光强结合支集等先验，使用 HIO／ER 迭代恢复物体 |
| 阿贝成像 | 相干 4f 成像、圆形低通／高通等频谱滤波、自定义振幅与相位遮罩 |
| 三维点云全息 | 猫、马、狼点云分别演示三种记录／编码流程，展示不同深度的光强、振幅和相位 |

桌面界面包含衍射、泰伯、离轴全息和阿贝成像四个基础标签页。CGH、图片 CDI 和点云案例通过独立模块运行。

## 安装

需要 **Python 3.11 或更新版本**。主要依赖为 NumPy、SciPy、Matplotlib、Pillow 和 LightPipes 2.1.5，安装声明统一在 [pyproject.toml](pyproject.toml)。

Windows PowerShell，在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux／macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

下文使用 Windows 路径；Linux／macOS 将 `.\.venv\Scripts\python.exe` 换成 `.venv/bin/python`。桌面界面需要 Tkinter 和图形显示环境；部分 Linux 发行版需要另行安装 `python3-tk`，命令行计算不需要打开桌面窗口。

## 快速开始

先运行一个不依赖外部图片的圆孔衍射案例：

```powershell
.\.venv\Scripts\python.exe -m optics_sim diffraction --object circle --z-m 0.02 --backend scipy --output outputs/quickstart
```

完成后打开 `outputs/quickstart/index.html` 查看结果。

启动桌面界面或查看参数帮助：

```powershell
.\.venv\Scripts\python.exe -m optics_sim gui
.\.venv\Scripts\python.exe -m optics_sim --help
```

安装并激活虚拟环境后，也可以使用 `optics-sim` 命令。

> `启动模拟.cmd` 会读取 `configs/default.json`。当前配置中的部分照片路径指向本地生成的 `outputs/`，新克隆的仓库通常没有这些文件。首次使用可按上面的命令启动界面，或把配置中的 `image_path` 改成自己的图片路径。

## 基础实验

```powershell
# 方孔衍射
.\.venv\Scripts\python.exe -m optics_sim diffraction --object square --z-m 0.02

# 泰伯自成像
.\.venv\Scripts\python.exe -m optics_sim talbot --z-steps 100

# 图片作为振幅输入，模拟离轴全息
.\.venv\Scripts\python.exe -m optics_sim holography --image examples/amplitude_target.png --image-mode amplitude

# 阿贝成像：圆形高通，截止频率单位为 m^-1
.\.venv\Scripts\python.exe -m optics_sim abbe --image examples/amplitude_target.png --filter highpass --cutoff 8000

# 使用内置物体运行四个基础模块
.\.venv\Scripts\python.exe -m optics_sim all --backend scipy
```

也可通过 `--config` 读取 JSON 配置。顶层包含 `shared`、`diffraction`、`talbot`、`holography`、`abbe`，字段与相应模块的 Config 数据类一致。CLI 参数覆盖配置中的同名设置。

标准孔径和光栅的完整流程案例：

```powershell
.\.venv\Scripts\python.exe -m optics_sim.workflow_catalog --case 01 02
# 省略 --case 可生成全部标准案例
# 已有案例数据时，可加 --render-only 重绘
```

汇总入口为 `outputs/workflows/index.html`。

## 图片输入、CDI 和计算全息

使用仓库内 Camera 示例，生成图片衍射、离轴全息和阿贝滤波报告：

```powershell
.\.venv\Scripts\python.exe -m optics_sim.image_workflows --image data/optical_holography/camera/source.png --include-abbe
```

结果保存在 `outputs/image_simulations/source/`。处理自己的图片时替换 `--image` 路径即可。

独立 CDI 案例使用 HIO／ER、矩形支集和实非负先验：

```powershell
.\.venv\Scripts\python.exe -m optics_sim.cdi_image --image data/optical_holography/camera/source.png --output outputs/cdi_image/camera --iterations 2000
```

二维 GS 计算全息、光学全息和相机数字全息示例：

```powershell
.\.venv\Scripts\python.exe -m optics_sim.cgh_standard
.\.venv\Scripts\python.exe -m optics_sim.optical_holography_demo
.\.venv\Scripts\python.exe -m optics_sim.digital_holography_demo
```

报告分别为：

- `outputs/cgh_standard/astronaut/report.html`
- `outputs/optical_holography/camera/report.html`
- `outputs/digital_holography/diatom/report.html`

上述示例使用不同模型。二维 GS 案例是透镜焦面傅里叶全息；Camera 案例模拟光学全息片再现；硅藻纹理的相机案例含噪声、量化与距离搜索。它们与下面的点云案例分别保存参数和结果。

## 三维点云的记录与分深度再现

三组点云各含 3400 个相干点源，按连续三维坐标计算物光：

| 输入 | 记录或编码 | 再现方式 |
| --- | --- | --- |
| 猫点云 | 离轴干涉光强记录为理想振幅全息片 | 共轭参考光照明、光学选级、向前传播形成实像 |
| 马点云 | 点源求和，编码会聚复场的振幅与相位 | 理想双调制器接受平面波照明，再向前传播 |
| 狼点云 | 一张 16 位离轴相机全息图 | FFT 选取一级、解调恢复复场、负距离数值传播 |

按顺序生成记录和流程图：

```powershell
# 限制矩阵计算线程，可按本机情况调整
$env:OPENBLAS_NUM_THREADS = '4'
$env:OMP_NUM_THREADS = '4'

.\.venv\Scripts\python.exe -m optics_sim.pointcloud_direct
.\.venv\Scripts\python.exe -m optics_sim.pointcloud_single_offaxis
.\.venv\Scripts\python.exe -m optics_sim.pointcloud_process_figures
```

`pointcloud_direct` 已默认生成单幅离轴的狼记录。第二条命令复用保存的正向物光，生成单独的 `single-offaxis/` 数据集；第三条命令优先使用该数据集绘图。后两条命令依赖第一步的输出。

结果位于 `outputs/pointcloud_direct/{cat,horse,wolf}/`，狼的独立单幅记录位于 `wolf/single-offaxis/`。各案例的 `full-process/` 包含实际全息图或调制图、各深度光强／振幅／相位图、再现复场与显示标度。2048×2048 复数网格和点源求和会占用较多内存，建议先完成基础案例再运行。

这三组使用有限距离的菲涅耳型模型，中心物距约 3 mm、波长 632.8 nm、物光带宽 120/mm。再现距离为预设值，没有自动选焦；点云 CGH 使用复振幅编码，没有 GS 或 WRP 迭代。完整参数见 [三维点云全息流程](docs/pointcloud_holographic_flow.md)。

## 数值约定与模型范围

- 长度使用 SI 单位 m，空间频率使用 m⁻¹，相位使用 rad；CLI 中标有 nm／μm 的参数会转换单位。
- 复场数组索引为 `[y, x]`。振幅为 `abs(U)`，光强为 `abs(U)**2`，包裹相位为 `angle(U)`。
- FFT 只是数值计算手段，全息属于菲涅耳型还是傅里叶型取决于记录与再现模型。
- 普通照片的灰度映射为振幅、光强或相位时，需要遵循具体案例的约定。指定的相位和深度属于建模输入。
- 不同传播方法可能产生不同输出坐标与采样间隔，比较时应读取对应坐标和归一化参数。
- 图像的归一化、γ提亮和对数色标用于显示，原始复数数组单独保存。零振幅处相位未定义，部分相干总光强没有单一相位。
- 点云案例展示同一全息数据在不同深度的光场，不代表恢复了可从任意角度观察的完整几何；模型未包含真实材质散射、遮挡和多次散射。

默认基础传播后端为 LightPipes；SciPy 后端以及角谱、远场等路径由具体配置选择。阿贝成像使用 SciPy FFT。详见 [数值说明](docs/numerical-notes.md) 和 [输入表达式约定](docs/input-expressions.md)。

## 项目结构与输出

```text
optics_sim/               计算、图表、GUI 和独立案例模块
configs/                  JSON 实验配置
examples/                 自生成靶标、滤波器及生成脚本
data/                     示例数据、来源记录及各自许可证
docs/                     算法、模型边界和实验说明
tests/                    物理与数值回归测试
scripts/launch.ps1        Windows 启动辅助脚本
outputs/                  生成的图表、复场、报告、PPT（Git 忽略）
.pptx-build/              本地 PPT 制作中间文件（Git 忽略）
pyproject.toml            包信息、依赖和命令行入口
LICENSE                   本项目 MIT 许可证
THIRD_PARTY_NOTICES.md     第三方数据与材料的许可说明
```

基础 CLI 默认输出到 `outputs/时间戳/`，可用 `--output` 指定目录。独立案例有各自的默认输出目录。根据案例不同，输出包括 HTML、PNG、NPZ、JSON 等文件；移动报告时请保留整个案例目录及相对路径。

`outputs/` 和 PPT 制作缓存不随源码分发，安装项目不会自动获得这些结果。汇报文件可在生成后另行分享。

### 专题文档

- [架构与目录](docs/architecture.md)、[阿贝成像](docs/abbe.md)、[图片 CDI](docs/cdi_image.md)
- [初始相位对比](docs/phase_comparison.md)、[WRP 试算](docs/wrp_experiment.md)
- [点源全息](docs/pointcloud_direct.md)、[分深度再现](docs/pointcloud_holographic_flow.md)
- 历史扩展：[多角度投影层析](docs/pointcloud_holography.md)、[带先验的三维反演](docs/pointcloud_3d_reconstruction.md)。这些扩展采用额外数据或先验，与单幅全息分深度再现分开讨论。

部分历史脚本依赖以前生成的照片结果或本地课程文档，应先阅读相应说明。README 中的基础示例不依赖这些历史输出。

## 开发与贡献

传播与采样约定位于 `optics_sim/core.py`，基础算法位于 `diffraction.py`、`talbot.py`、`holography.py`、`abbe.py`。新增算法时请记录坐标、单位、随机种子、显示变换和评价条件。报告问题时附上运行命令、配置、Python 版本及错误信息。

需要自行运行检查时：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m optics_sim validate
```

验证模块检查解析解、传播往返、功率、采样、泰伯自像和重建等基准，输出到 `outputs/validation/`。GPU CDI 扩展需要另外安装与本机 CUDA 匹配的 CuPy，基础功能不依赖 GPU。

`.gitignore` 排除虚拟环境、缓存、结果、PPT 中间产物及部分需另行确认许可的本地素材，保留源码、配置、示例与数据来源记录。**已跟踪或已暂存的文件不会因新增忽略规则自动移出 Git**。若此前添加过本地产物，可针对相应目录执行以下命令，工作目录中的文件会保留：

```powershell
git rm -r --cached --ignore-unmatch -- .pptx-build outputs
```

第三方材料的处理范围见下节，发布前也应核对暂存区。

## 开源许可与素材来源

本项目原创代码和原创技术文档采用 [MIT License](LICENSE)，允许使用、修改和再分发，分发时需保留版权和许可声明。

第三方图片、点云、上游代码、课程文档素材和机构标识保留各自的权利与使用条件。具体来源、许可及本地数据的分发范围见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。`data/` 中的 `source.json`、`*-source.json` 和已有许可证应随相关素材保留。
