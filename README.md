# 衍射、泰伯效应与离轴全息模拟

Python 数值模拟项目。三个实验模块共用复光场、物体构造、传播接口、定量评价和绘图。默认菲涅耳传播调用 LightPipes 2.1.5；NumPy/SciPy 负责数值计算，Matplotlib 绘图，Tkinter 提供桌面界面。

## 查看结果

打开 **[outputs/workflows/index.html](outputs/workflows/index.html)** 浏览20个完整案例。每例包含物理输入、输入振幅/相位、计算过程、输出、验证及原始数据。案例内的 `00_全流程图.png` / `.svg` 是带编号与箭头的全流程大图。

输入示意图和输入复振幅均保存在各案例内，统一随实验结果维护。每例的 `01_输入图/透过率与复振幅表达式.png` / `.svg` 明确写出该屏的具体透过率、入射场、屏后复场、实际采样表达式及参数，并纳入全流程图。

五类衍射案例的 `04_输出结果/` 同时保存近场和远场的振幅、相位、强度图。远场相位保留球面波二次相位，省略共同轴向相位 `exp(ikz)`；以弧度显示主值，屏蔽振幅低于峰值约 `1e-5` 的区域。

## 安装与启动

需要 Python 3.11 或更新版本。Windows 双击根目录的 **`启动模拟.cmd`** 启动桌面界面；它调用 `scripts/launch.ps1` 检查环境和依赖。本机已有的 `.venv` 可以继续使用。

手动安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -r requirements.txt
```

依赖统一声明在 `pyproject.toml`；`requirements.txt` 通过 `-e .` 安装当前项目。

## 运行实验

在项目根目录执行：

```powershell
# 桌面界面
.\.venv\Scripts\python.exe -m optics_sim gui --config configs/default.json

# 指定模块，或运行三个模块
.\.venv\Scripts\python.exe -m optics_sim diffraction --object square --z-m 0.02
.\.venv\Scripts\python.exe -m optics_sim talbot --z-steps 100
.\.venv\Scripts\python.exe -m optics_sim holography --theta-deg 1.5
.\.venv\Scripts\python.exe -m optics_sim all --config configs/default.json

# 完整案例：全部生成、指定生成、使用已保存数据重绘
.\.venv\Scripts\python.exe -m optics_sim.workflow_catalog
.\.venv\Scripts\python.exe -m optics_sim.workflow_catalog --case 01 02
.\.venv\Scripts\python.exe -m optics_sim.workflow_catalog --render-only
```

默认输出统一位于项目 `outputs/`：完整案例在 `outputs/workflows/`，常规模块和界面运行结果在 `outputs/时间戳/`，独立验证在 `outputs/validation/`。可用 `--output` 明确指定其它目录。

对自己的照片同时生成衍射与离轴全息的完整流程：

```powershell
.\.venv\Scripts\python.exe -m optics_sim.image_workflows --image "D:\下载\dog.jpg"
```

默认保存到 `outputs/image_simulations/图片名/`，衍射、全息各自一个文件夹，入口为 `index.html`。原照片复制到各例中，灰度指定为振幅透过率，初始相位设为0；不从普通照片推断真实相位。采用1536×1536衍射网格、1024×1024全息网格、2.5 μm采样、632.8 nm波长，照片保持长宽比缩放到最长边512样本，传播2 cm。全息带宽为44000 m⁻¹，参考角约5.25°；衍射远场距离独立按NF≤0.05选择。导出包含补零窗口对比、全息复场误差、16位归一化全息图及浮点原始强度；空间图采用y向下的图像坐标。

自定义照片的远场中心区域使用SciPy Zoom FFT直接计算601×601个频谱样本，保留完整复相位与物理归一化；全范围FFT场和局部精算场分别保存，不使用图片插值代替计算。

## 参数和图片

`configs/default.json` 是默认实验；`configs/phase_and_2d.json` 演示二维相位光栅和角谱传播。配置顶层为 `shared`、`diffraction`、`talbot`、`holography`，参数名对应各模块的 Config 数据类。JSON/Python 中长度统一为 m，参考角 `theta_deg` 使用度，带宽使用 m⁻¹。界面和 CLI 中明确标有 μm/nm 的参数会转换单位。

示例图片保存在 `examples/`；可运行 `examples/create_target.py` 重建。图片保留长宽比并嵌入中央零场：振幅模式把灰度映射为 [0,1]，相位模式在图片支集内设振幅1、相位为灰度×2π。

```powershell
.\.venv\Scripts\python.exe -m optics_sim diffraction --image examples/amplitude_target.png --image-mode amplitude
.\.venv\Scripts\python.exe -m optics_sim holography --image examples/phase_target.png --image-mode phase
```

## 数据和验证

`data.npz` 保存原始复场与坐标，`metrics.json` 保存参数、指标和模型说明。数组索引为 `[y,x]`；振幅为 `abs(field)`，相位为 `angle(field)`，强度为 `abs(field)**2`。远场和单FFT传播必须使用各自导出的输出坐标。零振幅处相位未定义；部分相干总强度没有单一相位。

开发依赖和回归测试：

```powershell
.\.venv\Scripts\python.exe -m pip install --index-url https://pypi.org/simple -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m optics_sim validate
```

`validate` 检查解析解、传播往返、功率、采样及补零收敛、泰伯、全息与CDI，保存 `validation.json` 和 `convergence.png`。检查失败时退出码为1。阈值适用于声明的基准参数，调整实验参数后应重新检查。

## 目录

```text
optics_sim/             计算、绘图、界面与导出代码
configs/                实验参数
examples/               示例输入图片及生成脚本
tests/                  物理回归测试
docs/
  architecture.md       架构与目录说明
  numerical-notes.md    数值约定、模型范围与方案修正
  plans/                三个原始执行方案归档
scripts/launch.ps1      Windows启动辅助脚本
outputs/                生成的结果与验证记录，Git忽略
  workflows/            20个完整案例
  validation/           独立物理与收敛验证
启动模拟.cmd           Windows双击入口
pyproject.toml          项目信息与依赖的统一声明
requirements.txt        当前项目安装入口
README.md               使用说明
```

目录详细说明见 [docs/architecture.md](docs/architecture.md)。原始方案保留在 [docs/plans/](docs/plans/)，当前实现的修正及适用范围见 [docs/numerical-notes.md](docs/numerical-notes.md)，具体输入函数的表达约定见 [docs/input-expressions.md](docs/input-expressions.md)。

## 开发维护

- 修改参数：`configs/`；添加物体：`objects.py`；修改传播：`core.py`。
- 实验算法分别位于 `diffraction.py`、`talbot.py`、`holography.py`。
- 修改完整案例的展示和导出：`workflow_catalog.py`；输入图绘制：`input_visualization.py`。
- 计算结果和缓存可再生成，统一由 `.gitignore` 排除；源码、原始方案和示例输入需要保留。
- 输出迁移时保持整个案例文件夹及内部相对路径，避免拆开图表、HTML与数据。
