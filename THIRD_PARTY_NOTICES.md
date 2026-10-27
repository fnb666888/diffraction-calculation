# 第三方材料与许可说明

根目录 [LICENSE](LICENSE) 中的 MIT 许可适用于本项目贡献者有权许可的原创代码和原创技术文档。第三方数据、代码、图片、课程材料与机构标识继续遵守各自的条件；根目录许可证不替代这些条件。

## 随项目保留的示例

| 本地位置 | 来源与署名 | 许可与说明 |
| --- | --- | --- |
| `data/pointcloud_holography/` | [PCL data](https://github.com/PointCloudLibrary/data) 的猫、马、狼 ISM 教程点云；[PCL 教程](https://pointclouds.org/documentation/tutorials/implicit_shape_model.html) 注明数据借用自 Ohio dataset | 上游仓库采用 BSD-3-Clause。保留本地 `LICENSE-PCL.txt` 和三个 `*-source.json`，继续保留 PCL／Ohio 来源信息。 |
| `data/optical_holography/camera/source.png` | Lav Varshney，scikit-image 的 Camera 示例 | CC0；[上游文档](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.camera) 说明已替换旧版受限图像。当前来源与校验值见同目录 `source.json`。 |
| `data/cgh_standard/astronaut/` | NASA 宇航员 Eileen Collins 照片，经 scikit-image 获取 | [上游文档](https://scikit-image.org/docs/0.24.x/api/skimage.data.html#skimage.data.astronaut) 标明 public domain、无已知版权限制。保留 NASA 来源，不暗示人物或机构背书。`target-gray.png` 为该图的处理结果。 |
| `data/digital_holography/diatom/source.png` | Reconrabbit，[Wikimedia Commons 原始页面](https://commons.wikimedia.org/wiki/File:Cocconeis_rapheless_valve_internal_view_SEM.png) | 本地 `source.json` 记录为 CC0 1.0。照片仅提供纹理，项目指定的光学相位并非实测相位。 |
| `data/two_depth_alphabets/` | Yingsi Qin，2021，[3DHologram](https://github.com/yingsiqin/3DHologram) 的示例图片与 `propagate2Depths.m` | MIT。保留该目录原有 [LICENSE](data/two_depth_alphabets/LICENSE) 和 `source.json`，其中包含上游作者版权声明。 |

下载链接、SHA-256 和使用方式以各目录已有的来源 JSON 为准。处理或裁剪素材后，也应保留其来源和许可信息。

## 本地历史材料与可选数据

以下材料不因本项目采用 MIT 而获得新的开放授权：

- **UW RGB-D 咖啡杯点云**：`data/rgbd/coffee_mug_1/` 来自 [University of Washington RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/dataset/)。现有 `source.json` 将用途记录为非商业研究／教学，但该记录不构成上游完整的再分发许可。当前查阅的数据集页面和 [PCD README](https://rgbd-dataset.cs.washington.edu/dataset/rgbd-dataset_pcd_ascii/README.txt) 未提供足以将其重新按 MIT 分发的授权。默认忽略该目录体系下的 PCD 和下载归档，保留来源元数据；使用或再分发前应向数据提供方确认适用条件。文献为 Kevin Lai、Liefeng Bo、Xiaofeng Ren、Dieter Fox，*A Large-Scale Hierarchical Multi-View RGB-D Object Dataset*，ICRA 2011。
- **课程文档提取的魔方／足球图片**：`data/cgh_slm/two_planes/input-color.png` 与 `input-depth-levels.png` 源于本地课程文档。现有记录没有单独的开源图片授权，因此默认忽略这两张图片；相应历史脚本需要使用者自行提供有权使用的输入。
- **PPT、Word、模板、机构标识与外部截图**：生成的 `outputs/`、制作中间目录 `.pptx-build/` 和 `docs/` 下的本地 Office 文档默认忽略。对含有上述素材的报告单独发布时，需要遵守素材原有条件。机构名称和校徽不因代码许可证而获得授权。

忽略规则仅影响尚未跟踪的文件。如果这些材料已进入索引或提交历史，应在发布前单独核对；不要把“Git 忽略”理解为已经完成许可清理。

## 软件依赖

运行时依赖通过 pip 安装，具体版本要求见 [pyproject.toml](pyproject.toml)。它们保留各自的许可证。若另行制作包含依赖的安装包，应一并保留相应发行包中的版权与许可文件。

LightPipes 2.1.5 的发行包 `LICENSE.txt` 为 BSD-3-Clause，版权属于 LightPipes Developers（2017），与 [上游许可证](https://github.com/opticspy/lightpipes/blob/master/LICENSE.txt) 的文本一致。其包元数据另有不一致的 MIT 分类标签，引用许可时以发行包实际附带的许可证文本为依据。

NumPy、SciPy、Matplotlib、Pillow、可选的 pytest／CuPy 以及制作文档时使用的工具，也不改用本项目的 MIT 许可证。这里列出的文件不是这些依赖许可证全文的替代品。

## 维护约定

新增第三方素材时，记录原始 URL、作者、明确许可、文件校验值和所做变换，并附上需要保留的许可证。无法确认授权的材料可作为本地可选输入，避免加入默认源码分发。

本项目采用的标准 MIT 文本可在 [Open Source Initiative](https://opensource.org/license/mit) 查阅。
