# 羽毛球追踪器 (Shuttlecock Tracker)

这是一个使用SAM2（Segment Anything 2）模型来检测和追踪羽毛球比赛视频中的羽毛球的工具。

## 功能特点

- 使用SAM2模型实现高精度的羽毛球追踪
- **自动检测**羽毛球位置，无需手动标记
- 支持任何格式的羽毛球比赛视频输入
- 生成带有羽毛球跟踪标记的输出视频
- 可在CPU或GPU上运行（推荐GPU以获得更好的性能）

## 环境要求

- Python 3.8+
- PyTorch 2.0+
- OpenCV
- CUDA（推荐，用于GPU加速）

## 安装

1. 克隆此仓库：

```bash
git clone <repository-url>
cd shuttlecock_tracker
```

2. 创建并激活虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或
.venv\Scripts\activate  # Windows
```

3. 安装依赖项：

```bash
pip install -r requirements.txt
pip install -e sam2/
```

## 使用方法

使用以下命令运行羽毛球追踪器：

```bash
python main.py --input <输入视频路径> --output <输出视频路径> [--device <cuda/cpu>]
```

参数说明：
- `--input`：要处理的羽毛球比赛视频的路径（必需）
- `--output`：输出视频的保存路径（默认为"output.mp4"）
- `--device`：使用的计算设备，可以是"cuda"（GPU）或"cpu"（默认为"cuda"）

示例：

```bash
python main.py --input badminton_match.mp4 --output tracked_match.mp4 --device cuda
```

## 工作原理

这个工具使用以下步骤来追踪羽毛球：

1. 加载预训练的SAM2（Segment Anything 2）模型
2. **自动检测**视频前几帧中的羽毛球位置：
   - 使用基于边缘检测的方法在单帧中寻找近似圆形的小物体
   - 使用运动检测方法分析相邻帧之间的差异找到快速移动的物体
   - 如果检测失败，则使用视频中心点作为默认位置
3. 使用SAM2的视频预测器在整个视频序列中传播分割掩码
4. 在每一帧上可视化检测到的羽毛球
5. 生成带有羽毛球追踪标记的输出视频

## 自动检测原理

羽毛球检测器使用两种互补的方法来找到视频中的羽毛球：

1. **单帧检测**：基于羽毛球的视觉特征（圆形、明亮、小体积）在单个帧中查找羽毛球
2. **运动检测**：通过分析相邻帧之间的差异来检测快速移动的物体，特别适合检测运动中的羽毛球

这种双重策略使得系统能够在各种环境和光照条件下可靠地找到羽毛球。

## 局限性和改进空间

- 当前的自动检测算法在某些情况下可能会失败（比如背景复杂、光线不足）。可以通过集成更先进的物体检测模型（如YOLOv8）来进一步提高检测准确率。
- 对于快速移动的羽毛球，追踪可能不够准确。可以通过增加关键帧的频率或使用运动预测来改进。
- 当羽毛球被球员遮挡时，追踪可能会丢失。可以考虑添加重新检测机制。

## 许可证

[添加许可证信息] 