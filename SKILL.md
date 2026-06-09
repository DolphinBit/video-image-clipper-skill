---
name: video-image-clipper
version: 1.0.0
author: Dolphin2025127
description: 多功能视频剪辑工具，支持 CLIP/ORB 图像匹配与 InsightFace 人脸检测，含批量处理、人脸马赛克、反向剪辑等功能。
tags:
  - video
  - image
  - clip
  - computer-vision
  - face-detection
  - insightface
---

# Video Image Clipper

多功能视频剪辑工具，支持三种运行模式：

- **CLIP 模式**：基于 CLIP 神经网络进行图像匹配
- **ORB 模式**：基于 ORB 特征点进行图像匹配
- **Face 模式**：基于 InsightFace 进行人脸检测与身份匹配

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基础用法

```bash
# 人脸模式
python clip_by_image.py --method face --video input.mp4 --face target.jpg --out output

# ORB 模式
python clip_by_image.py --method orb --video input.mp4 --image target.jpg --out output

# 批量处理
python clip_by_image.py --method face --video_dir ./videos --face target.jpg --out output
```

### 高级功能

- 人脸马赛克脱敏（`--blur_face 1`）
- 反向剪辑（`--reverse 1`）
- 合并输出（`--concat_all 1`）
- 片段缓冲合并（`--gap_buffer`）
- 时间区间裁剪（`--start_sec` / `--end_sec`）
- 跳帧加速（`--skip_frame`）

## 依赖

- Python 3.8+
- opencv-python
- numpy
- insightface
