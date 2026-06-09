# Video Image Clipper Skill

多功能视频剪辑工具，支持 **CLIP / ORB 特征匹配** 和 **InsightFace 人脸检测** 三种模式，可对视频进行智能裁剪、人脸匹配过滤、批量处理等操作。

## 功能概览

| 模式 | 说明 |
| --- | --- |
| `clip` | CLIP 神经网络图像匹配（需补全逻辑） |
| `orb` | ORB 特征点匹配，根据参考图片在视频中定位匹配画面 |
| `face` | InsightFace 人脸检测，支持单人匹配过滤路人、马赛克、片段合并 |

## 环境要求

- Python 3.8+
- `opencv-python`
- `numpy`
- `insightface`（Face 分析引擎，自动下载模型）

## 安装

```bash
pip install -r requirements.txt
```

## 命令行参数

### 基础参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--video` | 单视频文件路径 | - |
| `--video_dir` | 批量模式：存放 mp4 视频的文件夹 | - |
| `--image` | CLIP/ORB 模式：参考图片路径 | - |
| `--face` | FACE 模式：目标人物参考照片 | - |
| `--out` | 输出文件夹 | `clips_output` |
| `--method` | 运行模式：clip / orb / face | `clip` |
| `--min_clip_len` | 有效片段最小帧数 | `30` |

### 人脸模式专属参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--gap_buffer` | 片段合并缓冲帧数，连续无人脸 < 该值不切割 | `10` |
| `--blur_face` | 1=开启人脸马赛克，0=关闭 | `0` |
| `--face_threshold` | 人脸匹配相似度阈值，越大越严格 | `0.6` |

### 全模式通用扩展参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--start_sec` | 视频处理起始秒数 | `0` |
| `--end_sec` | 视频处理结束秒数 | 片尾 |
| `--skip_frame` | 跳帧检测步长，1=逐帧，越大越快 | `1` |
| `--reverse` | 1=反向剪辑，保留不匹配/无人脸画面 | `0` |
| `--out_size` | 输出统一分辨率，格式：宽,高（如 1280,720） | 原始尺寸 |
| `--mute` | 1=输出静音视频 | `0` |
| `--concat_all` | 1=合并所有片段为单个完整 mp4 | `0` |
| `--out_format` | 输出格式：mp4 / gif | `mp4` |
| `--crop` | 画面裁切比例，格式：左,上,右,下（0~1） | 不裁切 |

### ORB / CLIP 匹配阈值

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--orb_threshold` | ORB 图像匹配相似度阈值 | `0.7` |
| `--clip_threshold` | CLIP 图像匹配相似度阈值 | `0.7` |

## 运行示例

### 1. 人脸单人匹配（匹配目标人物，过滤路人）

```bash
python clip_by_image.py \
  --method face \
  --video test_video.mp4 \
  --face target_person.jpg \
  --out person_only \
  --face_threshold 0.65 \
  --min_clip_len 30 \
  --gap_buffer 10
```

### 2. ORB 图像匹配（根据参考图裁切视频片段）

```bash
python clip_by_image.py \
  --method orb \
  --video test_video.mp4 \
  --image target.jpg \
  --out orb_result \
  --orb_threshold 0.7
```

### 3. 批量视频处理

```bash
python clip_by_image.py \
  --method face \
  --video_dir ./videos \
  --face target_person.jpg \
  --out batch_result \
  --skip_frame 2
```

### 4. 反向剪辑（保留无人脸画面）+ 合并输出

```bash
python clip_by_image.py \
  --method face \
  --video test_video.mp4 \
  --face target_person.jpg \
  --out no_person \
  --reverse 1 \
  --concat_all 1
```

### 5. 人脸马赛克 + 统一分辨率输出

```bash
python clip_by_image.py \
  --method face \
  --video test_video.mp4 \
  --face target_person.jpg \
  --out blurred_output \
  --blur_face 1 \
  --out_size 1280,720
```

## 输出

- **分段视频**：`face_clip_001.mp4`、`face_clip_002.mp4` … 或 `orb_clipped.mp4`
- **CSV 清单**：`clip_info.csv`（片段起止秒数、帧数、时长、是否匹配）
- **运行日志**：项目根目录自动生成 `run_log.txt`
- **合并视频**：`all_concat.mp4`（`--concat_all 1` 时生成）

## 项目结构

```text
video-image-clipper-skill/
├── clip_by_image.py      # 主程序
├── requirements.txt      # Python 依赖
├── README.md             # 项目说明
├── SKILL.md              # Skills.sh 元数据
├── package.json          # npm 包配置
└── .gitignore            # Git 忽略配置
```

## 注意事项

- InsightFace 首次运行会自动下载人脸检测模型文件
- Windows 下 InsightFace 推荐使用 CPU 推理（`CPUExecutionProvider`）
- 处理长视频建议设置 `--skip_frame 2` 或更大值加速
- 人脸匹配阈值 `--face_threshold` 建议范围 `0.6 ~ 0.8`

## License

MIT
