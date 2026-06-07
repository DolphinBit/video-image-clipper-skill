---
name: video-image-clipper
version: 0.1.0
author: Dolphin2025127
description: Detect a target image in a video and clip the matching segments automatically.
tags:
  - video
  - image
  - clip
  - computer vision
---

# Video Image Clipper

This skill detects a target image within a video and extracts the matching time segments as separate clips.

## Usage

### Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Run the skill

```bash
python clip_by_image.py --video input.mp4 --image target.jpg --out clips
```

### Example with advanced options

```bash
python clip_by_image.py \
  --video input.mp4 \
  --image target.jpg \
  --out clips \
  --sample-rate 2 \
  --min-duration 1.0 \
  --match-threshold 0.1
```

### What it does

- Samples the video frames at the interval controlled by `--sample-rate`
- Uses ORB feature matching between the target image and each sampled frame
- Groups nearby matching frames into clip segments
- Uses `ffmpeg` to export each matching segment to `clips/clip_XXX.mp4`

## Requirements

- Python 3.8+
- `ffmpeg` installed and available on PATH
- `opencv-python`
- `numpy`
