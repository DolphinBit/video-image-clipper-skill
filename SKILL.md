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

```bash
python clip_by_image.py --video input.mp4 --image target.jpg --out clips
```

## Requirements

- Python 3.8+
- `ffmpeg` installed and available on PATH
- `opencv-python`
- `numpy`
