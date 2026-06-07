# Video Image Clipper Skill

Detect a target image inside a video and extract the matching segments as separate clips.

## Overview

This project locates a given reference image inside a video by sampling frames and using ORB feature matching. When the image appears, the script groups nearby matching frames into contiguous segments and exports each segment with `ffmpeg`.

## Requirements

- Python 3.8+
- `ffmpeg` installed and available on `PATH`
- `opencv-python`
- `numpy`

## Install

```bash
python -m pip install -r requirements.txt
```

## Basic usage

```bash
python clip_by_image.py --video input.mp4 --image target.jpg --out clips
```

This creates an output directory named `clips/` and writes files like `clip_001.mp4`, `clip_002.mp4`, etc.

## Options

- `--sample-rate`: sample every Nth frame to speed up detection (default: `1`)
- `--min-duration`: minimum duration of each output clip in seconds (default: `0.5`)
- `--match-threshold`: ORB match score threshold, from `0.0` to `1.0` (default: `0.08`)

## Advanced example

```bash
python clip_by_image.py \
  --video input.mp4 \
  --image target.jpg \
  --out clips \
  --sample-rate 2 \
  --min-duration 1.0 \
  --match-threshold 0.12
```

## How it works

1. Read the reference image and compute ORB keypoints/descriptors.
2. Open the video and sample frames.
3. Match each sampled frame against the reference image.
4. Collect frames with enough good feature matches.
5. Merge nearby matching frames into segments.
6. Use `ffmpeg` to cut each segment from the original video.

## Packaging as a skills.sh skill

If you want to publish it as a skills.sh skill:

```bash
npx skills init
```

Then push the repository to GitHub and install with:

```bash
npx skills add Dolphin2025127/video-image-clipper-skill
```

## Notes

- The current implementation is a lightweight ORB-based matcher. It works well for moderately textured targets, but may be less robust for low-detail or heavily distorted images.
- For better accuracy, you can replace ORB matching with a neural embedding model such as CLIP or a specialized object detector.
- `ffmpeg` is used in copy mode, so output clips are fast-exported without re-encoding.
