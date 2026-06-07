# Video Image Clipper Skill

Detect a target image inside a video and automatically extract (clip) the time segments where the image appears.

Prerequisites:
- `ffmpeg` available on PATH
- Python 3.8+

Install Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Quick usage:

```bash
python clip_by_image.py --video input.mp4 --image target.jpg --out clips
```

Options:
- `--sample-rate`: sample every Nth frame (default 1)
- `--min-duration`: minimum clip duration in seconds (default 0.5)
- `--match-threshold`: threshold for matching score (0..1)

Packaging as a skills.sh skill:

1. Initialize skill metadata locally:

```bash
npx skills init
```

2. Publish (example):

```bash
# After pushing to github.com/<owner>/video-image-clipper-skill
npx skills add <owner>/video-image-clipper-skill
```

Notes:
- This implementation uses ORB feature matching (OpenCV) as a lightweight approach. For higher robustness, replace matching with a learned embedding model (e.g., CLIP) or template-matching tuned to your domain.
