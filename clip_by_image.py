#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import os
import subprocess


def parse_args():
    p = argparse.ArgumentParser(description="Clip video segments where a target image appears")
    p.add_argument("--video", "-v", required=True, help="Input video file")
    p.add_argument("--image", "-i", required=True, help="Target image file to match in the video")
    p.add_argument("--out", "-o", default="clips", help="Output directory for clips")
    p.add_argument("--sample-rate", type=int, default=1, help="Sample every Nth frame (default=1)")
    p.add_argument("--min-duration", type=float, default=0.5, help="Minimum clip duration in seconds")
    p.add_argument("--match-threshold", type=float, default=0.08, help="Match score threshold (0..1)")
    return p.parse_args()


def find_matching_frames(video_path, image_path, sample_rate=1, match_threshold=0.08):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orb = cv2.ORB_create(1000)

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Cannot open image: {image_path}")
    kp1, des1 = orb.detectAndCompute(img, None)
    if des1 is None or len(kp1) == 0:
        raise RuntimeError("Failed to compute descriptors for target image")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matching_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(gray, None)
            score = 0.0
            if des2 is not None and len(kp2) > 0:
                matches = matcher.match(des1, des2)
                good = [m for m in matches if m.distance < 60]
                score = len(good) / max(1, len(kp1))
            if score >= match_threshold:
                matching_frames.append(frame_idx)
        frame_idx += 1

    cap.release()
    return matching_frames, fps


def frames_to_segments(frames, fps, max_gap_frames=3, min_duration=0.5):
    if not frames:
        return []
    frames = sorted(frames)
    segments = []
    start = frames[0]
    prev = frames[0]
    for f in frames[1:]:
        if f - prev <= max_gap_frames:
            prev = f
            continue
        else:
            s_time = start / fps
            e_time = (prev + 1) / fps
            if e_time - s_time >= min_duration:
                segments.append((s_time, e_time))
            start = f
            prev = f
    # last segment
    s_time = start / fps
    e_time = (prev + 1) / fps
    if e_time - s_time >= min_duration:
        segments.append((s_time, e_time))
    return segments


def cut_segments_with_ffmpeg(video_path, segments, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    outputs = []
    for idx, (s, e) in enumerate(segments, start=1):
        out_path = os.path.join(out_dir, f"clip_{idx:03d}.mp4")
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-ss",
            str(s),
            "-to",
            str(e),
            "-c",
            "copy",
            out_path,
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            outputs.append(out_path)
        except subprocess.CalledProcessError:
            print(f"ffmpeg failed to create clip {out_path}")
    return outputs


def main():
    args = parse_args()
    frames, fps = find_matching_frames(args.video, args.image, sample_rate=args.sample_rate, match_threshold=args.match_threshold)
    segments = frames_to_segments(frames, fps, max_gap_frames=max(1, int(args.sample_rate * 3)), min_duration=args.min_duration)
    if not segments:
        print("No segments found matching the target image.")
        return
    outputs = cut_segments_with_ffmpeg(args.video, segments, args.out)
    print(f"Created {len(outputs)} clip(s) in {args.out}")


if __name__ == "__main__":
    main()
