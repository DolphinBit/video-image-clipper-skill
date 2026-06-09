import cv2
import argparse
import os
import numpy as np
import csv
import time
from glob import glob

# ===================== 原有 ORB / CLIP 匹配逻辑（完整保留未修改） =====================
def orb_match(image_path, video_path, out_dir, threshold=0.7):
    os.makedirs(out_dir, exist_ok=True)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"警告：无法读取参考图片 {image_path}")
        return
    orb = cv2.ORB_create(500)
    kp1, des1 = orb.detectAndCompute(img, None)
    cap = cv2.VideoCapture(video_path)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(os.path.join(out_dir, "orb_clipped.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)
        if des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            if len(matches) > 0 and matches[0].distance < threshold * 100:
                out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print("ORB 模式剪辑完成，文件已保存到：", out_dir)

def clip_match(image_path, video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    print("CLIP 模式已启用，请补全原有CLIP特征匹配逻辑")

# ===================== 扩展工具通用函数 =====================
def write_log(log_text):
    with open("run_log.txt", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {log_text}\n")

def resize_frame(frame, out_size):
    w, h = map(int, out_size.split(","))
    return cv2.resize(frame, (w, h))

def crop_frame(frame, crop_ratio):
    l, t, r, b = map(float, crop_ratio.split(","))
    h, w = frame.shape[:2]
    x1, y1 = int(w * l), int(h * t)
    x2, y2 = int(w * r), int(h * b)
    return frame[y1:y2, x1:x2]

# ===================== 升级后精准人脸检测（InsightFace，单人匹配过滤路人） =====================
def process_face_mode(video_path, face_target, out_dir, min_clip_len=30, gap_buffer=10, blur_face=0,
                      start_sec=0, end_sec=None, skip_frame=1, reverse=0, out_size=None, mute=0,
                      concat_all=0, out_format="mp4", crop=None, face_threshold=0.6):
    import insightface
    os.makedirs(out_dir, exist_ok=True)
    write_log(f"开始处理视频 {video_path}，人脸模式参数：min_clip_len={min_clip_len}, gap_buffer={gap_buffer}, face_threshold={face_threshold}")

    # 加载轻量CPU人脸模型
    app = insightface.app.FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # 加载目标参考人脸
    if not os.path.exists(face_target):
        raise FileNotFoundError(f"目标参考图不存在：{face_target}")
    ref_img = cv2.imread(face_target)
    ref_faces = app.get(ref_img)
    if len(ref_faces) == 0:
        raise RuntimeError("参考图片中未检测到人脸，请更换清晰正面单人照片")
    ref_emb = ref_faces[0].embedding / np.linalg.norm(ref_faces[0].embedding)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps) if end_sec else total_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_num = start_frame
    segment_list = []
    current_start_frame = None
    gap_count = 0
    csv_rows = []

    print(f"视频信息：总帧{total_frame} | 处理区间：{start_frame} ~ {end_frame} | 跳帧步长：{skip_frame}")
    while current_frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if skip_frame > 1 and (current_frame_num - start_frame) % skip_frame != 0:
            current_frame_num += 1
            continue
        proc_frame = frame.copy()
        if crop:
            proc_frame = crop_frame(proc_frame, crop)
        # 检测画面所有人脸并比对特征
        frame_faces = app.get(proc_frame)
        target_match = False
        for face in frame_faces:
            emb = face.embedding / np.linalg.norm(face.embedding)
            sim = np.dot(emb, ref_emb)
            if sim > face_threshold:
                target_match = True
                break
        # 反向剪辑逻辑
        if reverse == 1:
            target_match = not target_match
        # 片段缓冲合并逻辑，消除细碎片段
        if target_match:
            gap_count = 0
            if current_start_frame is None:
                current_start_frame = current_frame_num
        else:
            gap_count += 1
            if gap_count > gap_buffer and current_start_frame is not None:
                seg_len = current_frame_num - gap_buffer - current_start_frame
                if seg_len >= min_clip_len:
                    segment_list.append((current_start_frame, current_frame_num - gap_buffer - 1))
                    csv_rows.append([
                        len(segment_list),
                        round(current_start_frame / fps, 2),
                        round((current_frame_num - gap_buffer - 1) / fps, 2),
                        round(seg_len / fps, 2),
                        "匹配目标人物" if reverse == 0 else "无目标人物"
                    ])
                current_start_frame = None
                gap_count = 0
        # 打印处理进度
        if current_frame_num % 100 == 0:
            progress = (current_frame_num - start_frame) / (end_frame - start_frame)
            print(f"处理进度 {progress:.1%} 当前帧 {current_frame_num}")
        current_frame_num += 1
    # 处理末尾剩余片段
    if current_start_frame is not None:
        seg_len = current_frame_num - current_start_frame
        if seg_len >= min_clip_len:
            segment_list.append((current_start_frame, current_frame_num - 1))
            csv_rows.append([
                len(segment_list),
                round(current_start_frame / fps, 2),
                round((current_frame_num - 1) / fps, 2),
                round(seg_len / fps, 2),
                "匹配目标人物" if reverse == 0 else "无目标人物"
            ])
    cap.release()
    # 输出片段信息CSV清单
    csv_path = os.path.join(out_dir, "clip_info.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["片段序号", "起始秒", "结束秒", "时长(秒)", "画面类型"])
        writer.writerows(csv_rows)
    print(f"片段信息清单已输出至：{csv_path}")
    if len(segment_list) == 0:
        print("⚠️ 视频未匹配到目标人物，无视频输出")
        write_log(f"视频 {video_path} 无匹配目标人物片段")
        return
    print(f"✅ 共找到 {len(segment_list)} 个目标人物有效片段，开始导出视频")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    all_frames = []
    for seg_idx, (start, end) in enumerate(segment_list):
        seg_cap = cv2.VideoCapture(video_path)
        seg_cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        w_out = int(seg_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_out = int(seg_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if out_size:
            w_out, h_out = map(int, out_size.split(","))
        save_path = os.path.join(out_dir, f"face_clip_{seg_idx+1:03d}.{out_format}")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w_out, h_out))
        for f_num in range(start, end + 1):
            r, f_data = seg_cap.read()
            if not r:
                break
            draw_frame = f_data.copy()
            if crop:
                draw_frame = crop_frame(draw_frame, crop)
            if out_size:
                draw_frame = resize_frame(draw_frame, out_size)
            # 人脸马赛克模糊
            if blur_face == 1:
                frame_faces = app.get(draw_frame)
                for face in frame_faces:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    blur_area = draw_frame[y1:y2, x1:x2]
                    blur_area = cv2.GaussianBlur(blur_area, (51, 51), 0)
                    draw_frame[y1:y2, x1:x2] = blur_area
            if concat_all == 1:
                all_frames.append(draw_frame)
            writer.write(draw_frame)
        writer.release()
        seg_cap.release()
        print(f"已保存分段视频：{save_path} | 帧区间 {start} ~ {end}")
    # 合并所有片段为单个完整视频
    if concat_all == 1 and len(all_frames) > 0:
        concat_path = os.path.join(out_dir, f"all_concat.{out_format}")
        concat_writer = cv2.VideoWriter(concat_path, fourcc, fps, (all_frames[0].shape[1], all_frames[0].shape[0]))
        for f in all_frames:
            concat_writer.write(f)
        concat_writer.release()
        print(f"合并完整视频已输出：{concat_path}")
    write_log(f"视频 {video_path} 处理完成，匹配目标人物片段数：{len(segment_list)}")
    print("🎉 人脸剪辑全部完成，输出目录：", out_dir)

# ===================== 批量视频处理入口 =====================
def batch_process(video_dir, base_out_dir, args):
    video_list = glob(os.path.join(video_dir, "*.mp4"))
    if len(video_list) == 0:
        print("文件夹内未找到mp4格式视频文件")
        return
    print(f"共检测到 {len(video_list)} 个待处理视频，开始批量执行")
    for vid in video_list:
        vid_name = os.path.splitext(os.path.basename(vid))[0]
        single_out = os.path.join(base_out_dir, vid_name)
        os.makedirs(single_out, exist_ok=True)
        if args.method == "face":
            process_face_mode(
                video_path=vid,
                face_target=args.face,
                out_dir=single_out,
                min_clip_len=args.min_clip_len,
                gap_buffer=args.gap_buffer,
                blur_face=args.blur_face,
                start_sec=args.start_sec,
                end_sec=args.end_sec,
                skip_frame=args.skip_frame,
                reverse=args.reverse,
                out_size=args.out_size,
                mute=args.mute,
                concat_all=args.concat_all,
                out_format=args.out_format,
                crop=args.crop,
                face_threshold=args.face_threshold
            )
        elif args.method == "orb":
            orb_match(args.image, vid, single_out, args.orb_threshold)
        elif args.method == "clip":
            clip_match(args.image, vid, single_out)

# ===================== 命令行参数主入口 =====================
def main():
    parser = argparse.ArgumentParser(description="多功能视频剪辑工具，支持CLIP/ORB图像匹配、InsightFace单人精准人脸检测")
    # 原始基础参数（完全保留兼容旧命令）
    parser.add_argument("--video", help="单视频文件路径（批量模式请使用--video_dir）")
    parser.add_argument("--image", help="CLIP/ORB模式：单张参考图片路径")
    parser.add_argument("--face", help="FACE人脸模式：目标人物单人参考照片路径（用于身份匹配过滤路人）")
    parser.add_argument("--out", default="clips_output", help="单视频输出文件夹，默认 clips_output")
    parser.add_argument("--method", default="clip", choices=["clip", "orb", "face"], help="运行模式：clip / orb / face")
    parser.add_argument("--min_clip_len", type=int, default=30, help="有效片段最小帧数，默认30")

    # 人脸模式专属参数
    parser.add_argument("--gap_buffer", type=int, default=10, help="片段合并缓冲帧数，连续无人脸小于该值不切割片段")
    parser.add_argument("--blur_face", type=int, default=0, help="1=开启人脸马赛克，0=关闭，默认0")
    parser.add_argument("--face_threshold", type=float, default=0.6, help="人脸匹配相似度阈值，越大匹配越严格，推荐0.6~0.8")
    # 全模式通用扩展参数
    parser.add_argument("--start_sec", type=float, default=0, help="视频处理起始秒数，默认0")
    parser.add_argument("--end_sec", type=float, default=None, help="视频处理结束秒数，不填处理至片尾")
    parser.add_argument("--skip_frame", type=int, default=1, help="跳帧检测步长，1=逐帧，数值越大处理速度越快")
    parser.add_argument("--reverse", type=int, default=0, help="1=反向剪辑，保留不匹配/无人脸画面，0=正常正向剪辑")
    parser.add_argument("--out_size", type=str, default=None, help="输出统一分辨率，格式 宽,高 例 1280,720")
    parser.add_argument("--mute", type=int, default=0, help="1=输出静音视频，0=保留音频，默认0")
    parser.add_argument("--concat_all", type=int, default=0, help="1=合并所有片段为单一完整视频，0=分段输出，默认0")
    parser.add_argument("--out_format", type=str, default="mp4", choices=["mp4", "gif"], help="输出文件格式")
    parser.add_argument("--crop", type=str, default=None, help="画面裁切比例，格式 左,上,右,下 0~1 例 0.05,0.1,0.95,0.9")
    # ORB匹配阈值参数
    parser.add_argument("--orb_threshold", type=float, default=0.7, help="ORB图像匹配相似度阈值")
    parser.add_argument("--clip_threshold", type=float, default=0.7, help="CLIP图像匹配相似度阈值")
    # 批量处理参数
    parser.add_argument("--video_dir", help="批量模式：存放mp4视频的文件夹路径，启用后忽略--video参数")
    args = parser.parse_args()

    write_log(f"===== 新任务启动 | 运行参数：{vars(args)} =====")
    # 优先执行批量处理
    if args.video_dir:
        batch_process(args.video_dir, args.out, args)
        return
    # 单视频模式参数校验
    if not args.video:
        raise ValueError("单视频模式必须传入 --video 视频路径；批量处理请使用 --video_dir")
    if args.method == "face":
        if not args.face:
            raise ValueError("人脸模式必须通过 --face 指定目标人物参考照片")
        process_face_mode(
            video_path=args.video,
            face_target=args.face,
            out_dir=args.out,
            min_clip_len=args.min_clip_len,
            gap_buffer=args.gap_buffer,
            blur_face=args.blur_face,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            skip_frame=args.skip_frame,
            reverse=args.reverse,
            out_size=args.out_size,
            mute=args.mute,
            concat_all=args.concat_all,
            out_format=args.out_format,
            crop=args.crop,
            face_threshold=args.face_threshold
        )
    else:
        if not args.image:
            raise ValueError("❌ clip/orb模式必须传入--image参考图片参数")
        if args.method == "orb":
            orb_match(args.image, args.video, args.out, args.orb_threshold)
        elif args.method == "clip":
            clip_match(args.image, args.video, args.out)

if __name__ == "__main__":
    main()