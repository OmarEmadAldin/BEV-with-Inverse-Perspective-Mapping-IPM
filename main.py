import cv2
import numpy as np
import argparse
import os
import json
from config import *
from visualization import *
from IPM import compute_homography , warp_to_bev
from detector import detect

def process_video(video_path, H, out_path="bev_video.mp4", max_frames=300):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    ret, frame = cap.read()
    assert ret

    bev_test = warp_to_bev(frame, H)
    out_frame = compose(draw_perspective(frame, []), draw_bev(bev_test, [], H))
    out_h, out_w = out_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    idx = 0
    while idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detect(frame)
        bev = warp_to_bev(frame, H)
        p_vis = draw_perspective(frame, detections)
        b_vis = draw_bev(bev, detections, H)
        out_frame = compose(p_vis, b_vis)
        writer.write(out_frame)
        idx += 1
        print(f"  Frame {idx}  dets={len(detections)}", end="\r")

    cap.release()
    writer.release()
    print(f"\n  Done → {out_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BEV IPM + YOLO")
    parser.add_argument("--video",     default=None)
    parser.add_argument("--out",       default=None)

    args = parser.parse_args()

    H = compute_homography()
    if args.video:
        out = args.out or "bev_video.mp4"
        process_video(args.video, H, out)
