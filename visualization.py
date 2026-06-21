import cv2
import numpy as np
from config import *
from IPM import project_point

def color_for(cls_name):
    return CLASS_COLORS.get(cls_name, DEFAULT_COLOR)


def draw_perspective(frame, detections):
    vis = frame.copy()
    for (x1, y1, x2, y2, conf, cls_id, cls_name) in detections:
        c = color_for(cls_name)
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
        # bottom-centre dot (ground contact point)
        bx, by = (x1 + x2) // 2, y2
        cv2.circle(vis, (bx, by), 5, c, -1)
        label = f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = y1 - 5 if y1 > th + 5 else y2 + th + 5
        cv2.rectangle(vis, (x1, ly - th - 2), (x1 + tw + 4, ly + 2), c, -1)
        cv2.putText(vis, label, (x1 + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    # Label
    cv2.putText(vis, "Perspective view", (8, 22),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def draw_bev(bev_frame, detections, H):
    vis = bev_frame.copy()

    for (x1, y1, x2, y2, conf, cls_id, cls_name) in detections:
        # Ground contact = bottom-centre of box
        bx, by = (x1 + x2) / 2.0, float(y2)
        bev_x, bev_y = project_point((bx, by), H)
        bev_x, bev_y = int(bev_x), int(bev_y)

        # Only draw if inside BEV canvas
        if 0 <= bev_x < BEV_W and 0 <= bev_y < BEV_H:
            c = color_for(cls_name)
            cv2.circle(vis, (bev_x, bev_y), 8, c, -1)
            cv2.circle(vis, (bev_x, bev_y), 8, (255, 255, 255), 1)
            cv2.putText(vis, cls_name[:3],
                        (bev_x + 10, bev_y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw ego vehicle indicator at bottom-centre
    ego_x = BEV_W // 2
    ego_pts = np.array([
        [ego_x - 15, BEV_H - 10],
        [ego_x + 15, BEV_H - 10],
        [ego_x,      BEV_H - 50],
    ], dtype=np.int32)
    cv2.fillPoly(vis, [ego_pts], (60, 60, 200))
    cv2.putText(vis, "ego", (ego_x - 12, BEV_H - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 255), 1)

    # Draw IPM source trapezoid projected into BEV
    src_bev = np.array(DEFAULT_DST, dtype=np.int32)
    cv2.polylines(vis, [src_bev], True, (80, 80, 80), 1)

    # Distance grid lines (every 100px ≈ some real-world distance)
    for gy in range(0, BEV_H, 100):
        cv2.line(vis, (0, gy), (BEV_W, gy), (60, 60, 60), 1)
        cv2.putText(vis, f"{(BEV_H - gy) // 10}m",
                    (4, gy - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

    cv2.putText(vis, "BEV (top-down)", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def make_legend(h):
    """Small legend panel."""
    leg = np.zeros((h, 140, 3), dtype=np.uint8)
    cv2.putText(leg, "Legend", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
    y = 44
    for name, c in CLASS_COLORS.items():
        cv2.circle(leg, (16, y), 7, c, -1)
        cv2.putText(leg, name, (30, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 26
    cv2.putText(leg, "● ground pt", (6, y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)
    cv2.putText(leg, "▲ ego car", (6, y + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (130, 130, 220), 1)
    return leg


def compose(persp_vis, bev_vis):
    """Stack perspective | BEV side by side, resized to same height."""
    H = max(persp_vis.shape[0], bev_vis.shape[0])
    # Resize perspective to height H keeping aspect
    pw = int(persp_vis.shape[1] * H / persp_vis.shape[0])
    p = cv2.resize(persp_vis, (pw, H))
    # Resize BEV to height H
    bw = int(bev_vis.shape[1] * H / bev_vis.shape[0])
    b = cv2.resize(bev_vis, (bw, H))
    leg = make_legend(H)
    return np.hstack([p, b, leg])
