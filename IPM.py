import cv2
import numpy as np
from config import DEFAULT_SRC , DEFAULT_DST , BEV_W , BEV_H

def compute_homography(src_pts=DEFAULT_SRC, dst_pts=DEFAULT_DST):
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H


def warp_to_bev(frame, H, bev_size=(BEV_W, BEV_H)):
    return cv2.warpPerspective(frame, H, bev_size,
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(40, 40, 40))


def project_point(pt, H):
    """Project a single (x, y) point through homography H."""
    p = np.array([pt[0], pt[1], 1.0], dtype=np.float64)
    q = H @ p
    return (q[0] / q[2], q[1] / q[2])
