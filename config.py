import numpy as np

DEFAULT_SRC = np.float32([
    [595, 210],   # far-left   (vanishing zone)
    [685, 210],   # far-right
    [1100, 340],  # near-right
    [220, 340],   # near-left
])

DEFAULT_DST_SIZE = (400, 600)   # (width, height) of BEV canvas
DEFAULT_DST = np.float32([
    [100, 0],               # far-left  → top-left in BEV
    [300, 0],               # far-right → top-right
    [300, DEFAULT_DST_SIZE[1]],  # near-right → bottom-right
    [100, DEFAULT_DST_SIZE[1]],  # near-left  → bottom-left
])

BEV_W, BEV_H = DEFAULT_DST_SIZE
YOLO_MODEL    = "yolov8n.pt"        # nano — fast; swap for yolov8s/m for accuracy
CONF_THRESH   = 0.35
VEHICLE_CLASSES = {2, 3, 5, 7}     # COCO: car, motorcycle, bus, truck
PERSON_CLASS    = 0
CLASS_COLORS = {
    "person":     (50,  200,  50),
    "car":        (50,  120, 255),
    "truck":      (255, 140,  50),
    "bus":        (200,  50, 200),
    "motorcycle": (50,  220, 220),
}
DEFAULT_COLOR = (200, 200, 200)

