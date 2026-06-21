from config import YOLO_MODEL , CONF_THRESH
from ultralytics import YOLO
_yolo_model = None

def load_yolo():
    if _yolo_model is None:

        _yolo_model = YOLO(YOLO_MODEL)
    return _yolo_model


def detect(frame):
    """Run YOLOv8, return list of (x1,y1,x2,y2,conf,cls_id,cls_name)."""
    model = load_yolo()
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf  = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        detections.append((x1, y1, x2, y2, conf, cls_id, cls_name))
    return detections
