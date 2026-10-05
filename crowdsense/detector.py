import numpy as np
import torch
from ultralytics import YOLO

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class PersonDetector:
    """mode: 'standard' (one pass) or 'tiled' (SAHI slicing for small people)."""

    def __init__(self, weights="yolov8m.pt", imgsz=1280, conf=0.25,
                 mode="standard", tile=640, overlap=0.2):
        self.mode, self.imgsz, self.conf = mode, imgsz, conf
        self.tile, self.overlap = tile, overlap
        if mode == "tiled":
            from sahi import AutoDetectionModel
            self.sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics", model_path=weights,
                confidence_threshold=conf, device=DEVICE)
        else:
            self.model = YOLO(weights)

    def detect(self, img):
        """Returns an array of boxes [x1, y1, x2, y2] for people."""
        if self.mode == "tiled":
            from sahi.predict import get_sliced_prediction
            r = get_sliced_prediction(
                img, self.sahi_model,
                slice_height=self.tile, slice_width=self.tile,
                overlap_height_ratio=self.overlap, overlap_width_ratio=self.overlap,
                postprocess_type="NMS", postprocess_match_metric="IOU",
                postprocess_match_threshold=0.5, verbose=0)
            boxes = [o.bbox.to_xyxy() for o in r.object_prediction_list
                     if o.category.id == 0]
            return np.array(boxes, dtype=float).reshape(-1, 4)
        res = self.model(img, classes=[0], imgsz=self.imgsz,
                         conf=self.conf, verbose=False)[0]
        return res.boxes.xyxy.cpu().numpy()


def foot_points(boxes):
    return [((x1 + x2) / 2, y2) for x1, y1, x2, y2 in boxes]