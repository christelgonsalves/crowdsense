from ultralytics import YOLO
import cv2

IMG = "crowd.png"

configs = [
    ("yolov8n.pt", 640, 0.25),
    ("yolov8n.pt", 1280, 0.25),
    ("yolov8m.pt", 1280, 0.25),
    ("yolov8m.pt", 1280, 0.15),
]

for weights, imgsz, conf in configs:
    model = YOLO(weights)
    res = model(IMG, classes=[0], imgsz=imgsz, conf=conf, verbose=False)[0]
    print(f"{weights:12} imgsz={imgsz:5} conf={conf:.2f} -> {len(res.boxes)} people")
    cv2.imwrite(f"out_{weights[:-3]}_{imgsz}_{conf}.jpg", res.plot())