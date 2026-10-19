import cv2
from ultralytics import YOLO

mall = cv2.imread(r"C:\Projects\data\mall_dataset\frames\seq_000500.jpg")
plaza = cv2.imread("crowd.png")

for w in ("yolov8m.pt", "best.pt"):
    m = YOLO(w)
    for s in (640, 960, 1280, 1920):
        r = m(mall, classes=[0], imgsz=s, conf=0.12, verbose=False)[0]
        print(f"MALL  {w:11} imgsz={s} -> {len(r.boxes)} people")
    for s in (960, 1280, 1920):
        r = m(plaza, classes=[0], imgsz=s, conf=0.25, verbose=False)[0]
        print(f"PLAZA {w:11} imgsz={s} -> {len(r.boxes)} people")
