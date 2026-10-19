import cv2
from ultralytics import YOLO
img = cv2.imread("crowd_2.png")
m = YOLO("yolov8m.pt")
for c in (0.05, 0.10, 0.15, 0.25, 0.40):
    r = m(img, classes=[0], imgsz=1280, conf=c, verbose=False)[0]
    print(f"conf={c:.2f} -> {len(r.boxes)} people")
    if c == 0.10:
        cv2.imwrite("outputs/crowd_2_c10.jpg", r.plot())
