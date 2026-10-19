import cv2
from ultralytics import YOLO

img = cv2.imread(r"C:\Projects\data\mall_dataset\frames\seq_000500.jpg")
for w in ("yolov8m.pt", "best.pt"):
    m = YOLO(w)
    for c in (0.03, 0.05, 0.12, 0.25):
        r = m(img, classes=[0], imgsz=1280, conf=c, verbose=False)[0]
        print(f"{w:12} conf={c:.2f} -> {len(r.boxes)} people")
        if c == 0.05:
            cv2.imwrite(f"outputs/mall_{w[:-3]}_c05.jpg", r.plot())
