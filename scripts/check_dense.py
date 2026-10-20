import cv2
from ultralytics import YOLO

img = cv2.imread("crowd_4.png")
if img is None:
    raise SystemExit("crowd_4.png not found in C:\\Projects\\crowdsense")
print("image size:", img.shape[1], "x", img.shape[0])

m = YOLO("yolov8m.pt")
for scale in (1, 2):
    im = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    for c in (0.10, 0.20, 0.30):
        r = m(im, classes=[0], imgsz=1280, conf=c, verbose=False)[0]
        print(f"scale x{scale} conf={c:.2f} -> {len(r.boxes)} people")
        if scale == 2 and c == 0.10:
            cv2.imwrite("outputs/crowd_4_x2_c10.jpg", r.plot())
