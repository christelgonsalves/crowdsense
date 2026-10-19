import cv2
from crowdsense.detector import PersonDetector

img = cv2.imread("crowd.png")
setups = [
    ("Default 640, conf 0.25", dict(mode="standard", imgsz=640, conf=0.25)),
    ("Tuned 1280, conf 0.25", dict(mode="standard", imgsz=1280, conf=0.25)),
    ("Tiled 640, conf 0.35", dict(mode="tiled", tile=640, conf=0.35)),
]
panels = []
for title, kw in setups:
    boxes = PersonDetector(weights="yolov8m.pt", **kw).detect(img)
    print(f"{title}: {len(boxes)} people")
    out = img.copy()
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    fs = out.shape[1] / 1100
    cv2.rectangle(out, (0, 0), (out.shape[1], int(70 * fs)), (0, 0, 0), -1)
    cv2.putText(out, f"{title}: {len(boxes)} people", (15, int(48 * fs)),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), max(2, int(fs * 2)))
    panels.append(cv2.resize(out, (out.shape[1] * 540 // out.shape[0], 540)))
cv2.imwrite("assets/plaza_comparison.jpg", cv2.hconcat(panels))
print("Saved assets/plaza_comparison.jpg")
