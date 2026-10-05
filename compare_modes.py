import cv2
from crowdsense.detector import PersonDetector, foot_points
from crowdsense.analysis import zone_counts, draw_zones, draw_heatmap

IMG = "crowd.png"
img = cv2.imread(IMG)

setups = [
    ("standard_640",  dict(mode="standard", imgsz=640)),
    ("standard_1280", dict(mode="standard", imgsz=1280)),
    ("tiled_640",     dict(mode="tiled", tile=640)),
]

for name, kw in setups:
    det = PersonDetector(weights="yolov8m.pt", conf=0.25, **kw)
    boxes = det.detect(img)
    pts = foot_points(boxes)
    print(f"{name:14} -> {len(boxes)} people")
    out = draw_zones(draw_heatmap(img, pts), zone_counts(pts, img.shape))
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.imwrite(f"cmp_{name}.jpg", out)