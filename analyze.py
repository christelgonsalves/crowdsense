import cv2
import numpy as np
from ultralytics import YOLO

IMG = "crowd.png"
WEIGHTS, IMGSZ, CONF = "yolov8m.pt", 1280, 0.25
ROWS, COLS = 3, 4

# (minimum people in a cell, name, BGR colour)
LEVELS = [
    (0, "Low",      (0, 200, 0)),
    (2, "Medium",   (0, 215, 255)),
    (4, "High",     (0, 128, 255)),
    (6, "Critical", (0, 0, 255)),
]


def level_of(count):
    chosen = LEVELS[0]
    for lvl in LEVELS:
        if count >= lvl[0]:
            chosen = lvl
    return chosen


model = YOLO(WEIGHTS)
img = cv2.imread(IMG)
h, w = img.shape[:2]

res = model(img, classes=[0], imgsz=IMGSZ, conf=CONF, verbose=False)[0]
boxes = res.boxes.xyxy.cpu().numpy()
feet = [((x1 + x2) / 2, y2) for x1, y1, x2, y2 in boxes]  # where people stand
print("Total people:", len(feet))

# ---------- 1. Density heatmap ----------
dmap = np.zeros((h, w), np.float32)
for x, y in feet:
    if 0 <= x < w and 0 <= y < h:
        dmap[int(y), int(x)] += 1
dmap = cv2.GaussianBlur(dmap, (0, 0), sigmaX=w / 40)
norm = (dmap / dmap.max() * 255).astype(np.uint8) if dmap.max() > 0 else dmap.astype(np.uint8)
color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
blend = cv2.addWeighted(img, 0.45, color, 0.55, 0)
heat = img.copy()
mask = norm > 25                      # tint only where people are
heat[mask] = blend[mask]
cv2.putText(heat, f"People: {len(feet)}", (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
cv2.imwrite("result_heatmap.jpg", heat)

# ---------- 2. Grid zones with crowd levels ----------
counts = np.zeros((ROWS, COLS), int)
for x, y in feet:
    c = min(int(x / (w / COLS)), COLS - 1)
    r = min(int(y / (h / ROWS)), ROWS - 1)
    counts[r, c] += 1

zones = img.copy()
overlay = img.copy()
for r in range(ROWS):
    for c in range(COLS):
        x1, y1 = int(c * w / COLS), int(r * h / ROWS)
        x2, y2 = int((c + 1) * w / COLS), int((r + 1) * h / ROWS)
        _, name, col = level_of(counts[r, c])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
        cv2.rectangle(zones, (x1, y1), (x2, y2), col, 2)
        cv2.putText(zones, f"{counts[r, c]} | {name}", (x1 + 8, y1 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
zones = cv2.addWeighted(overlay, 0.25, zones, 0.75, 0)
cv2.imwrite("result_zones.jpg", zones)

print("Counts per zone (rows x cols):")
print(counts)