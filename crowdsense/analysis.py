import cv2
import numpy as np

# (minimum people in a zone, name, BGR colour)
LEVELS = [
    (0, "Low",      (0, 200, 0)),
    (4, "Medium",   (0, 215, 255)),
    (6, "High",     (0, 128, 255)),
    (7, "Critical", (0, 0, 255)),
]


def level_of(count):
    chosen = LEVELS[0]
    for lvl in LEVELS:
        if count >= lvl[0]:
            chosen = lvl
    return chosen


def zone_counts(points, shape, rows=3, cols=4):
    h, w = shape[:2]
    counts = np.zeros((rows, cols), int)
    for x, y in points:
        c = min(max(int(x / (w / cols)), 0), cols - 1)
        r = min(max(int(y / (h / rows)), 0), rows - 1)
        counts[r, c] += 1
    return counts


def draw_zones(img, counts):
    rows, cols = counts.shape
    h, w = img.shape[:2]
    out, overlay = img.copy(), img.copy()
    for r in range(rows):
        for c in range(cols):
            x1, y1 = int(c * w / cols), int(r * h / rows)
            x2, y2 = int((c + 1) * w / cols), int((r + 1) * h / rows)
            _, name, col = level_of(counts[r, c])
            cv2.rectangle(overlay, (x1, y1), (x2, y2), col, -1)
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            cv2.putText(out, f"{counts[r, c]} | {name}", (x1 + 8, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return cv2.addWeighted(overlay, 0.25, out, 0.75, 0)


def draw_heatmap(img, points):
    h, w = img.shape[:2]
    d = np.zeros((h, w), np.float32)
    for x, y in points:
        if 0 <= x < w and 0 <= y < h:
            d[int(y), int(x)] += 1
    d = cv2.GaussianBlur(d, (0, 0), sigmaX=w / 40)
    if d.max() == 0:
        return img.copy()
    norm = (d / d.max() * 255).astype(np.uint8)
    blend = cv2.addWeighted(img, 0.45, cv2.applyColorMap(norm, cv2.COLORMAP_JET), 0.55, 0)
    out = img.copy()
    out[norm > 25] = blend[norm > 25]
    return out