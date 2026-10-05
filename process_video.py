import argparse
import csv
import cv2
from crowdsense.detector import PersonDetector, foot_points
from crowdsense.analysis import zone_counts, draw_zones, draw_heatmap, level_of

p = argparse.ArgumentParser()
p.add_argument("--source", default="crowd_720.mp4")
p.add_argument("--out", default="outputs/result_video.mp4")
p.add_argument("--weights", default="yolov8m.pt")
p.add_argument("--imgsz", type=int, default=1280)
p.add_argument("--conf", type=float, default=0.30)
p.add_argument("--skip", type=int, default=3, help="detect every Nth frame (speed on CPU)")
p.add_argument("--max_frames", type=int, default=0, help="0 = whole video")
a = p.parse_args()

det = PersonDetector(weights=a.weights, imgsz=a.imgsz, conf=a.conf, mode="standard")

cap = cv2.VideoCapture(a.source)
w, h = int(cap.get(3)), int(cap.get(4))
fps = cap.get(5) or 25
writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

log = open("outputs/crowd_log.csv", "w", newline="")
csv_w = csv.writer(log)
csv_w.writerow(["frame", "time_s", "people", "max_zone_count", "max_level"])

frame_no, counts, boxes, pts = 0, None, [], []
while cap.isOpened():
    ok, frame = cap.read()
    if not ok or (a.max_frames and frame_no >= a.max_frames):
        break

    if frame_no % a.skip == 0:
        boxes = det.detect(frame)
        pts = foot_points(boxes)
        counts = zone_counts(pts, frame.shape)

    out = draw_zones(draw_heatmap(frame, pts), counts)
    top_level = level_of(counts.max())
    cv2.putText(out, f"People: {len(boxes)}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    if top_level[1] in ("High", "Critical"):
        r, c = divmod(int(counts.argmax()), counts.shape[1])
        cv2.rectangle(out, (0, 0), (w, 50), top_level[2], -1)
        cv2.putText(out, f"ALERT: Zone ({r+1},{c+1}) {top_level[1].upper()}  [{counts.max()} people]",
                    (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    writer.write(out)
    csv_w.writerow([frame_no, round(frame_no / fps, 2), len(boxes),
                    int(counts.max()), top_level[1]])
    frame_no += 1
    if frame_no % 30 == 0:
        print("frames processed:", frame_no)

cap.release()
writer.release()
log.close()
print("Saved", a.out, "and outputs/crowd_log.csv")
