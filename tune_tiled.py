import cv2
from crowdsense.detector import PersonDetector

img = cv2.imread("crowd.png")

for conf in (0.25, 0.35, 0.45):
    for overlap in (0.1, 0.2):
        det = PersonDetector(weights="yolov8m.pt", conf=conf, mode="tiled",
                             tile=640, overlap=overlap)
        n = len(det.detect(img))
        print(f"conf={conf:.2f} overlap={overlap:.1f} -> {n} people")
