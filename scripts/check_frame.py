import cv2
from crowdsense.detector import PersonDetector
img = cv2.imread(r"C:\Projects\data\mall_dataset\frames\seq_000500.jpg")
boxes = PersonDetector(imgsz=1280, conf=0.25).detect(img)
for x1, y1, x2, y2 in boxes.astype(int):
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
cv2.putText(img, f"Detected: {len(boxes)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imwrite("outputs/mall_check.jpg", img)
print("Detected:", len(boxes))
