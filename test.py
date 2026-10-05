from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")
result = model("crowd.png", classes=[0], conf=0.25)[0]

print("People detected:", len(result.boxes))
cv2.imwrite("output.jpg", result.plot())