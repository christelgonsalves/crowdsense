import cv2

src = cv2.VideoCapture("crowd.mp4")
fps = src.get(5)
W, H = 1280, 720
out = cv2.VideoWriter("crowd_720.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

n = 0
while True:
    ok, frame = src.read()
    if not ok:
        break
    out.write(cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA))
    n += 1
src.release()
out.release()
print("Done. Frames written:", n)
