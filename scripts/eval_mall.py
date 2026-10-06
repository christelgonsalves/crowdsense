import argparse
import csv
import cv2
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from crowdsense.detector import PersonDetector

p = argparse.ArgumentParser()
p.add_argument("--root", default=r"C:\Projects\data\mall_dataset")
p.add_argument("--weights", default="yolov8m.pt")
p.add_argument("--mode", default="standard", choices=["standard", "tiled"])
p.add_argument("--imgsz", type=int, default=1280)
p.add_argument("--conf", type=float, default=0.35)
p.add_argument("--step", type=int, default=10, help="use every Nth frame (10 = 200 frames)")
p.add_argument("--tag", default="run")
a = p.parse_args()

gt = loadmat(str(Path(a.root, "mall_gt.mat")))["count"].flatten()
frames = sorted(Path(a.root, "frames").glob("*.jpg"))
print("Frames found:", len(frames), "| GT counts:", len(gt))

det = PersonDetector(weights=a.weights, imgsz=a.imgsz, conf=a.conf, mode=a.mode)

rows, errs = [], []
for i in range(0, min(len(frames), len(gt)), a.step):
    pred = len(det.detect(cv2.imread(str(frames[i]))))
    errs.append(pred - int(gt[i]))
    rows.append([frames[i].name, int(gt[i]), pred, pred - int(gt[i])])
    if len(rows) % 20 == 0:
        print(f"{len(rows)} frames done")

e = np.array(errs)
mae, rmse, bias = np.abs(e).mean(), np.sqrt((e ** 2).mean()), e.mean()
print(f"\n[{a.tag}] frames={len(e)} mode={a.mode} imgsz={a.imgsz} conf={a.conf}")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"Bias: {bias:+.2f}  (negative = under-counting)")

with open(f"outputs/eval_mall_{a.tag}.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["frame", "gt", "pred", "error"])
    w.writerows(rows)
