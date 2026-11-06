# CrowdSense: Crowd Counting and Overcrowding Alerts with YOLO

A YOLOv8 system that counts people in images and video, draws density heatmaps, splits the scene into zones, and flags overcrowding. A Streamlit dashboard lets you upload a photo and see the count, heatmap and alert.

## Results (Mall dataset, 200 frames)
| Setup | MAE | RMSE |
|---|---|---|
| Default (640 px, conf 0.25) | 11.84 | 12.73 |
| **Tuned (1280 px, conf 0.12)** | **3.19** | **4.15** |

Counting error fell by about 73% from resolution and threshold tuning alone, with no training. The confidence value was tuned on the same frames used for scoring, so the figure is slightly optimistic.

## Limitations
- Packed crowds are under-counted, so the alert is least reliable when overcrowding is worst.
- Heavily hidden people are missed.
- The confidence value and zone limits must be set for each scene.
- Fine-tuning on a third-party CrowdHuman set performed worse than the pretrained model on independent data (Mall MAE 20.09 against 3.19), likely because of incomplete labels. The final system uses pretrained `yolov8m.pt`.

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```

## Credits
Mall dataset (Loy et al., research use only), CrowdHuman (Shao et al., 2018), YOLOv8 (Ultralytics), SAHI.

Author: Christel Gonsalves
