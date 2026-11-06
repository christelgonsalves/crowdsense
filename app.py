import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import crowdsense.analysis as an
from crowdsense.detector import PersonDetector, foot_points

st.set_page_config(page_title="CrowdSense", layout="wide")
st.title("CrowdSense: crowd counting and overcrowding alerts")

st.sidebar.header("Settings")
mode = st.sidebar.radio("Detection mode", ["standard", "tiled"])
imgsz = st.sidebar.selectbox("Image size (standard mode)", [640, 960, 1280, 1920], index=2)
conf = st.sidebar.slider("Confidence threshold", 0.05, 0.60, 0.25, 0.01)
rows = st.sidebar.slider("Zone rows", 1, 6, 3)
cols = st.sidebar.slider("Zone columns", 1, 8, 4)
st.sidebar.subheader("People per zone")
med = st.sidebar.number_input("Medium from", 1, 100, 5)
high = st.sidebar.number_input("High from", 1, 100, 9)
crit = st.sidebar.number_input("Critical from", 1, 100, 13)
an.LEVELS = [(0, "Low", (0, 200, 0)), (med, "Medium", (0, 215, 255)),
             (high, "High", (0, 128, 255)), (crit, "Critical", (0, 0, 255))]


@st.cache_resource
def load(mode, imgsz, conf):
    return PersonDetector(weights="yolov8m.pt", imgsz=imgsz, conf=conf, mode=mode)


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def analyse(img):
    boxes = load(mode, imgsz, conf).detect(img)
    pts = foot_points(boxes)
    counts = an.zone_counts(pts, img.shape, rows, cols)
    top = an.level_of(int(counts.max()))
    return boxes, pts, counts, top


def is_over(top):
    return top[1] in ("High", "Critical")


tab_img, tab_vid = st.tabs(["Image", "Video"])

with tab_img:
    up = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"])
    if up is not None:
        img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
        with st.spinner("Detecting people..."):
            boxes, pts, counts, top = analyse(img)
        boxed = img.copy()
        for x1, y1, x2, y2 in boxes.astype(int):
            cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)
        c1, c2, c3 = st.columns(3)
        c1.metric("People detected", len(boxes))
        c2.metric("Busiest zone", int(counts.max()))
        c3.metric("Highest level", top[1])
        if is_over(top):
            st.error("OVERCROWDED")
        else:
            st.success("NOT OVERCROWDED")
        t1, t2, t3 = st.tabs(["Detections", "Density heatmap", "Zone levels"])
        with t1:
            st.image(rgb(boxed), use_container_width=True)
        with t2:
            st.image(rgb(an.draw_heatmap(img, pts)), use_container_width=True)
        with t3:
            st.image(rgb(an.draw_zones(img, counts)), use_container_width=True)
            st.dataframe(counts)
    else:
        st.info("Upload an image to start. Limits are set per scene in the sidebar.")

with tab_vid:
    st.caption("Use standard mode for video, because tiled mode is very slow on a CPU.")
    vid = st.file_uploader("Upload a crowd video", type=["mp4", "avi", "mov", "mkv"])
    skip = st.slider("Detect every Nth frame (higher = faster)", 1, 15, 5)
    max_sec = st.slider("Analyse the first N seconds", 5, 60, 20)
    if vid is not None and st.button("Analyse video"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(vid.read())
        tmp.close()
        cap = cv2.VideoCapture(tmp.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        limit = min(int(max_sec * fps), total)
        status = st.empty()
        frame_box = st.empty()
        bar = st.progress(0.0)
        history, last, n = [], None, 0
        while n < limit:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > 960:
                frame = cv2.resize(frame, (960, int(h * 960 / w)))
            if n % skip == 0:
                last = analyse(frame)
                boxes, pts, counts, top = last
                out = an.draw_zones(an.draw_heatmap(frame, pts), counts)
                over = is_over(top)
                if over:
                    cv2.rectangle(out, (0, 0), (out.shape[1], 50), (0, 0, 255), -1)
                    cv2.putText(out, "OVERCROWDED", (15, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
                sec = round(n / fps, 1)
                history.append((sec, len(boxes), over))
                frame_box.image(rgb(out), use_container_width=True)
                label = "OVERCROWDED" if over else "Not overcrowded"
                status.markdown(f"**Time {sec}s | People: {len(boxes)} | {label}**")
            n += 1
            bar.progress(min(n / limit, 1.0))
        cap.release()
        try:
            os.remove(tmp.name)
        except OSError:
            pass
        if not history:
            st.error("Could not read this video. Try an mp4 file.")
        else:
            df = pd.DataFrame(history, columns=["second", "people", "overcrowded"])
            m1, m2, m3 = st.columns(3)
            m1.metric("Frames analysed", len(df))
            m2.metric("Peak people", int(df["people"].max()))
            m3.metric("Time overcrowded", f"{100 * df['overcrowded'].mean():.0f}%")
            st.subheader("People detected over time")
            st.line_chart(df.set_index("second")["people"])
            if df["overcrowded"].any():
                st.error("Overcrowding occurred during this video.")
            else:
                st.success("No overcrowding in this video.")
