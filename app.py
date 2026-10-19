import cv2
import numpy as np
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


def show(img_bgr, caption):
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


up = st.file_uploader("Upload a crowd image", type=["jpg", "jpeg", "png"])
if up is not None:
    img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    with st.spinner("Detecting people..."):
        boxes = load(mode, imgsz, conf).detect(img)
    pts = foot_points(boxes)
    counts = an.zone_counts(pts, img.shape, rows, cols)
    top = an.level_of(int(counts.max()))

    boxed = img.copy()
    for x1, y1, x2, y2 in boxes.astype(int):
        cv2.rectangle(boxed, (x1, y1), (x2, y2), (0, 255, 0), 2)

    c1, c2, c3 = st.columns(3)
    c1.metric("People detected", len(boxes))
    c2.metric("Busiest zone", int(counts.max()))
    c3.metric("Highest level", top[1])
    if top[1] in ("High", "Critical"):
        r, c = divmod(int(counts.argmax()), counts.shape[1])
        st.error(f"ALERT: zone ({r + 1},{c + 1}) is {top[1].upper()} with {int(counts.max())} people")
    else:
        st.success("No overcrowding alert")

    t1, t2, t3 = st.tabs(["Detections", "Density heatmap", "Zone levels"])
    with t1:
        show(boxed, "Detected people")
    with t2:
        show(an.draw_heatmap(img, pts), "Density heatmap")
    with t3:
        show(an.draw_zones(img, counts), "Crowd level per zone")
        st.write("People per zone (rows x columns):")
        st.dataframe(counts)
else:
    st.info("Upload an image to start. Thresholds are set per scene in the sidebar.")
