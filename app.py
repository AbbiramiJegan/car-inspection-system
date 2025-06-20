import streamlit as st
from ultralytics import YOLO
import tempfile
import cv2
import os
import time

# App configuration
st.set_page_config(page_title="🚗 Car Inspection System", layout="wide")

# Title and description
st.markdown("<h1 style='text-align: center;'>🚗 Car Inspection System</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: gray;'>Upload a video to automatically detect car door components using a custom-trained YOLO World model</h4>", unsafe_allow_html=True)
st.markdown("---")

# Sidebar: About and instructions
with st.sidebar:
    st.header("📋 How to Use")
    st.write("1. Upload a video file (`.mp4`, `.mov`, `.avi`, `.mkv`).")
    st.write("2. The system will process the video in real time.")
    st.write("3. Watch the live detection results frame-by-frame.")
    st.write("4. No need to wait for the whole video to finish!")

    st.markdown("---")
    st.info("💡 Tip: Shorter videos process faster!")
    st.markdown("---")
    st.caption("Developed by SHRDC")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Load your custom YOLO-World model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("📤 Upload a video for inspection", type=["mp4", "mov", "avi", "mkv"])

# Process video if uploaded
if uploaded_file is not None:
    st.video(uploaded_file)  # Show the raw uploaded video
    st.markdown("### 🔍 Processing... Please wait.")

    # Save to temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    # Define line X-coordinate for detection
    LINE_X = 600  

    with st.spinner("Running detection on each frame..."):
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            # YOLO Inference
            results = model.predict(frame, conf=0.3, verbose=False)
            annotated_frame = results[0].plot()

            # Draw boundary zone (vertical red line)
            ZONE_WIDTH = 5
            cv2.rectangle(
                annotated_frame,
                (LINE_X - ZONE_WIDTH // 2, 0),
                (LINE_X + ZONE_WIDTH // 2, frame.shape[0]),
                color=(0, 0, 255),
                thickness=-1
            )

            # Show live detection frame
            stframe.image(annotated_frame, channels="BGR", use_container_width=True)

            time.sleep(0.03)

        cap.release()

    st.success("✅ Video processing complete!")