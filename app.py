import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import tempfile
import os



st.set_page_config(page_title="Multi Object Detection", layout="centered")
st.title("🚗 Multi Object Detection on Road Scenes")
st.markdown("Upload an image or video to detect multiple road objects using YOLOv5.")

@st.cache_resource
def load_model():
    model = torch.hub.load('yolov5', 'yolov5s', source='local', force_reload=True)
    return model

model = load_model()


# Image Upload
uploaded_image = st.file_uploader("📷 Upload Image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy for model
    image_np = np.array(image)
    results = model(image_np)
    results.render()

    st.image(results.ims[0], caption="Detected Objects", use_column_width=True)

# Video Upload
uploaded_video = st.file_uploader("🎥 Upload Video", type=['mp4', 'avi', 'mov'])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    output_path = os.path.join("output", "result.mp4")
    os.makedirs("output", exist_ok=True)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    stframe = st.empty()
    with st.spinner("Processing video..."):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Inference
            results = model(frame)
            results.render()
            frame = results.ims[0]

            # Show frame in Streamlit
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, channels="RGB", use_column_width=True)

            # Write to output video
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        cap.release()
        out.release()

    st.success("Video processing complete.")
    st.video(output_path)
