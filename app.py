import streamlit as st
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from PIL import Image
import cv2

# =========================================================
# CONFIG
# =========================================================
CLASS_NAMES = ["Bird", "Drone"]

st.set_page_config(page_title="Aerial Object Classification & Detection", layout="wide")

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("models/cnn_model.keras")

@st.cache_resource
def load_resnet_model():
    return tf.keras.models.load_model("models/resnet_model.keras")

@st.cache_resource
def load_yolo_model():
    return YOLO("models/yolov8n.pt")

cnn_model = load_cnn_model()
resnet_model = load_resnet_model()
yolo_model = load_yolo_model()

# =========================================================
# PREPROCESS FUNCTION
# =========================================================
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=0)

# =========================================================
# STREAMLIT UI
# =========================================================
st.title("🛩️ Aerial Object Classification & Detection")
st.markdown("Upload an image to classify (CNN/ResNet) or detect (YOLO).")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_column_width=True)

    tab1, tab2, tab3 = st.tabs(["🔍 CNN Prediction", "🔍 ResNet Prediction", "🎯 YOLO Detection"])

    # =========================================================
    # CNN PREDICTION
    # =========================================================
    with tab1:
        st.subheader("CNN Classification Result")

        processed = preprocess_image(img)
        preds = cnn_model.predict(processed)
        class_idx = np.argmax(preds)
        confidence = preds[0][class_idx]

        st.write(f"**Prediction:** {CLASS_NAMES[class_idx]}")
        st.write(f"**Confidence:** {confidence:.4f}")

    # =========================================================
    # RESNET PREDICTION
    # =========================================================
    with tab2:
        st.subheader("ResNet Classification Result")

        processed = preprocess_image(img)
        preds = resnet_model.predict(processed)
        class_idx = np.argmax(preds)
        confidence = preds[0][class_idx]

        st.write(f"**Prediction:** {CLASS_NAMES[class_idx]}")
        st.write(f"**Confidence:** {confidence:.4f}")

    # =========================================================
    # YOLO DETECTION
    # =========================================================
    with tab3:
        st.subheader("YOLO Object Detection")

        # Convert PIL → OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Run YOLO inference
        results = yolo_model(img_cv)

        # Draw bounding boxes
        annotated = results[0].plot()  # returns NumPy image with boxes

        st.image(annotated, caption="YOLO Detection Output", use_column_width=True)

else:
    st.info("Please upload an image to start.")
