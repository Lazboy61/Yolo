import os
import cv2
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO
import streamlit as st

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="YOLO Object Detection", layout="centered")
st.title("YOLO Object Detection Demo")

# --- 2. MODEL LOADING ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "runs", "detect", "train27", "weights", "best.pt")


# Model selection
model_choice = st.sidebar.radio(
    "Select Model:",
    ("Pretrained YOLOv8n", "Custom Trained Model"),
    index=1  # Default to custom
)

if model_choice == "Pretrained YOLOv8n":
    model = YOLO('yolov8n.pt')
    st.sidebar.warning("Using general-purpose YOLOv8n (lower accuracy for gestures)")
else:
    model = YOLO(model_path) if os.path.exists(model_path) else YOLO('yolov8n.pt')
    st.sidebar.success("Using your custom gesture model")

# --- 3. HELPER FUNCTIONS ---
def apply_filter(results, allowed_classes, model_names):
    filtered = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model_names[class_id].lower()
        if class_name in allowed_classes:
            confidence = float(box.conf[0])
            filtered.append((class_name, confidence))
    return filtered

# --- 4. MAIN APP ---
filter_input = st.text_input("Which objects to detect? (e.g., 1,2):")
allowed_classes = [cls.strip().lower() for cls in filter_input.split(",")] if filter_input else []

input_option = st.radio("Choose input method:", ["Camera", "Upload Image"])

if input_option == "Camera":
    start_camera = st.button("Start Camera")
    
    if start_camera:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        
        if not cap.isOpened():
            st.error("Failed to open camera.")
        else:
            st.success("Camera started! Press 'q' to stop.")
            
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to capture frame.")
                        break

                    results = model(frame, verbose=False)
                    annotated_frame = results[0].plot()
                    stframe.image(annotated_frame, channels="BGR")

                    filtered = apply_filter(results[0], allowed_classes, model.names)
                    if filtered:
                        st.subheader("Detected:")
                        for name, conf in filtered:
                            st.write(f"✅ {name} ({conf:.2%})")
                    time.sleep(0.1)
                    
            except Exception as e:
                st.error(f"Error: {e}")
            finally:
                cap.release()

elif input_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", 
                                   type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=False)
    
    if uploaded_file is not None:
        if uploaded_file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            results = model(image_np, verbose=False)
            annotated_image = results[0].plot()
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            with col2:
                st.image(annotated_image, caption="Detected Objects", use_container_width=True)

            filtered = apply_filter(results[0], allowed_classes, model.names)
            st.subheader("Results")
            if filtered:
                for name, conf in filtered:
                    st.success(f"✅ {name} ({conf:.2%})")
            else:
                st.warning("No specified objects found.")
        else:
            st.error("Invalid file type. Please upload JPG/PNG.")

##streamlit run app.py