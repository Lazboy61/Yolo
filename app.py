import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")  # or "yolov5s.pt" if you prefer

# Filter function
def apply_filter(results, allowed_classes, model_names):
    filtered = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model_names[class_id].lower()
        if class_name in allowed_classes:
            confidence = float(box.conf[0])
            filtered.append((class_name, confidence))
    return filtered

st.set_page_config(page_title="YOLO Object Detection", layout="centered")
st.title("📸 YOLO Object Detection Demo")

# Filter input
filter_input = st.text_input("Which objects to detect? (e.g., car,person,dog):")
allowed_classes = [cls.strip().lower() for cls in filter_input.split(",")] if filter_input else []

# Input method
input_option = st.radio("Choose input method:", ["Camera", "Upload Image"])

# --- Camera Mode ---
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

                    # Run YOLO detection
                    results = model(frame, verbose=False)
                    
                    # Draw bounding boxes on the frame
                    annotated_frame = results[0].plot()  # This adds boxes + labels
                    
                    # Display the annotated frame
                    stframe.image(annotated_frame, channels="BGR")

                    # Show filtered results
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

# --- Image Upload Mode ---
elif input_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Run YOLO detection
        results = model(image_np, verbose=False)
        
        # Draw bounding boxes
        annotated_image = results[0].plot()  # Adds boxes + labels
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)  # Fix color for PIL
        
        # Display original and annotated image side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(annotated_image, caption="Detected Objects", use_container_width=True)

        # Show filtered results
        filtered = apply_filter(results[0], allowed_classes, model.names)
        st.subheader("Results")
        if filtered:
            for name, conf in filtered:
                st.success(f"✅ {name} ({conf:.2%})")
        else:
            st.warning("No specified objects found.")
##streamlit run app.py