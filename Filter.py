import streamlit as st
import cv2
import time

# Titel
st.title("🎯 Camera met Objectfilter")

# Stap 1: Vraag gebruiker naar objecten die gedetecteerd mogen worden
user_input = st.text_input("Welke objecten wil je detecteren? (bijv: car, person, dog):")

# Verwerk de input
allowed_classes = [cls.strip().lower() for cls in user_input.split(",") if cls.strip()]

# Stap 2: Camera activeren
if st.button("📷 Maak foto"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Kon geen beeld maken met de camera.")
    else:
        st.image(frame, caption="📸 Gemaakte foto", channels="BGR")

        st.info("Wachten op detectie...")


        class DummyBox:
            def __init__(self, cls, conf):
                self.cls = [cls]
                self.conf = [conf]

        class DummyResults:
            def __init__(self):
                self.boxes = [DummyBox(0, 0.88), DummyBox(2, 0.75)]  # Voorbeeld: car en dog

        dummy_results = DummyResults()
        dummy_model_names = {0: "car", 1: "person", 2: "dog"}

        # --- Jouw filterfunctie ---
        def apply_filter(results, allowed_classes, model_names):
            filtered = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                class_name = model_names[class_id].lower()
                if class_name in allowed_classes:
                    confidence = float(box.conf[0])
                    filtered.append((class_name, confidence))
            return filtered

      
        filtered_objects = apply_filter(dummy_results, allowed_classes, dummy_model_names)

        # Toon resultaat
        if filtered_objects:
            st.success("Gedetecteerde objecten:")
            for name, conf in filtered_objects:
                st.write(f"✅ {name} ({conf:.2%} vertrouwen)")
        else:
            st.warning("Er zijn geen opgegeven objecten gevonden in de foto.")
