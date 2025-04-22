import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Camera of Afbeelding", layout="centered")
st.title("YOLO Demo: Kies je invoermethode")

# Filter input
filter_input = st.text_input("Welke objecten wil je detecteren? (bijv: car,person,dog):")
allowed_classes = [cls.strip().lower() for cls in filter_input.split(",")] if filter_input else []

# Invoermethode kiezen
invoer_optie = st.radio("Kies een invoermethode:", ["Activeer Camera", "Upload een Foto"])

if invoer_optie == "Activeer Camera":
    start_camera = st.button("Start mijn camera")

    if start_camera:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()

        if not cap.isOpened():
            st.error("Kon de camera niet openen.")
        else:
            st.success("Camera gestart! Sluit het venster om te stoppen.")

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Kan frame niet lezen van de camera.")
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    stframe.image(frame, channels="RGB")
                    time.sleep(0.03)
            except:
                st.info("Camera gestopt.")
            finally:
                cap.release()

elif invoer_optie == "Upload een Foto":
    uploaded_file = st.file_uploader("Upload een afbeelding", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Geüploade afbeelding", use_column_width=True)

        # Hier zou je team's YOLO later de afbeelding kunnen laten analyseren
        st.info("Zodra YOLO klaar is, kunnen we hier objectdetectie uitvoeren op deze afbeelding.")


##streamlit run app.py