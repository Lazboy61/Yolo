import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time

st.set_page_config(page_title="Camera Activatie", layout="centered")
st.title("Camera Activatie App")

start_camera = st.button("Start mijn camera")

if start_camera:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()  

    if not cap.isOpened():
        st.error("Kon de camera niet openen.")
    else:
        st.success("Camera gestart! Klik op de 'Stop' knop of sluit het tabblad om te stoppen.")

        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("Kan frame niet lezen van de camera.")
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Van BGR naar RGB
            stframe.image(frame, channels="RGB")  

            
            time.sleep(0.03) 

cap.release()
