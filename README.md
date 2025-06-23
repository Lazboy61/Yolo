YOLO Object Detection Application
Overview

This application provides real-time object detection using YOLO (You Only Look Once) through a Streamlit interface. It supports both live camera input and image uploads for detection.

Features

Live object detection from webcam feed
Image upload functionality
Custom YOLO model integration
Detection history logging
Installation

Prerequisites

Python 3.8-3.10
pip package manager
Setup

Clone the repository:
git clone https://github.com/Lazboy61/Yolo
cd Yolo

Create and activate virtual environment (Reccomended):
bash
python -m venv venv
source venv/bin/activate # macOS/Linux
venv\Scripts\activate # Windows

Install dependencies:
pip install -r requirements.txt

Usage

Run the application:
streamlit run app.py
