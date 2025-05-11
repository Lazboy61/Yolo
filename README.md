# 🧠 YOLO Object Detection App

Deze applicatie maakt gebruik van **YOLO (You Only Look Once)** om objecten te herkennen in afbeeldingen of camerabeelden.

## 📸 Functionaliteit

- Live camerabeeld openen via **Streamlit**
- Mogelijkheid om één foto te maken
- (In ontwikkeling) Objectherkenning via YOLO-model
- Optie om handmatig een afbeelding te uploaden

## ▶️ Installatie & Setup

1. **Clone de repository:**

````bash
git clone https://github.com/Lazboy61/Yolo
cd yolo
Activeer je virtuele omgeving (optioneel maar aanbevolen):
python -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
pip install -r requirements.txt


## Hardware Notes
- Apple Silicon: Requires PyTorch nightly (`pip install --pre torch`)
- NVIDIA GPU: Install CUDA-enabled PyTorch
- CPU-only: May be slower
## Download Pretrained Model
Place custom model weights in:runs/detect/train/weights/best.pt

### Windows Webcam Alternative
Use OpenCV directly:
```python
import cv2
cap = cv2.VideoCapture(0)
````
