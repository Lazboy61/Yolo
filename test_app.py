import pytest
from PIL import Image
from io import BytesIO
import numpy as np
from app import process_image, convert_to_format, load_model
from unittest.mock import MagicMock, patch
import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([DetectionModel])

@pytest.fixture
def dummy_image():
    return Image.fromarray(np.ones((300, 300, 3), dtype=np.uint8) * 255)

def test_load_model():
    model = load_model("yolov8n.pt")
    assert model is not None

def test_process_image(dummy_image):
    model = load_model("yolov8n.pt")
    results, processed = process_image(dummy_image, model, 0.25)
    assert results is not None
    assert processed is not None

def test_convert_to_format_csv():
    dummy_data = [{"label": "a", "confidence": 0.9, "timestamp": "2025-06-23 14:00:00"}]
    csv_data = convert_to_format(dummy_data, "CSV")
    assert "label" in csv_data

def test_convert_to_format_json():
    dummy_data = [{"label": "c", "confidence": 0.99, "timestamp": "2025-06-23 14:02:00"}]
    json_data = convert_to_format(dummy_data, "JSON")
    assert '"label": "c"' in json_data

def test_process_image_no_detections(dummy_image):
    model = load_model("yolov8n.pt")
    results, processed = process_image(dummy_image, model, 0.99)  # Hoog threshold voor geen detecties
    assert results[0].boxes.shape[0] == 0  # Geen detecties verwacht
    assert processed is not None

def test_convert_to_format_txt_with_funny_comments():
    dummy_data = [{"label": "a", "confidence": 0.9, "timestamp": "2025-06-23 14:00:00"}] * 6
    txt_data = convert_to_format(dummy_data, "TXT")
    assert "🤖 Model says:" in txt_data  # Controleer of grappige comments zijn toegevoegd

def test_show_camera_section():
    with patch('cv2.VideoCapture') as mock_capture:
        mock_capture.return_value.read.return_value = (
    True, 
    np.zeros((480, 640, 3), dtype=np.uint8)
)