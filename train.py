# from ultralytics import YOLO

# model = YOLO('yolov8n.pt')  # Load pretrained model
# model.train(
#     data='dataset.yaml',
#     epochs=50,
#     imgsz=320,
#     batch=8,
#     device='mps'  # Use Apple Metal
# )
from ultralytics import YOLO

# Laad het getrainde model
model = YOLO('runs/detect/train8/weights/best.pt')

# Test op een losse afbeelding (vervang het pad met jouw bestand)
results = model('test.jpg', save=True)

# Toon resultaat (optioneel, alleen als je GUI hebt)
results[0].show()
