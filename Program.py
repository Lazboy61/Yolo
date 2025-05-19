from ultralytics import *
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

image_folder = "D:\CODE\PROJECTD\yolo_alphabet_dataset\images\\train"  
model_path = "D:\CODE\PROJECTD\PROJECTD\yolov8n.pt"
model = YOLO("yolov8n.pt")  

model.train(
    data="D:\CODE\PROJECTD\yolo_alphabet_dataset\data.yaml", 
    epochs=1,
    imgsz=512,
    batch=16,
    project="alphabet_gesture_project",
    name="yolov8_alphabet_model",
    workers=2,
    verbose=True
)

results = model.val()
 

# print("this is the number of objects observed")
# n = len(results[0].boxes)
# print(n)
 
 
# for i in range(n):
#     id = int(results[0].boxes.cls[i])
#     name = (results[0].names[id])
#     confidence = float(results[0].boxes.conf[i])
#     print(name)
#     print(round(confidence, 2))
 
 
 
# a = int(results[0].boxes.cls[0])
# print(results[0].names[a])
 

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith((".jpg", ".png"))]
random_image = random.choice(image_files)
image_path = os.path.join(image_folder, random_image)

results = model(image_path)
img = Image.open(image_path)
plt.imshow(img)
plt.axis("off")


for result in results:
    if len(result.boxes.cls) > 0:
        predicted_class_id = int(result.boxes.cls[0].item())
        predicted_letter = model.names[predicted_class_id]
        print(f" Letter: {predicted_letter}")
        plt.title(f"Letter: {predicted_letter}")
    else:
        print("No letter detected")
        plt.title("No letter detected")

plt.show()