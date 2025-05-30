import os
import xml.etree.ElementTree as ET

# Zet deze mappen naar jouw structuur
xml_folder = "D:/Users/Gebruiker/Documents/Project D/Yolo/images/train_aug"
output_folder = "D:/Users/Gebruiker/Documents/Project D/Yolo/labels/train_aug"

# Mapping van class-namen naar nummers (YOLO-format)
class_mapping = {
    "i": 0,
    "k": 1
}

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for xml_file in os.listdir(xml_folder):
    if not xml_file.endswith(".xml"):
        continue

    tree = ET.parse(os.path.join(xml_folder, xml_file))
    root = tree.getroot()

    image_width = int(root.find("size/width").text)
    image_height = int(root.find("size/height").text)

    yolo_lines = []

    for obj in root.findall("object"):
        class_name = obj.find("name").text.lower()
        if class_name not in class_mapping:
            continue

        class_id = class_mapping[class_name]
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # YOLO formaat: class x_center y_center width height (allemaal genormaliseerd)
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        width = (xmax - xmin) / image_width
        height = (ymax - ymin) / image_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # Sla het .txt-bestand op met dezelfde naam
    txt_filename = os.path.splitext(xml_file)[0] + ".txt"
    with open(os.path.join(output_folder, txt_filename), "w") as f:
        f.write("\n".join(yolo_lines))
