# import os
# import xml.etree.ElementTree as ET

# # Config - UPDATE THESE PATHS!
# XML_DIR = "/Users/hacakir/Documents/Project D/Yolo/xml_labels/train"
# YOLO_DIR = "/Users/hacakir/Documents/Project D/Yolo/dataset/labels/train"
# CLASSES = {'i': 0, 'k': 2}  # Must match dataset.yaml

# # Create output directory
# os.makedirs(YOLO_DIR, exist_ok=True)

# print(f"🔍 Scanning {XML_DIR}...")
# xml_files = [f for f in os.listdir(XML_DIR) if f.endswith('.xml')]
# print(f"Found {len(xml_files)} XML files to convert")

# for xml_file in xml_files:
#     try:
#         # Parse XML
#         tree = ET.parse(os.path.join(XML_DIR, xml_file))
#         root = tree.getroot()
        
#         # Get dimensions
#         size = root.find('size')
#         width = int(size.find('width').text)
#         height = int(size.find('height').text)
        
#         # Prepare YOLO file
#         txt_file = os.path.join(YOLO_DIR, xml_file.replace('.xml', '.txt'))
#         print(f"📄 Converting {xml_file} → {txt_file}")
        
#         with open(txt_file, 'w') as f:
#             for obj in root.findall('object'):
#                 class_name = obj.find('name').text
#                 if class_name not in CLASSES:
#                     print(f"⚠️ Unknown class '{class_name}' in {xml_file}")
#                     continue
                
#                 # Convert bbox
#                 bndbox = obj.find('bndbox')
#                 xmin = int(bndbox.find('xmin').text)
#                 ymin = int(bndbox.find('ymin').text)
#                 xmax = int(bndbox.find('xmax').text)
#                 ymax = int(bndbox.find('ymax').text)
                
#                 # YOLO format
#                 x_center = ((xmin + xmax) / 2) / width
#                 y_center = ((ymin + ymax) / 2) / height
#                 box_width = (xmax - xmin) / width
#                 box_height = (ymax - ymin) / height
                
#                 f.write(f"{CLASSES[class_name]} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")
        
#         print(f"✅ Successfully created {txt_file}")
    
#     except Exception as e:
#         print(f"❌ Failed to process {xml_file}: {str(e)}")

# print(f"\n🎉 Done! Converted {len(xml_files)} files")

