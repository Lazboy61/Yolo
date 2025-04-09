def get_user_filters():
    allowed = input("Welke objecten wil je detecteren? (bijv: car,person,dog): ")
    return [cls.strip().lower() for cls in allowed.split(",")]

def apply_filter(results, allowed_classes, model_names):
    filtered = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model_names[class_id].lower()
        if class_name in allowed_classes:
            confidence = float(box.conf[0])
            filtered.append((class_name, confidence))
    return filtered
