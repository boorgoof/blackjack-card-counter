from ultralytics import YOLO

# Percorso del modello YOLO .pt
model_path = 'yolov11s_single_cards_1280.pt'  # Sostituisci con il percorso corretto

# Carica il modello
model = YOLO(model_path)

# Esporta in ONNX con una dimensione di input di 1280x1280
model.export(format="onnx", opset=12, imgsz=1280)

print(f"Model exported successfully in ONNX: {model_path.replace('.pt', '.onnx')}")