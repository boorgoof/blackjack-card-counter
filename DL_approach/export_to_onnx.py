from ultralytics import YOLO

# Percorso del modello YOLO .pt
model_path = 'yolo11s_synthetic_1280.pt'  # Sostituisci con il percorso corretto

# Carica il modello
model = YOLO(model_path)

# Esporta in ONNX con una dimensione di input di 1280x1280
model.export(format="onnx", opset=12, imgsz=1280)

print(f"Modello esportato con successo in ONNX: {model_path.replace('.pt', '.onnx')}")