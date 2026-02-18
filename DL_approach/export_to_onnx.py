from ultralytics import YOLO

model_path = 'yolov11s_single_cards_1280.pt'

model = YOLO(model_path)

model.export(format="onnx", opset=12, imgsz=1280)

print(f"Model exported successfully in ONNX: {model_path.replace('.pt', '.onnx')}")