from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
model = YOLO("") # model path

# Export the model to NCNN format
model.export(format="ncnn", imgsz=640)  # creates 'ncnn_model'