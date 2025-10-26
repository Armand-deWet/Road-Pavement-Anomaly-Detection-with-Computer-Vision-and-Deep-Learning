from ultralytics import YOLO
import os
import glob

# --- SETTINGS ---
MODEL_PATH = "/home/armand/ext_data/yolov12/runs/detect/yolo12x_training_full_2/weights/best.pt"  # Your trained YOLOv12 model
IMAGE_DIR = "/home/armand/ext_data/data_upload/images/train"  # Folder with unlabeled images
OUTPUT_DIR = "/home/armand/ext_data/yolov12_semi_supervised_learning"    # Where .txt labels will be saved
CONF_THRESHOLD = 0.294  # Only keep predictions above this confidence

# Make sure output folder exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- LOAD MODEL ---
model = YOLO(MODEL_PATH)

# --- RUN INFERENCE ---
# save_txt=True automatically creates YOLO format .txt label files
results = model.predict(
    source=IMAGE_DIR,
    save_txt=True,
    save_conf=True,  # Adds confidence as last column
    conf=CONF_THRESHOLD,  # Filter predictions directly
    project=OUTPUT_DIR,
    name="",  # Avoid extra subfolder nesting
    exist_ok=True
)

print(f"✅ Labels saved to: {OUTPUT_DIR}/labels")
