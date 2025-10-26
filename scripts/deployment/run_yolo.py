import cv2
from ultralytics import YOLO

# ---- CONFIG ----
input_video_path = "input.mp4"      # Path to your input video
output_video_path = "output_annotated.mp4"  # Path to save the annotated video
model_path = "yolov12n_ncnn_model"  # Path to your converted NCNN model folder

# ---- LOAD MODEL ----
model = YOLO(model_path)

# ---- OPEN VIDEO ----
cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {input_video_path}")

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---- OUTPUT VIDEO WRITER ----
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- INFERENCE ----
    results = model(frame)
    annotated_frame = results[0].plot()

    # ---- OPTIONAL: ADD FPS INFO ----
    inference_time = results[0].speed['inference']
    fps_text = f"FPS: {1000 / inference_time:.1f}" if inference_time > 0 else ""
    cv2.putText(
        annotated_frame,
        fps_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # ---- WRITE FRAME ----
    out.write(annotated_frame)
    frame_count += 1

# ---- CLEANUP ----
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ Done! Annotated video saved to: {output_video_path}")
