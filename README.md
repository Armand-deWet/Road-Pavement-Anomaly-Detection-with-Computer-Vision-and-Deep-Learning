# Road Pavement Anomaly Detection with Computer Vision and Deep Learning

This repository contains the source code, trained models, and experimental framework accompanying the mini-dissertation:

**Road Pavement Anomaly Detection with Computer Vision and Deep Learning: Experimental Details and Practical Application**  
by **Armand de Wet**, University of Pretoria, 2025.

---

## 📘 Abstract

The continuous and efficient upkeep of roads is critical to societal and economic activity. Current road damage detection methods are manual, time-consuming, and costly, motivating research into automated techniques. This project investigates the latest YOLO architectures (v10–v12) for automated road pavement anomaly detection, emphasizing lightweight deployment on embedded hardware. Models were trained and evaluated on the RDD2022 and Roboflow pothole datasets, and the best YOLOv12n model was deployed on a Raspberry Pi 5. Results demonstrate an F1 score of 0.57, showing strong potential for low-cost, scalable road monitoring in developing regions.

---

## 🧠 Project Overview

This repository focuses on reproducibility of all experiments performed in the thesis.  
The workflow includes:

1. **Dataset preparation** — cleaning and splitting the RDD2022 and Roboflow datasets.  
2. **Initial training** — training YOLOv10, YOLOv11, and YOLOv12 models on the cleaned data.  
3. **Transfer & semi-supervised learning** — using pseudo-label generation to attempt to enhance performance.  
4. **Deployment** — converting the best YOLOv12n model to NCNN format and deploying it on a Raspberry Pi 5.

---

## 📂 Dataset Information

- **RDD2022 dataset:** [https://crddc2022.sekilab.global/](https://crddc2022.sekilab.global/)  
- **Roboflow pothole dataset:** [https://universe.roboflow.com/brad-dwyer/pothole-voxrl](https://universe.roboflow.com/brad-dwyer/pothole-voxrl)

### Preprocessing steps:
- Removed all images with empty labels to reduce training time.  
- Converted XML labels to YOLO-compatible `.txt` format.  
- Performed a 70/10/20 train/validation/test split using `train_val_test_split.ipynb`.  

---

## ⚙️ Training Pipeline

| Phase | Script | Description |
|-------|---------|-------------|
| Initial training | `YOLOv10_training.py`, `YOLOv11_training.py`, `YOLOv12_training.py` | Trains all YOLO models and compares baseline performance. |
| Transfer learning | `YOLOv12_training_full.py`, `YOLOv12_training_val_and_train.py` | Fine-tunes models on validation and combined data. |
| Pseudo-label generation | `generate_sudo_labels.py` | Generates labels for unlabelled RDD test images. |
| Semi-supervised learning | `YOLOv12_training_pseudo_data_only.py`, `YOLOv12_training_all_data.py` | Trains models on pseudo-labelled and combined data. |

---

## 💻 Deployment

- **Conversion to NCNN:** `convert_to_ncnn.py`  
  Converts the YOLOv12n model for ARM-based processors.  
- **Inference on Raspberry Pi 5:** `run_yolo.py`  
  Performs real-time video inference and saves annotated video output.

Hardware used:
- **Training:** Dual NVIDIA RTX A6000 GPUs (49 GB VRAM each)  
- **Deployment:** Raspberry Pi 5 (8 GB RAM)  

---

## Model Training Results

Find links to the run outpouts from the various training steps: [Google Drive link - training outputs](https://drive.google.com/drive/folders/1SxqA3UDCcGlkGrrAeSqFD6LG1w6nABnd?usp=drive_link)
---

## Annotated video

Find links to various annoated videos resulting from the Rasberry PI YOLOv12n model deployment: [Google Drive link - annotated videos](https://drive.google.com/drive/folders/1BTYNl8PoUMPVb_no6_JXLcIq9TwQ4lpu?usp=drive_link)
---
