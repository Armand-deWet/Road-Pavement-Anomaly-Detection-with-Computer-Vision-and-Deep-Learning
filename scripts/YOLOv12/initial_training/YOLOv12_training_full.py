# imports
import os
import cv2
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO
import shutil
import torch
import time

yolov12_models = [
    '/home/armand/ext_data/yolov12/runs/detect/yolo12n_training/weights/best.pt', 
    '/home/armand/ext_data/yolov12/runs/detect/yolo12s_training/weights/best.pt', 
    '/home/armand/ext_data/yolov12/runs/detect/yolo12m_training/weights/best.pt', 
    '/home/armand/ext_data/yolov12/runs/detect/yolo12l_training/weights/best.pt', 
    '/home/armand/ext_data/yolov12/runs/detect/yolo12x_training/weights/best.pt'
    ]

# set configuration file path
config_path = '/ext_data/armand/yolo_config.yaml'

for yolo_model in yolov12_models:

    start_time = time.time()

    # set model
    model = YOLO(yolo_model)

    # initiate training
    train_results = model.train(
        data=config_path,
        epochs=200,
        imgsz=640,
        batch=-1,
        device=0,
        save_period = 25,
        cos_lr=True,
        name=yolo_model.split('/')[-3]+'_full',
        patience=10
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    text_to_add = f"{yolo_model.split('.')[0]}: Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s.\n"

    with open('yolov12_training_time_full.txt', 'a') as file:
        file.write(text_to_add)

    print("#" * 69)
    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("#" * 69)