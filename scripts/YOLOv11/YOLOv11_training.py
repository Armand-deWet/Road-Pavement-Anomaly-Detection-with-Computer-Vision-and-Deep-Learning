# imports
import os
import cv2
import matplotlib.pyplot as plt
import random
from ultralytics import YOLO
import shutil
import torch
import time

yolov11_models = ['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt']

# set configuration file path
config_path = '/ext_data/armand/yolo_config.yaml'

for yolo_model in yolov11_models:

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
        name=yolo_model.split('.')[0]+'_training',
        patience=10
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    with open('yolov11_training_time.txt', 'a'):
        text_to_add = f"{yolo_model.split('.')[0]}: Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s.\n"

    print("#" * 69)
    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print("#" * 69)