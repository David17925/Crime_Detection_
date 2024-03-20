import super_gradients
import torch
from super_gradients.common.object_names import Models
from super_gradients.training import models
from roboflow import Roboflow

import matplotlib.pyplot as plt
import cv2

def model_import():
    DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"

    from roboflow import Roboflow
    rf = Roboflow(api_key="f9kQz0bdRZnXeykCXxk8")
    project = rf.workspace("berdav-on5oc").project("crime_detection-5vgcx")
    dataset = project.version(1).download("yolov5")

    MODEL_ARCH = 'yolo_nas_m'

    LOCATION = dataset.location
    print("location:", LOCATION)
    CLASSES = sorted(project.classes.keys())
    print("classes:", CLASSES)

    import os
    HOME = os.getcwd()
    print(HOME)

    CHECKPOINT_DIR = f'{HOME}/checkpoints'
    EXPERIMENT_NAME = project.name.lower().replace(" ", "_")

    dataset_params = {
        'data_dir': LOCATION,
        'train_images_dir': 'train/images',
        'train_labels_dir': 'train/labels',
        'val_images_dir': 'valid/images',
        'val_labels_dir': 'valid/labels',
        'test_images_dir': 'test/images',
        'test_labels_dir': 'test/labels',
        'classes': CLASSES
    }

    # model = models.get(r'Crime_Detection_\Violence Detection.v1-v1.coco\checkpoints\violence_dectection\RUN_20231207_020205_138908\ckpt_best.pth', pretrained_weights="coco")
    best_model = models.get(
        MODEL_ARCH,
        num_classes=len(dataset_params['classes']),
        checkpoint_path=r"C:\Users\davbe\Crime_detection\Crime_Detection_\Cime_detection_best_model.pth"
    ).to(DEVICE)
    # model = r'Crime_Detection_\Violence Detection.v1-v1.coco\checkpoints\violence_dectection\RUN_20231207_020205_138908\ckpt_best.pth'
    return best_model

def crime_detection_yolocustom_model(best_model,video,ret):

    if not ret:
        print("Erreur de capture vidéo.")

    print("ret: ", ret)
    prediction = best_model.predict(video)

    print(prediction)

    #prediction.show()
    # Save as .mp4
    #prediction.save(r"C:\Users\davbe\Crime_detection\Crime_Detection_\result_nas_cd_best_model.mp4")
    return prediction

'''
media_predictions.save("output_video.gif") # Save as .gif

bboxes = prediction.bboxes_xyxy # [Num Instances, 4] List of predicted bounding boxes for each object
poses  = prediction.poses       # [Num Instances, Num Joints, 3] list of predicted joints for each detected object (x,y, confidence)
scores = prediction.scores      # [Num Instances] - Confidence value for each predicted instance
'''
