import os
HOME = os.getcwd()
print(HOME)

import torch
from multiprocessing import freeze_support


if __name__ == '__main__':
    freeze_support()
    # Rest of your code


    DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"


    import roboflow
    from roboflow import Roboflow

    #roboflow.login()

    #rf = Roboflow()

    #project = rf.workspace("roboflow-jvuqo").project("football-players-detection-2frwp")
    #dataset = project.version(1).download("yolov5")



    from roboflow import Roboflow
    rf = Roboflow(api_key="Un8OzsDp2YusNmsekfEY")
    project = rf.workspace("berdav-on5oc").project("violence_dectection")
    dataset = project.version(3).download("yolov5")


    LOCATION = dataset.location
    print("location:", LOCATION)
    CLASSES = sorted(project.classes.keys())
    print("classes:", CLASSES)


    MODEL_ARCH = 'yolo_nas_m'
    BATCH_SIZE = 8
    MAX_EPOCHS = 25
    CHECKPOINT_DIR = f'{HOME}/checkpoints'
    EXPERIMENT_NAME = project.name.lower().replace(" ", "_")

    print("AAAAA")

    from super_gradients.training import Trainer

    trainer = Trainer(experiment_name=EXPERIMENT_NAME, ckpt_root_dir=CHECKPOINT_DIR)

    print("BBBBBB")
    dataset_params = {
        'data_dir': LOCATION,
        'train_images_dir':'train/images',
        'train_labels_dir':'train/labels',
        'val_images_dir':'valid/images',
        'val_labels_dir':'valid/labels',
        'test_images_dir':'test/images',
        'test_labels_dir':'test/labels',
        'classes': CLASSES
    }

    print("CCCCCCCCC")

    from super_gradients.training.dataloaders.dataloaders import (
        coco_detection_yolo_format_train, coco_detection_yolo_format_val)

    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 4
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 2
        }
    )

    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['test_images_dir'],
            'labels_dir': dataset_params['test_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': BATCH_SIZE,
            'num_workers': 2
        }
    )

    print("EEEE")

    train_data.dataset.transforms

    print("FFFFFFFFFF")

    from super_gradients.training import models

    model = models.get(
        MODEL_ARCH,
        num_classes=len(dataset_params['classes']),
        pretrained_weights="coco"
    )




    from super_gradients.training.losses import PPYoloELoss
    from super_gradients.training.metrics import DetectionMetrics_050
    from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

    train_params = {
        'silent_mode': False,
        "average_best_models":True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": MAX_EPOCHS,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }



    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data
    )




    #%load_ext tensorboard
    #%tensorboard --logdir {CHECKPOINT_DIR}/{EXPERIMENT_NAME}

    #!zip -r yolo_nas.zip {CHECKPOINT_DIR}/{EXPERIMENT_NAME}

    import os
    import shutil
    from tensorboard import program

    # Spécifier le répertoire des journaux TensorBoard
    logdir = os.path.join(CHECKPOINT_DIR, EXPERIMENT_NAME)

    # Démarrer TensorBoard
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', logdir])
    url = tb.launch()

    # Afficher l'URL de TensorBoard dans la console
    print(f"TensorBoard is running at {url}")

    # Créer une archive zip
    zip_filename = os.path.join(CHECKPOINT_DIR, f"{EXPERIMENT_NAME}.zip")
    shutil.make_archive(zip_filename[:-4], 'zip', logdir)

    # Fermer TensorBoard après la création de l'archive
    tb.close()


    best_model = models.get(
        MODEL_ARCH,
        num_classes=len(dataset_params['classes']),
        checkpoint_path=f"{CHECKPOINT_DIR}/{EXPERIMENT_NAME}/average_model.pth"
    ).to(DEVICE)




    trainer.test(
        model=best_model,
        test_loader=test_data,
        test_metrics_list=DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    )




    import supervision as sv

    ds = sv.DetectionDataset.from_yolo(
        images_directory_path=f"{dataset.location}/test/images",
        annotations_directory_path=f"{dataset.location}/test/labels",
        data_yaml_path=f"{dataset.location}/data.yaml",
        force_masks=False
    )



    import supervision as sv

    CONFIDENCE_TRESHOLD = 0.5

    predictions = {}

    for image_name, image in ds.images.items():
        result = list(best_model.predict(image, conf=CONFIDENCE_TRESHOLD))[0]
        detections = sv.Detections(
            xyxy=result.prediction.bboxes_xyxy,
            confidence=result.prediction.confidence,
            class_id=result.prediction.labels.astype(int)
        )
        predictions[image_name] = detections



    import random
    random.seed(10)

    import supervision as sv

    MAX_IMAGE_COUNT = 5

    n = min(MAX_IMAGE_COUNT, len(ds.images))

    keys = list(ds.images.keys())
    keys = random.sample(keys, n)

    box_annotator = sv.BoxAnnotator()

    images = []
    titles = []

    for key in keys:
        frame_with_annotations = box_annotator.annotate(
            scene=ds.images[key].copy(),
            detections=ds.annotations[key],
            skip_label=True
        )
        images.append(frame_with_annotations)
        titles.append('annotations')
        frame_with_predictions = box_annotator.annotate(
            scene=ds.images[key].copy(),
            detections=predictions[key],
            skip_label=True
        )
        images.append(frame_with_predictions)
        titles.append('predictions')


    import matplotlib.pyplot as plt

    sv.plot_images_grid(images=images, titles=titles, grid_size=(n, 2), size=(2 * 4, n * 4))

    plt.show()





    import os

    import numpy as np

    from onemetric.cv.object_detection import ConfusionMatrix

    keys = list(ds.images.keys())

    annotation_batches, prediction_batches = [], []

    for key in keys:
        annotation=ds.annotations[key]
        annotation_batch = np.column_stack((
            annotation.xyxy,
            annotation.class_id
        ))
        annotation_batches.append(annotation_batch)

        prediction=predictions[key]
        prediction_batch = np.column_stack((
            prediction.xyxy,
            prediction.class_id,
            prediction.confidence
        ))
        prediction_batches.append(prediction_batch)

    confusion_matrix = ConfusionMatrix.from_detections(
        true_batches=annotation_batches,
        detection_batches=prediction_batches,
        num_classes=len(ds.classes),
        conf_threshold=CONFIDENCE_TRESHOLD
    )

    confusion_matrix.plot(os.path.join(HOME, "confusion_matrix.png"), class_names=ds.classes)