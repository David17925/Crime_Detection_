
dataset = ("") # loader le dataset

##LES VERSIONS DES BIBLIOTHEQUES A AVOIR ET LES COMMANDES A LANCES SUR TERMINAL PYCHARM:

#pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 ; pip install pytorch-quantization==2.1.2 --extra-index-url https://pypi.ngc.nvidia.com ; pip install git+https://github.com/Deci-AI/super-gradients.git@master --upgrade


from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.transforms.transforms import DetectionMosaic, DetectionRandomAffine, DetectionHSV, \
    DetectionHorizontalFlip, DetectionPaddedRescale, DetectionStandardize, DetectionTargetsFormatTransform
from super_gradients.training.utils.detection_utils import DetectionCollateFN
from super_gradients.training import dataloaders
from super_gradients.training.datasets.datasets_utils import worker_init_reset_seed

#changer les images par rapport au train du dataset de anomaly detection
# Load the data
train = pd.read_csv("../data/train.csv.zip", compression='zip')
test = pd.read_csv("../data/test.csv.zip", compression='zip')
trainset = COCOFormatDetectionDataset(data_dir="/content/soccer-players-2/",
                                      images_dir="train",
                                      json_annotation_file="train/_annotations.coco.json",
                                      input_dim=(640, 640),
                                      ignore_empty_annotations=False,
                                      transforms=[
                                          DetectionMosaic(prob=1., input_dim=(640, 640)),
                                          DetectionRandomAffine(degrees=0., scales=(0.5, 1.5), shear=0.,
                                                                target_size=(640, 640),
                                                                filter_box_candidates=False, border_value=128),
                                          DetectionHSV(prob=1., hgain=5, vgain=30, sgain=30),
                                          DetectionHorizontalFlip(prob=0.5),
                                          DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
                                          DetectionStandardize(max_value=255),
                                          DetectionTargetsFormatTransform(max_targets=300, input_dim=(640, 640),
                                                                          output_format="LABEL_CXCYWH")
                                      ])


valset = COCOFormatDetectionDataset(data_dir="/content/soccer-players-2/",
                                    images_dir="valid",
                                    json_annotation_file="valid/_annotations.coco.json",
                                    input_dim=(640, 640),
                                    ignore_empty_annotations=False,
                                    transforms=[
                                        DetectionPaddedRescale(input_dim=(640, 640), max_targets=300),
                                        DetectionStandardize(max_value=255),
                                        DetectionTargetsFormatTransform(max_targets=300, input_dim=(640, 640),
                                                                        output_format="LABEL_CXCYWH")
                                    ])

train_loader = dataloaders.get(dataset=trainset, dataloader_params={
    "shuffle": True,
    "batch_size": 16,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": DetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed,
    "min_samples": 512
})

valid_loader = dataloaders.get(dataset=valset, dataloader_params={
    "shuffle": False,
    "batch_size": 32,
    "num_workers": 2,
    "drop_last": False,
    "pin_memory": True,
    "collate_fn": DetectionCollateFN(),
    "worker_init_fn": worker_init_reset_seed
})

