import torch
from super_gradients.training import models

DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu' # a changé en cuda mais nous n'avons pas de GPU nvidia pour le moment
MODEL_ARCH = 'yolo_nas_m'

model = models.get(MODEL_ARCH, pretrained_weights="coco").to(DEVICE)
CONFIDENCE_TRESHOLD = 0.35

result = list(model.predict(image, conf=CONFIDENCE_TRESHOLD))[0]