import cv2
import torch
from super_gradients.training import models
import numpy as np
import math

from DeepSORT.deep_sort_pytorch.utils.parser import get_config
from DeepSORT.deep_sort_pytorch.deep_sort import DeepSort


def compute_color_for_labels(label):
    color_palette = [
        (0, 255, 0),  # Class 0 (e.g., person)
        (0, 0, 255),  # Class 1 (e.g., car)
        (255, 0, 0),  # Class 2 (e.g., bicycle)
        # Ajoutez autant de couleurs que nécessaire pour vos classes
    ]

    # Assurez-vous que la classe a une couleur attribuée
    if 0 <= label < len(color_palette):
        return color_palette[label]
    else:
        # Si la classe n'a pas de couleur attribuée, retournez une couleur par défaut
        return (255, 255, 255)  # Blanc par défaut




def draw_boxes(img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
  for i, box in enumerate(bbox):
    x1, y1, x2, y2 = [int(i) for i in box]
    x1 += offset[0]
    x2 += offset[0]
    y1 += offset[0]
    y2 += offset[0]
    cat = int(categories[i]) if categories is not None else 0
    id = int(identities[i]) if identities is not None else 0
    cv2.rectangle(img, (x1, y1), (x2, y2), color=compute_color_for_labels(cat), thickness=2, lineType=cv2.LINE_AA)
    label = str(id) + ":" + classNames[cat]
    (w, h), _ = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 / 2, thickness=1)
    t_size = cv2.getTextSize(str(label), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1 / 2, thickness=1)[0]
    c2 = x1 + t_size[0], y1 - t_size[1] - 3
    cv2.rectangle(frame, (x1, y1), c2, color=compute_color_for_labels(cat), thickness=-1, lineType=cv2.LINE_AA)
    cv2.putText(frame, str(label), (x1, y1 - 2), 0, 1 / 2, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
  return img




cap = cv2.VideoCapture(r'/fighting.mp4')  # Change 'your_video_file.mp4' to your actual video file
# Check if the video file is opened successfully
if not cap.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

frame_width = int(cap.get(3))  # Width of the frames in the video
frame_height = int(cap.get(4))  # Height of the frames in the video
print("frame width: ", frame_width)
print("frame height: ", frame_height)


device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model=models.get('yolo_nas_l', pretrained_weights="coco").to(device)

count = 0

out = cv2.VideoWriter('../crime_detection/output_tracking.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 20, (frame_width, frame_height))



cfg_deep = get_config()
cfg_deep.merge_from_file(r"C:\Users\davbe\Crime_detection\crime_detection\DeepSORT\deep_sort_pytorch\configs\deep_sort.yaml")
deepsort = DeepSort(r'C:/Users/davbe/Crime_detection/crime_detection/DeepSORT/deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7',
                    max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT,
                    nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                    use_cuda=False) #changer à true avec un GPU nvidia disponible






classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

while True:
    xywh_bboxs = []
    confs = []
    oids = []
    output = []
    ret, frame = cap.read()
    count +=1
    print("count: ", count)
    if ret:
      result = list(model.predict(frame, conf=0.5))[0]
      bbox_xyxys = result.prediction.bboxes_xyxy.tolist()
      confidences = result.prediction.confidence
      labels = result.prediction.labels.tolist()

      print("Number of detections:", len(bbox_xyxys))  # Ajoutez cette ligne

      for (bbox_xyxy, confidence, cls) in zip(bbox_xyxys, confidences, labels):
          bbox = np.array(bbox_xyxy)
          x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
          x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
          conf = math.ceil((confidence*100))/100
          cx, cy = int((x1+x2)/2), int((y1+y2)/2)
          bbox_width = abs(x1-x2)
          bbox_height = abs(y1-y2)
          xcycwh = [cx, cy, bbox_width, bbox_height]
          xywh_bboxs.append(xcycwh)
          confs.append(conf)
          oids.append(int(cls))

      print("Number of valid detections:", len(xywh_bboxs))  # Ajoutez cette ligne

      xywhs = torch.tensor(xywh_bboxs)

      confss= torch.tensor(confs)
      print("confss: ", confss)
      outputs = deepsort.update(xywhs, confss, oids, frame)
      print("outputs: ", outputs)
      if len(outputs)>0:
          bbox_xyxy = outputs[:,:4]
          identities = outputs[:, -2]
          object_id = outputs[:, -1]
          draw_boxes(frame, bbox_xyxy, identities, object_id)
      out.write(frame)
    else:
        break
