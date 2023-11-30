import super_gradients

from super_gradients.common.object_names import Models
from super_gradients.training import models

model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")

MEDIA_PATH = r'fighting.gif'
prediction = model.predict(MEDIA_PATH)

print(prediction)

prediction.show()
prediction.save(r'C:\Users\davbe\Crime_detection\crime_detection\result.gif') # Save as .mp4


'''
media_predictions.save("output_video.gif") # Save as .gif

bboxes = prediction.bboxes_xyxy # [Num Instances, 4] List of predicted bounding boxes for each object
poses  = prediction.poses       # [Num Instances, Num Joints, 3] list of predicted joints for each detected object (x,y, confidence)
scores = prediction.scores      # [Num Instances] - Confidence value for each predicted instance
'''