from ultralytics import NAS
from super_gradients.common.object_names import Models
from super_gradients.training import models


model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")


from super_gradients.training import models
import torch
import torchvision.models as models
# import migraphx and numpy
import migraphx
import numpy as np

inception = models.inception_v3(pretrained=True)
torch.onnx.export(inception,torch.randn(1,3,299,299), "inceptioni1.onnx")

# import and parse inception model
model = migraphx.parse_onnx(r"C:\Users\davbe\Crime_detection\Crime_Detection_\inceptioni1.onnx")
# compile model for the GPU target
model.compile(migraphx.get_target("gpu"))
# optionally print compiled model
model.print()
# create random input image
input_image = r'C:\Users\davbe\Crime_detection\fighting.gif'
# feed image to model, 'x.1` is the input param name
results = model.run({'x.1': input_image})
# get the results back
result_np = np.array(results[0])
# print the inferred class of th
# Load a COCO-pretrained YOLO-NAS-s model
#model = NAS('yolo_nas_m.pt')

# Display model information (optional)
#model.info()


# Run inference with the YOLO-NAS-s model on the 'bus.jpg' image
results = model.predict(r'C:\Users\davbe\Crime_detection\fighting.gif').cuda

print(results)
# Itérez sur le générateur pour obtenir les résultats
for result in results:
    # Accédez aux boîtes englobantes (bbox) à partir de chaque résultat
    boxes = result.boxes
    print(boxes)
"""
prediction = session.run(output_names, {input_name: image})

    # Extract the confidences and bounding boxes for each person detected
    predictions = prediction[0][0]

    confidences_scores = []
    boxes_detected = []

    for detection in predictions:
        confidence = detection[0]  # Extract the confidence score
        label = int(detection[1])  # Extract the class label (assuming it is at index 1)

        if label == 0:  # Check if the label corresponds to the "person" class
            box = detection[2:]  # Extract the bounding box coordinates (x1, y1, x2, y2)

            confidences_scores.append(confidence)
            boxes_detected.append(box)

    # Convert the lists to NumPy arrays
    confidences_scores = np.array(confidences_scores)
    boxes_detected = np.array(boxes_detected)
"""