from ultralytics import NAS
from super_gradients.common.object_names import Models
from super_gradients.training import models

model = models.get(Models.YOLO_NAS_M, pretrained_weights="coco")




# Load a COCO-pretrained YOLO-NAS-s model
#model = NAS('yolo_nas_m.pt')

# Display model information (optional)
#model.info()


# Run inference with the YOLO-NAS-s model on the 'bus.jpg' image
results = model.predict(r'C:\Users\davbe\Crime_detection\fighting.gif')

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