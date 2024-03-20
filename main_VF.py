import cv2
import numpy as np
import torch
from super_gradients.training import models
from super_gradients.common.object_names import Models
import numpy as np
import math
from DeepSORT.deep_sort_pytorch.utils.parser import get_config
from DeepSORT.deep_sort_pytorch.deep_sort import DeepSort
from tracker_fonction import tracking
from  Yolo_nas_custom_dataset_function import model_import
from  Yolo_nas_custom_dataset_function import crime_detection_yolocustom_model
from datetime import datetime
import sys

# Fonction pour calculer la heatmap
def calculate_activation(frame, prev_frame, activation_threshold=500):
    # Convertir les images en niveaux de gris
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Calculer la différence entre les deux images
    diff = cv2.absdiff(gray_frame, gray_prev_frame)

    # seuil pour détecter les changements dans l'image
    _, threshold = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

    # Calculer la somme des pixels pour déterminer l'activation
    activation = np.sum(threshold)
    return activation

if __name__=="__main__":
    #Charger le modèle
    best_model = model_import()
    device=torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model=models.get('yolo_nas_m', pretrained_weights="coco").to(device)
    # Charger la vidéo
    video_path = r'C:\Users\davbe\Crime_detection\fighting.gif'
    cap = cv2.VideoCapture(video_path)
    # Obtenir la largeur et la hauteur des frames
    frame_width = int(cap.get(3))  # Width of the frames in the video
    frame_height = int(cap.get(4))  # Height of the frames in the video

    # Lire la première image pour l'utiliser comme référence
    ret, prev_frame = cap.read()
    if not ret:
        print("Video terminé.")
        sys.exit()


    # Create VideoWriter object
    out = cv2.VideoWriter('output_video_vf.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    while True:
        # Lire le frame suivant

        ret, frame = cap.read()
        # Afficher l'image webcam
        #cv2.imshow('Webcam', frame)

        # Attendre 1 milliseconde et vérifier si l'utilisateur appuie sur la touche 'q' pour quitter
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if not ret:
            break
        if prev_frame is None:
            print("Erreur: Image précédente vide.")
            continue  # ou effectuez une autre action en conséquence
        # Calculer l'activation
        out.write(frame)
        activation = calculate_activation(frame, prev_frame)
        # Utiliser l'activation pour décider d'activer ou non le modèle de détection
        if activation > 500:
            while (activation>10):
                ret, frame = cap.read()
                if not ret:
                    print("Erreur de capture vidéo.")
                    sys.exit()
                prediction = crime_detection_yolocustom_model(best_model,frame,ret)
                out.write(frame)
                prediction.save(r'C:\Users\davbe\Crime_detection\crime_detection\resultatdeladetection.jpg')  # Save as .mp4

                label = prediction[0].prediction.labels
                if len(label) > 0:
                    for i in label:
                        if i == 1:

                            print("activation agression à:", datetime.now())
                            out.write(frame)
                            print("LABEL: ", label)
                            index = np.where(label == 1)[0]
                            print("INDEX: ", index)
                            bboxes = prediction[0].prediction.bboxes_xyxy[index]
                            print("BBOXES: ", bboxes)
                            tracking(cap,model,bboxes)
                            out.write(frame)

                else:
                    # Mettre à jour l'image de référence
                    prev_frame = frame
                    activation = calculate_activation(frame, prev_frame)








    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()
