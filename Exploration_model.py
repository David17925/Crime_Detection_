import torch
from torchsummary import summary
from torchviz import make_dot

# Chargement du modèle PyTorch enregistré au format .pt
model_path = r'C:\Users\davbe\Crime_detection\yolo_nas_m.pt'
model = torch.load(model_path)

# Affichage du résumé du modèle
input_channels, input_height, input_width = (3, 416, 416)  # Remplacez ces valeurs par les dimensions d'entrée de votre modèle
summary(model, (input_channels, input_height, input_width))

# Visualisation graphique du modèle

dummy_input = torch.randn(1, input_channels, input_height, input_width)
graph = make_dot(model(dummy_input), params=dict(model.named_parameters()))
graph.render("model_plot", format="png", cleanup=True)

