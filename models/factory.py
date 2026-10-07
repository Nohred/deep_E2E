from models.cnn import CNN
from models.mlp import MLP
from models.vgg import VGG11

# patron de diseño factory - permite crear objetos sin exponer la lógica de creación 
# al cliente y referirse a la nueva instancia mediante una interfaz común
def create_model(model_name, num_classes=10):
    if model_name == 'cnn':
        return CNN()
    if model_name == 'mlp':
        return MLP()
    if model_name == 'vgg11':
        return VGG11(num_classes=num_classes)
    raise ValueError(f"Unknown model name: {model_name}")