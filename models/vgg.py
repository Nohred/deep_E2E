import torch
import torch.nn as nn
from torchvision.models import vgg11, VGG11_Weights


class VggDetector(nn.Module):
    def __init__(
        self,
        num_classes=20,
        pretrained=True,
        freeze_backbone=True,
    ):
        super().__init__()

        weights = VGG11_Weights.DEFAULT if pretrained else None
        vgg = vgg11(weights=weights)

        # Módulos originales de VGG11, con los pesos ya cargados.
        layers = list(vgg.features.children())

        self.conv1 = nn.Sequential(*layers[0:2]) # Conv2d + ReLU 
        self.maxpool1 = layers[2]

        self.conv2 = nn.Sequential(*layers[3:5]) # Conv2d + ReLU
        self.maxpool2 = layers[5]

        self.conv3 = nn.Sequential(*layers[6:10]) # Conv2d + ReLU + Conv2d + ReLU
        self.maxpool3 = layers[10]

        self.conv4 = nn.Sequential(*layers[11:15]) # Conv2d + ReLU + Conv2d + ReLU
        self.maxpool4 = layers[15]

        self.conv5 = nn.Sequential(*layers[16:20]) # Conv2d + ReLU + Conv2d + ReLU
        self.maxpool5 = layers[20]

        # Pooling propio: no usamos el avgpool ni el classifier
        # originales de VGG.
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

        feature_dim = 512 * 4 * 4
        hidden_dim = 256

        # Cabezas nuevas: no tienen pesos de ImageNet.
        self.detection_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )

        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

        if freeze_backbone:
            self.set_backbone_trainable(False)

    def set_backbone_trainable(self, trainable):
        for block in (
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
        ):
            for parameter in block.parameters():
                parameter.requires_grad = trainable

    def forward(self, x):
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv2(x))
        x = self.maxpool3(self.conv3(x))
        x = self.maxpool4(self.conv4(x))
        x = self.maxpool5(self.conv5(x))

        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)

        boxes = self.detection_head(x)
        class_logits = self.classification_head(x)

        return boxes, class_logits