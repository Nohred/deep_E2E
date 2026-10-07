import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )


    def forward(self, x):
        return self.block(x)



class ConvDetector(nn.Module):
    def __init__(self,num_classes=21):
        super(ConvDetector, self).__init__()

        self.conv1 = ConvBlock(3, 32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = ConvBlock(32, 64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(64, 128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = ConvBlock(128, 256)

        self.avgpool = nn.AdaptiveAvgPool2d((4, 4)) 

        feature_dim = 256 * 4 * 4
        hidden_dim = 256

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


    def forward(self, x):

        # Feature extraction
        c1 = self.conv1(x)
        p1 = self.maxpool1(c1)
        c2 = self.conv2(p1)
        p2 = self.maxpool2(c2)
        c3 = self.conv3(p2)
        p3 = self.maxpool3(c3)
        c4 = self.conv4(p3)

        # Global average pooling
        x = self.avgpool(c4)

        # Flatten the tensor
        x = torch.flatten(x, start_dim=1)

        # Detection and classification
        boxes = self.detection_head(x)
        class_logits = self.classification_head(x)

        return boxes, class_logits
