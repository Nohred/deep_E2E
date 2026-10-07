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



class UNet(nn.Module):
    def __init__(self,num_classes=21):
        super(UNet, self).__init__()

        self.encoder1 = ConvBlock(3, 32)
        self.encoder2 = ConvBlock(32, 64) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Max pooling tumba valore

        self.latent = ConvBlock(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = ConvBlock(64, 32)

        self.output = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x) # 3 canales de entrada, 32 canales de salida dim(256,256,32)
        e2 = self.encoder2(self.pool(e1)) # 32 canales de entrada, 64 canales de salida dim(128,128,64)

        # Latent space
        l = self.latent(self.pool(e2)) # 64 canales de entrada, 128 canales de salida dim(64,64,128)

        # Decoder
        d2 = self.up2(l) # 64 canales de entrada, 64 canales de salida dim(128,128,64)
        d2 = torch.cat((d2, e2), dim=1)  # Skip connection - dim (128,128,128)
        d2 = self.decoder2(d2) # 64 canales de entrada, 64 canales de salida dim(128,128,64)

        # De
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)  # Skip connection
        d1 = self.decoder1(d1)

        return self.output(d1)
        
