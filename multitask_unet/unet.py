import torch
import torch.nn as nn

from unet_parts import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 64)
        self.down_convolution_2 = DownSample(64, 128)
        self.down_convolution_3 = DownSample(128, 256)
        self.down_convolution_4 = DownSample(256, 512)

        self.bottle_neck = DoubleConv(512, 1024)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes)
        )
        # ------------------------------------

        self.up_convolution_1 = UpSample(1024, 512)
        self.up_convolution_2 = UpSample(512, 256)
        self.up_convolution_3 = UpSample(256, 128)
        self.up_convolution_4 = UpSample(128, 64)
        
        self.out_seg = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

    def forward(self, x):
       down_1, p1 = self.down_convolution_1(x)
       down_2, p2 = self.down_convolution_2(p1)
       down_3, p3 = self.down_convolution_3(p2)
       down_4, p4 = self.down_convolution_4(p3)

       b = self.bottle_neck(p4)

       # --- ROTA DE CLASSIFICAÇÃO ---
       cls_feat = self.avgpool(b)
       cls_feat = torch.flatten(cls_feat, 1) # Transforma em vetor 1D por batch
       out_cls = self.classifier(cls_feat)
       # -----------------------------

       # --- ROTA DE SEGMENTAÇÃO ---
       up_1 = self.up_convolution_1(b, down_4)
       up_2 = self.up_convolution_2(up_1, down_3)
       up_3 = self.up_convolution_3(up_2, down_2)
       up_4 = self.up_convolution_4(up_3, down_1)
       out_seg = self.out_seg(up_4)
       # -----------------------------

       return out_seg, out_cls # O modelo agora retorna duas coisas