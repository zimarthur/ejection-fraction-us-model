import torch
import torch.nn as nn
from unet_parts import DoubleConv, DownSample, UpSample

class MultiTaskUNet(nn.Module):
    def __init__(self, in_channels=1, num_classes_seg=1, base_filters=16):
        super().__init__()

        f1 = base_filters
        f2 = f1 * 2
        f3 = f2 * 2
        f4 = f3 * 2
        f5 = f4 * 2

        # ENCODER
        self.down_convolution_1 = DownSample(in_channels, f1)
        self.down_convolution_2 = DownSample(f1, f2)
        self.down_convolution_3 = DownSample(f2, f3)
        self.down_convolution_4 = DownSample(f3, f4)

        self.bottle_neck = nn.Sequential(
            DoubleConv(f4, f5),
            nn.Dropout2d(0.3)
        )

        # DECODER (Segmentação)
        self.up_convolution_1 = UpSample(f5, f4)
        self.up_convolution_2 = UpSample(f4, f3)
        self.up_convolution_3 = UpSample(f3, f2)
        self.up_convolution_4 = UpSample(f2, f1)
        self.seg_out = nn.Conv2d(in_channels=f1, out_channels=num_classes_seg, kernel_size=1)

        # CLASSIFICATION HEAD (A2C vs A4C)
        self.class_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(f5, f5 // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(f5 // 2, 1) 
        )

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)

        b = self.bottle_neck(p4)

        # Saída 1: Máscara
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)
        mask_pred = self.seg_out(up_4)

        # Saída 2: Classe
        class_pred = self.class_head(b)

        return mask_pred, class_pred