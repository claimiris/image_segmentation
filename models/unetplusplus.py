# UNet++ Architecture for Image Segmentation

import torch
import torch.nn as nn

# Basic convolution block: Conv → BatchNorm → ReLU
class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.block(x)

# Double convolution block used in encoder and decoder
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(ConvBnRelu(in_ch, out_ch), ConvBnRelu(out_ch, out_ch))
    def forward(self, x): return self.block(x)

# Upsampling block
class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_ch, out_ch, 1, bias=False)
    def forward(self, x): return self.conv(self.up(x))

# UNet++ main architecture
class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, deep_supervision=True):
        super().__init__()
        f = [32, 64, 128, 256, 512]
        self.deep_supervision = deep_supervision
        self.pool = nn.MaxPool2d(2)
        
        # Encoder (contracting path)
        self.enc0_0 = DoubleConv(in_channels, f[0])
        self.enc1_0 = DoubleConv(f[0], f[1])
        self.enc2_0 = DoubleConv(f[1], f[2])
        self.enc3_0 = DoubleConv(f[2], f[3])
        self.enc4_0 = DoubleConv(f[3], f[4])

        # Nested skip connections
        self.up_3_1 = UpSample(f[4], f[3]); self.node3_1 = DoubleConv(f[3]*2, f[3])
        self.up_2_1 = UpSample(f[3], f[2]); self.node2_1 = DoubleConv(f[2]*2, f[2])
        self.up_1_1 = UpSample(f[2], f[1]); self.node1_1 = DoubleConv(f[1]*2, f[1])
        self.up_0_1 = UpSample(f[1], f[0]); self.node0_1 = DoubleConv(f[0]*2, f[0])

        self.up_2_2 = UpSample(f[3], f[2]); self.node2_2 = DoubleConv(f[2]*3, f[2])
        self.up_1_2 = UpSample(f[2], f[1]); self.node1_2 = DoubleConv(f[1]*3, f[1])
        self.up_0_2 = UpSample(f[1], f[0]); self.node0_2 = DoubleConv(f[0]*3, f[0])

        self.up_1_3 = UpSample(f[2], f[1]); self.node1_3 = DoubleConv(f[1]*4, f[1])
        self.up_0_3 = UpSample(f[1], f[0]); self.node0_3 = DoubleConv(f[0]*4, f[0])

        self.up_0_4 = UpSample(f[1], f[0]); self.node0_4 = DoubleConv(f[0]*5, f[0])
        
        # Deep supervision outputs
        self.out1 = nn.Conv2d(f[0], out_channels, 1)
        self.out2 = nn.Conv2d(f[0], out_channels, 1)
        self.out3 = nn.Conv2d(f[0], out_channels, 1)
        self.out4 = nn.Conv2d(f[0], out_channels, 1)

    def forward(self, x):
        x0_0 = self.enc0_0(x);           x1_0 = self.enc1_0(self.pool(x0_0))
        x2_0 = self.enc2_0(self.pool(x1_0)); x3_0 = self.enc3_0(self.pool(x2_0))
        x4_0 = self.enc4_0(self.pool(x3_0))

        x3_1 = self.node3_1(torch.cat([x3_0, self.up_3_1(x4_0)], 1))
        x2_1 = self.node2_1(torch.cat([x2_0, self.up_2_1(x3_1)], 1))
        x1_1 = self.node1_1(torch.cat([x1_0, self.up_1_1(x2_1)], 1))
        x0_1 = self.node0_1(torch.cat([x0_0, self.up_0_1(x1_1)], 1))

        x2_2 = self.node2_2(torch.cat([x2_0, x2_1, self.up_2_2(x3_1)], 1))
        x1_2 = self.node1_2(torch.cat([x1_0, x1_1, self.up_1_2(x2_2)], 1))
        x0_2 = self.node0_2(torch.cat([x0_0, x0_1, self.up_0_2(x1_2)], 1))

        x1_3 = self.node1_3(torch.cat([x1_0, x1_1, x1_2, self.up_1_3(x2_2)], 1))
        x0_3 = self.node0_3(torch.cat([x0_0, x0_1, x0_2, self.up_0_3(x1_3)], 1))

        x0_4 = self.node0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up_0_4(x1_3)], 1))

        o1, o2, o3, o4 = self.out1(x0_1), self.out2(x0_2), self.out3(x0_3), self.out4(x0_4)
        if self.deep_supervision:
            return [o1, o2, o3, o4] if self.training else (o1+o2+o3+o4)/4.0
        return o4
