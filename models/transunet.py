# transunet architecture for image segmentation

import torch
import torch.nn as nn
import torch.nn.functional as F

# standard convolutional layer with group normalization
class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1,2,3], keepdim=True, unbiased=False)
        w = (w - m) / (torch.sqrt(v) + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

# pre-activation bottleneck block
class PreActBottleneck(nn.Module):
    def __init__(self, in_ch, out_ch=None, stride=1):
        super().__init__()
        out_ch = out_ch or in_ch; mid_ch = out_ch // 4
        self.gn1 = nn.GroupNorm(32, in_ch); self.conv1 = StdConv2d(in_ch, mid_ch, 1, bias=False)
        self.gn2 = nn.GroupNorm(32, mid_ch); self.conv2 = StdConv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.gn3 = nn.GroupNorm(32, mid_ch); self.conv3 = StdConv2d(mid_ch, out_ch, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.ds = StdConv2d(in_ch, out_ch, 1, stride=stride, bias=False) if (stride != 1 or in_ch != out_ch) else None
    def forward(self, x):
        res = x; y = self.relu(self.gn1(x))
        if self.ds: res = self.ds(y)
        return self.conv3(self.relu(self.gn3(self.conv2(self.relu(self.gn2(self.conv1(y))))))) + res

# transformer block
class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=12):
        super().__init__()
        self.n1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        x = x + self.attn(self.n1(x), self.n1(x), self.n1(x))[0]; return x + self.mlp(self.n2(x))

# transunet main architecture
class TransUNet(nn.Module):
    # n_layers is 4 because it is the number of transformer blocks in the encoder
    def __init__(self, img_size=256, n_layers=4):
        super().__init__()
        self.img_size = img_size
        self.root = nn.Sequential(StdConv2d(3, 64, 7, stride=2, padding=3, bias=False), nn.GroupNorm(32, 64), nn.ReLU(inplace=True))
        self.body = nn.Sequential(PreActBottleneck(64, 256, stride=1), PreActBottleneck(256, 512, stride=2), PreActBottleneck(512, 1024, stride=2))
        self.proj = nn.Conv2d(1024, 768, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size//8)**2, 768))
        self.transformer = nn.Sequential(*[TransformerBlock(768) for _ in range(n_layers)])
        self.decoder = nn.Sequential(nn.ConvTranspose2d(768, 256, 2, 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(256, 64, 2, 2), nn.ReLU(inplace=True))
        self.head = nn.Conv2d(64, 1, 1)
    def forward(self, x):
        x = self.root(x); x = self.body(x); z = self.proj(x)
        B, C, H, W = z.shape; z = z.flatten(2).transpose(1, 2) + self.pos_embed
        z = self.transformer(z).transpose(1, 2).reshape(B, C, H, W)
        x = self.decoder(z); x = F.interpolate(x, (self.img_size, self.img_size), mode='bilinear', align_corners=False)
        return self.head(x)
