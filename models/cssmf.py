import torch.nn.functional as F
import torch.nn as nn
from mamba_ssm import Mamba
import torch

class CSM(nn.Module):
    def __init__(self, ):
        super(CSM, self).__init__()
        self.conv=DepthwiseSeparableConv3d(768)
        self.relu=nn.LeakyReLU()
        self.drop=nn.Dropout(0.1)
        self.mamba1 = Mamba(
            d_model=768,  # Model dimension d_model
            d_state=64,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
            bimamba_type="v3",
            nslices=8,
        )
        self.norm = nn.LayerNorm(768)
        self.bn=nn.BatchNorm3d(768)
    def forward(self, x):
        x=self.bn(x)
        B, C = x.shape[:2]
        x=self.drop(x)
        x = self.relu(x)
        x=self.conv(x)
        x=self.drop(x)
        x=self.relu(x)
        n_tokens = x.shape[2:].numel()
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2).contiguous()
        x_flat=self.norm(x_flat)
        x_flip_l = torch.flip(x_flat, dims=[2]).contiguous()
        x_flip_c = torch.flip(x_flat, dims=[1]).contiguous()
        x_flip_lc = torch.flip(x_flat, dims=[1, 2]).contiguous()
        x_ori = self.mamba1(x_flat)
        x_mamba_l = self.mamba1(x_flip_l)
        x_mamba_c = self.mamba1(x_flip_c)
        x_mamba_lc = self.mamba1(x_flip_lc)
        x_ori_l = torch.flip(x_mamba_l, dims=[2])
        x_ori_c = torch.flip(x_mamba_c, dims=[1])
        x_ori_lc = torch.flip(x_mamba_lc, dims=[1, 2])
        out = (x_ori + x_ori_l + x_ori_c + x_ori_lc) / 4
        return out

class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, in_channels, kernel_size=1,
                                   stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class CSSMF(nn.Module):
    def __init__(self, ):
        super(CSSMF, self).__init__()
        self.csmx=CSM()
        self.drop=nn.Dropout(0.1)
        self.relu=nn.LeakyReLU()
        self.dwconv=DepthwiseSeparableConv3d(768)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.ln = nn.LayerNorm(768)
        self.bn1 = nn.BatchNorm3d(768)

    def forward(self, x, y):
        B, C, D, H, W = x.shape
        shortx=x
        shorty=y
        img_dims = x.shape[2:]
        x=x+y
        x=self.csmx(x)
        x = x.transpose(-1, -2).reshape(B, C, *img_dims)
        x=x+shortx+shorty
        x=self.bn1(x)
        x=self.dwconv(x)
        x=self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x=self.ln(x)
        return x

class IDH_Predict(nn.Module):
    def __init__(self):
        super(IDH_Predict, self).__init__()
        self.kl1 = nn.Linear(768, 1024)
        self.kl2 = nn.Linear(1024,512)
        self.kl3 = nn.Linear(512, 64)
        self.kl4 = nn.Linear(64, 2)
        self.drop1= nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU()
        # self.norm1 = nn.LayerNorm(512)
        # self.norm2 = nn.LayerNorm(64)
    def forward(self, x):
        x = self.kl1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.kl2(x)
        x = self.relu(x)
        x = self.drop2(x)
        x=self.kl3(x)
        x = self.relu(x)
        x = self.drop2(x)
        x=self.kl4(x)
        return x


class Grade_Predict(nn.Module):
    def __init__(self):
        super(Grade_Predict, self).__init__()
        self.kl1 = nn.Linear(768, 1024)
        self.kl2 = nn.Linear(1024, 512)
        self.kl3 = nn.Linear(512, 64)
        self.kl4 = nn.Linear(64, 2)
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.relu = nn.LeakyReLU()
    def forward(self, x):
        x = self.kl1(x)
        x = self.relu(x)
        x = self.drop1(x)
        x = self.kl2(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.kl3(x)
        x = self.relu(x)
        x = self.drop2(x)
        x = self.kl4(x)
        return x