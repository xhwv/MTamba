from __future__ import annotations
import torch.nn as nn
import torch
import math
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from mamba_ssm import Mamba
import torch.nn.functional as F

class MambaLayer(nn.Module):
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, num_slices=None):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v3",
            nslices=num_slices,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.bat1 = nn.BatchNorm3d(dim)
        self.bat2 = nn.BatchNorm3d(dim)
        self.relu = nn.LeakyReLU()
        # self.att=ATT(dim)
    def forward(self, x):
        # sig=self.att(x)
        B, C = x.shape[:2]
        x_skip = x
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_flat = self.norm1(x_flat)
        x_mamba = self.mamba(x_flat)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        out = out + x_skip
        out = self.bat2(out)
        return out


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim, ):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.drop=nn.Dropout(0.1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)
        self.bat=nn.BatchNorm3d(hidden_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x=self.drop(x)
        x = self.fc2(x)
        x=self.bat(x)
        return x


class GSC(nn.Module):
    def __init__(self, in_channles) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.BatchNorm3d(in_channles)
        self.nonliner = nn.LeakyReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.BatchNorm3d(in_channles)
        self.nonliner2 = nn.LeakyReLU()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.BatchNorm3d(in_channles)
        self.nonliner3 = nn.LeakyReLU()
        self.drop1=nn.Dropout(0.1)
        self.norm4=nn.BatchNorm3d(in_channles)
        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.drop2 = nn.Dropout(0.1)
        self.nonliner4 = nn.LeakyReLU()

    def forward(self, x):
        x_residual = x
        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1=self.drop1(x1)
        x1=self.nonliner(x1)
        x1 = self.proj2(x1)
        x1 = self.nonliner2(x1)
        x1=self.drop2(x1)
        x2 = self.proj3(x)
        x = x1 + x2
        x=self.norm3(x)
        x = self.proj4(x)
        x=x + x_residual
        return x


class MambaEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])

            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )

            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)

    def forward(self, x):
        x = self.forward_features(x)
        return x


class MSCB(nn.Module):
    def __init__(self, n, intc,outc):
        super(MSCB, self).__init__()

        if n == 3:
            stride = 1  # 核大小为3时，步长为1
            padding = 1  # 填充为1
        elif n == 5:
            stride = 1  # 核大小为5时，步长为2
            padding = 2  # 填充为2
        elif n == 7:
            stride = 1  # 核大小为7时，步长为3
            padding = 3  # 填充为3
        else:
            raise ValueError("只支持核大小为 3, 5 或 7")

            # 定义卷积层
        self.conv = nn.Conv3d(
            in_channels=8,  # 输入通道数
            out_channels=8,  # 输出通道数
            kernel_size=(n, n, n),  # 卷积核大小
            stride=stride,  # 固定步长
            padding=padding  # 固定填充
        )

        self.convn11 = nn.Conv3d(
            in_channels=8,
            out_channels=8,
            kernel_size=(n, 1, 1),
            stride=1,
            padding=(n // 2, 0, 0)
        )
        self.conv1n1 = nn.Conv3d(
            in_channels=8,
            out_channels=8,
            kernel_size=(1, n, 1),
            stride=1,
            padding=(0, n // 2, 0)
        )
        self.conv11n = nn.Conv3d(
            in_channels=8,
            out_channels=8,
            kernel_size=(1, 1, n),
            stride=1,
            padding=(0, 0, n // 2)
        )
        self.convnn1 = nn.Conv3d(
            in_channels=8,
            out_channels=8,
            kernel_size=(n, n, 1),
            stride=1,
            padding=(n // 2, n // 2, 0)
        )
        self.convn1n = nn.Conv3d(
            in_channels=8,
            out_channels=8,
            kernel_size=(n, 1, n),
            stride=1,
            padding=(n // 2, 0, n // 2)
        )
        self.conv1nn = nn.Conv3d(
            in_channels=8,
            out_channels=8,
            kernel_size=(1, n, n),
            stride=1,
            padding=(0, n // 2, n // 2)
        )
        self.conv1 = nn.Conv3d(
            in_channels=intc,
            out_channels=8,
            kernel_size=(1, 1, 1),
            stride=1,
            padding=(0, 0, 0)
        )
        self.conv2 = nn.Conv3d(
            in_channels=24,  # 总输出通道数（6+2+6），可根据实际需要调整
            out_channels=outc,
            kernel_size=(1, 1, 1),
            stride=1,
            padding=(0, 0, 0)
        )

        self.sig = nn.Sigmoid()
        self.bat = nn.BatchNorm3d(24)
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.conv1(x)  # 初始卷积操作
        x = self.drop(x)

        # 分支 A
        a = self.convn11(x)
        a = self.conv1n1(a)
        a = self.conv11n(a)

        # 分支 B
        b = self.convnn1(x)
        b = self.convn1n(b)
        b = self.conv1nn(b)

        # 分支 C
        c = self.conv(x)

        x = torch.cat((a, b, c), dim=1)
        x = self.bat(x)
        x = self.conv2(x)
        return x


class SMEM(nn.Module):
    def __init__(self):
        super(SMEM, self).__init__()
        self.mtc7 = MSCB(n=7, intc=8, outc=24)
        self.mtc5 = MSCB(n=5, intc=8, outc=8)
        self.mtc3 = MSCB(n=3, intc=2, outc=8)
        self.upx = nn.ConvTranspose3d(
            in_channels=24,  
            out_channels=1,  
            kernel_size=3,
            stride=2,  # 步长设置为2可以使分辨率变为原来的两倍
            padding=1,  # 填充值设置为1，配合步长和核大小保证输出尺寸合适
            output_padding=1  # 额外的输出填充，确保输出尺寸正确
        )
        self.upy = nn.ConvTranspose3d(
            in_channels=24,  
            out_channels=1,  
            kernel_size=3,
            stride=2,  # 步长设置为2可以使分辨率变为原来的两倍
            padding=1,  # 填充值设置为1，配合步长和核大小保证输出尺寸合适
            output_padding=1  # 额外的输出填充，确保输出尺寸正确
        )
        self.sig = nn.Sigmoid()
        self.bat = nn.BatchNorm3d(1)
        self.relu=nn.LeakyReLU()
        self.drop1=nn.Dropout(0.1)
        self.drop3 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.drop4 = nn.Dropout(0.1)
    def forward(self, x, y):
        s_x = x
        s_y = y
        x_half_size = tuple([s // 2 for s in x.shape[2:]])  
        y_half_size = tuple([s // 2 for s in y.shape[2:]])  
        x_max = F.adaptive_max_pool3d(x, output_size=x_half_size)  
        x_avg = F.adaptive_avg_pool3d(x, output_size=x_half_size)  
        y_max = F.adaptive_max_pool3d(y, output_size=y_half_size) 
        y_avg = F.adaptive_avg_pool3d(y, output_size=y_half_size)  
        # 在通道维度上拼接最大池化和平均池化的结果
        x = torch.cat([x_max, x_avg], dim=1)
        y = torch.cat([y_max, y_avg], dim=1)
        x = self.mtc3(x)
        y = self.mtc3(y)
        x=self.relu(x)
        x=self.drop1(x)
        y = self.relu(y)
        y = self.drop2(y)
        x = self.mtc5(x)
        y = self.mtc5(y)
        x = self.relu(x)
        x = self.drop3(x)
        y = self.relu(y)
        y = self.drop4(y)
        x = self.mtc7(x)
        y = self.mtc7(y)
        x = self.upx(x)
        y = self.upy(y)
        x = self.sig(x)
        y = self.sig(y)
        x = s_x * x
        y = s_y * y
        x = x - y
        x = self.bat(x)
        return x

class DMEM(nn.Module):
    def __init__(
            self,
            in_chans=1,
            out_chans=4,
            depths=[2, 2, 2, 2],
            feat_size=[48, 96, 192, 384],
            drop_path_rate=0,
            layer_scale_init_value=1e-6,
            hidden_size: int = 768,
            norm_name="instance",
            conv_block: bool = True,
            res_block: bool = True,
            spatial_dims=3,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.depths = depths
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.spatial_dims = spatial_dims
        self.vit = TFEncoder(in_chans,
                             depths=depths,
                             dims=feat_size,
                             drop_path_rate=drop_path_rate,
                             layer_scale_init_value=layer_scale_init_value,
                             )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.hidden_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.gdm=SMEM()
    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x, y):
        x = self.gdm(x,y)
        outs = self.vit(x)
        enc_hidden = self.encoder5(outs[3])
        return enc_hidden

class TFEncoder(nn.Module):
    def __init__(self, in_chans=1, depths=[2, 2, 2, 2], dims=[48, 96, 192, 384],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3]):
        super().__init__()
        self.downsample_layers_down = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_down = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=7, stride=2, padding=3),
        )
        self.downsample_layers_down.append(stem_down)
        for i in range(3):
            downsample_layer_down = nn.Sequential(
                # LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers_down.append(downsample_layer_down)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.bas=nn.ModuleList()
        num_slices_list = [64, 32, 16, 8]
        cur = 0
        for i in range(4):
            gsc = GSC(dims[i])
            stage = nn.Sequential(
                *[MambaLayer(dim=dims[i], num_slices=num_slices_list[i]) for j in range(depths[i])]
            )
            ba = nn.BatchNorm3d(dims[i])
            self.bas.append(ba)
            self.stages.append(stage)
            self.gscs.append(gsc)
            cur += depths[i]
        self.out_indices = out_indices
        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer]))

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers_down[i](x)
            x = x
            x=self.bas[i](x)
            x = self.gscs[i](x)
            x = self.stages[i](x)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)

        return tuple(outs)
    def forward(self, x):
        x = self.forward_features(x)
        return x

