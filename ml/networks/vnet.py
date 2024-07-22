import torch
from torch import nn
import torch.nn.functional as F
from secretflow.ml.nn.core.torch import BaseModule # for secretflow

# for training_step and so on
from ..utils.util import compute_sdf_and_edge, select_device, compute_sdf
from torch.nn import BCEWithLogitsLoss, MSELoss
from ..utils import ramps, losses #, metrics
import torch.optim as optim
from torchmetrics import MeanSquaredError, Metric
from ml.utils.metric import DiceCoefficient, JaccardIndex, AverageSurfaceDistance, HausdorffDistance95

class ProjectExciteLayer(BaseModule):
    """
    Redesign the spatial information integration method for
    feature recalibration, based on the original
    Project & Excite Module
    """

    def __init__(self, num_channels, D, H, W, reduction_ratio=2):
        """
        :param num_channels: Num of input channels 输入张量的通道数
        :param D, H, W: Spatial dimension of the input feature cube 输入张量的空间维度
        :param reduction_ratio: By how much should the num_channels should be reduced 通道数要减少多少倍
        """
        super(ProjectExciteLayer, self).__init__() 
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        # kernel_size是卷积核的大小，stride是步长的大小
        self.convModule = nn.Sequential(
            nn.Conv3d(in_channels=num_channels, out_channels=num_channels_reduced, kernel_size=1, stride=1), \
            nn.ReLU(inplace=True), \
            nn.Conv3d(in_channels=num_channels_reduced, out_channels=num_channels, kernel_size=1, stride=1), \
            nn.Sigmoid())
        self.spatialdim = [D, H, W]

        self.D_squeeze = nn.Conv3d(in_channels=D, out_channels=1, kernel_size=1, stride=1)
        self.H_squeeze = nn.Conv3d(in_channels=H, out_channels=1, kernel_size=1, stride=1)
        self.W_squeeze = nn.Conv3d(in_channels=W, out_channels=1, kernel_size=1, stride=1)

        self.dw = nn.ModuleList()
        for i in range(num_channels):
            conv_layer = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, dilation=1)
            self.dw.append(conv_layer)

        self.ins = nn.InstanceNorm3d(num_channels)

        self.pw = nn.Conv3d(num_channels, num_channels, kernel_size=1)


    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W) 输入一个5维张量input_tensor
        :return: output tensor, mapping
        """
        D, H, W = self.spatialdim[0], self.spatialdim[1], self.spatialdim[2]
        D_channel = input_tensor.permute(0, 2, 1, 3, 4)
        H_channel = input_tensor.permute(0, 3, 2, 1, 4)
        W_channel = input_tensor.permute(0, 4, 2, 3, 1)

        squeeze_tensor_1D = self.D_squeeze(D_channel)
        squeeze_tensor_HW = squeeze_tensor_1D.permute(0, 2, 1, 3, 4)

        squeeze_tensor_1H = self.H_squeeze(H_channel)
        squeeze_tensor_DW = squeeze_tensor_1H.permute(0, 3, 2, 1, 4)

        squeeze_tensor_1W = self.W_squeeze(W_channel)
        squeeze_tensor_DH = squeeze_tensor_1W.permute(0, 4, 2, 3, 1)

        out = []
        for i in range(input_tensor.shape[1] - 1):
            out.append(self.dw[i](input_tensor[:, i:i + 1]))
        out.append(self.dw[input_tensor.shape[1] - 1](input_tensor[:, input_tensor.shape[1] - 1:]))
        out = torch.cat([i for i in out], dim=1)
        ins = self.ins(out)
        squeeze_tensor_C = self.pw(ins)

        final_squeeze_tensor = squeeze_tensor_HW + squeeze_tensor_DW + squeeze_tensor_DH + squeeze_tensor_C

        final_squeeze_tensor = self.convModule(final_squeeze_tensor)
        output_tensor = torch.mul(input_tensor, final_squeeze_tensor)

        return output_tensor

class SpatialDropout(BaseModule):
    def __init__(self, channel_num=1, thr=0.3):
        super(SpatialDropout, self).__init__()
        self.channel_num = channel_num
        self.threshold = thr

    def forward(self, x):
        if self.training:
            r = torch.rand(x.shape[0], self.channel_num, 1, 1, 1).cuda()
            r[r < self.threshold] = 0
            r[r >= self.threshold] = 1
            r = r * self.channel_num / (r.sum() + 0.01)
            return x * r
        else:
            return x

class ConvBlock(BaseModule):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(BaseModule):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x

# 下采样
class DownsamplingConvBlock(BaseModule):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(BaseModule):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(BaseModule):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear', align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class VNet(BaseModule):
    def __init__(self, n_channels=1, n_classes=1, n_filters=16, down_sample=1, normalization='groupnorm', has_dropout=True,
                 has_residual=False, Dmax = 96, Hmax = 128, Wmax = 160):
        super(VNet, self).__init__()
        self.iter_num = 0
        self.log_var_a = torch.zeros((1,)).cuda()
        self.log_var_a.requires_grad = True
        self.log_var_b = torch.zeros((1,)).cuda()
        self.log_var_b.requires_grad = True
        self.ce_loss = BCEWithLogitsLoss().to(self.device)
        self.mse_loss = MSELoss().to(self.device)

        self.has_dropout = has_dropout
        convBlock = ConvBlock if not has_residual else ResidualConvBlock

        self.block_one = convBlock(1, n_channels, n_filters, normalization=normalization)
        self.block_one_fr = ProjectExciteLayer(16, Dmax, Hmax, Wmax)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization) # 下采样

        self.block_two = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_two_fr = ProjectExciteLayer(32, Dmax // 2, Hmax // 2, Wmax // 2)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_three_fr = ProjectExciteLayer(64, Dmax // 4, Hmax // 4, Wmax // 4)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_four_fr = ProjectExciteLayer(128, Dmax // 8, Hmax // 8, Wmax // 8)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_fr = ProjectExciteLayer(256, Dmax // 16, Hmax // 16, Wmax // 16)
        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_fr = ProjectExciteLayer(128, Dmax // 8, Hmax // 8, Wmax // 8)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_fr = ProjectExciteLayer(64, Dmax // 4, Hmax // 4, Wmax // 4)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_fr = ProjectExciteLayer(32, Dmax // 2, Hmax // 2, Wmax // 2)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = convBlock(1, n_filters, n_filters, normalization=normalization)
        self.block_nine_fr = ProjectExciteLayer(16, Dmax, Hmax, Wmax)

        ##########
        self.block_five1 = convBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.block_five_up1 = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six1 = convBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.block_six_up1 = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven1 = convBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.block_seven_up1 = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight1 = convBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.block_eight_up1 = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine1 = convBlock(1, n_filters, n_filters, normalization=normalization)

        ######
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.out_conv2 = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        self.tanh = nn.Tanh()


        ######

        self.convgs1 = nn.Conv3d(16, 2, kernel_size=1, stride=1, padding=0)
        self.convgs2 = nn.Conv3d(32, 2, kernel_size=1, stride=1, padding=0)
        self.convgs3 = nn.Conv3d(64, 2, kernel_size=1, stride=1, padding=0)
        self.convgs4 = nn.Conv3d(128, 2, kernel_size=1, stride=1, padding=0)
        self.convgs6 = nn.Conv3d(128, 2, kernel_size=1, stride=1, padding=0)
        self.convgs7 = nn.Conv3d(64, 2, kernel_size=1, stride=1, padding=0)
        self.convgs8 = nn.Conv3d(32, 2, kernel_size=1, stride=1, padding=0)
        self.convgs9 = nn.Conv3d(16, 2, kernel_size=1, stride=1, padding=0)
        self.up_sample1 = nn.Upsample(scale_factor=down_sample, mode='trilinear', align_corners=True)
        self.up_sample2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up_sample3 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up_sample4 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up_sample6 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up_sample7 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up_sample8 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up_sample9 = nn.Upsample(scale_factor=down_sample, mode='trilinear', align_corners=True)

        self.dc0_0 = nn.Conv3d(8, n_classes, kernel_size=1, stride=1, padding=0)
        self.dc0_1 = nn.Conv3d(8, n_classes, kernel_size=1, stride=1, padding=0)
        self.dc0_0_edge = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)
        self.dc0_1_edge = nn.Conv3d(8, 1, kernel_size=1, stride=1, padding=0)

        self.dropout1 = SpatialDropout(channel_num=8, thr=0.3)
        self.dropout2 = SpatialDropout(channel_num=8, thr=0.3)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def encoder(self, input):
        x1 = self.block_one(input)
        x1 = self.block_one_fr(x1)
        s1 = self.convgs1(x1)
        s1 = self.up_sample1(s1)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2 = self.block_two_fr(x2)
        s2 = self.convgs2(x2)
        s2 = self.up_sample2(s2)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3 = self.block_three_fr(x3)
        s3 = self.convgs3(x3)
        s3 = self.up_sample3(s3)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4 = self.block_four_fr(x4)
        s4 = self.convgs4(x4)
        s4 = self.up_sample4(s4)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        x5 = self.block_five_fr(x5)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]
        pred = self.dc0_0(self.dropout1(torch.cat((s1, s2, s3, s4), 1)))

        return res,pred

    def decoder_seg(self, features, pred):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5) #upsampling
        x5_up = x5_up + x4 # cat

        x6 = self.block_six(x5_up)
        x6 = self.block_six_fr(x6)
        s6 = self.convgs6(x6)
        s6 = self.up_sample6(s6)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7 = self.block_seven_fr(x7)
        s7 = self.convgs7(x7)
        s7 = self.up_sample7(s7)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8 = self.block_eight_fr(x8)
        s8 = self.convgs8(x8)
        s8 = self.up_sample8(s8)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1

        x9 = self.block_nine(x8_up)
        x9 = self.block_nine_fr(x9)
        s9 = self.convgs9(x9)
        s9 = self.up_sample9(s9)
        if self.has_dropout:
            x9 = self.dropout(x9)

        out_seg0 = pred
        out_seg1 = self.dc0_1(self.dropout2(torch.cat((s6, s7, s8, s9), 1)))

        return out_seg0, out_seg1

    def decoder_sdf(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up1(x5)
        x5_up = x5_up + x4

        x6 = self.block_six1(x5_up)
        x6_up = self.block_six_up1(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven1(x6_up)
        x7_up = self.block_seven_up1(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight1(x7_up)
        x8_up = self.block_eight_up1(x8)
        x8_up = x8_up + x1
        x9 = self.block_nine1(x8_up)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        out_tanh = self.tanh(out)

        return out_tanh

    def forward(self, x, turnoff_drop=False):
        if turnoff_drop:
            has_dropout = self.has_dropout
            self.has_dropout = False
        features, pred = self.encoder(x)
        out_seg0, out_seg1 = self.decoder_seg(features, pred) # 分割结果
        out_tanh = self.decoder_sdf(features)  # 有符号距离函数

        if turnoff_drop:
            self.has_dropout = has_dropout
        return out_seg0, out_seg1, out_tanh

    @property
    def device(self):
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def compute_sdf_loss(self, label_batch, out_seg, pred_dis, gt_dis):
        mse_loss = self.mse_loss
        sdf_loss = mse_loss(out_seg[:, 0, ...], gt_dis)
        sdf_consistency_loss = mse_loss(out_seg[:, 0, ...], pred_dis)
        return sdf_loss, sdf_consistency_loss

    def compute_seg_loss(self, out_seg, label_batch):
        ce_loss = self.ce_loss
        loss_seg = ce_loss(out_seg[:, 0, ...], label_batch[:].float())
        outputs_soft = torch.sigmoid(out_seg)
        loss_seg_dice = losses.dice_loss(outputs_soft[:, 0, :, :, :], label_batch[:] == 1)
        return loss_seg + loss_seg_dice

    def compute_sdf(self, data, shape):
        return torch.from_numpy(compute_sdf(data.cpu().numpy(), shape)).float().to(self.device)

    def get_current_consistency_weight(self, epoch):
        return 0.01 * ramps.sigmoid_rampup(epoch, 40)

    def loss_function(self, sampled_batch, batch_idx, out_seg0, out_seg1, out_tanh):
        volume_batch, label_batch = sampled_batch
        volume_batch, label_batch = volume_batch.to(self.device), label_batch.to(self.device)

        with torch.no_grad():
            gt_dis0 = self.compute_sdf(label_batch, out_seg0[:, 0, ...].shape)
            gt_dis1 = self.compute_sdf(label_batch, out_seg1[:, 0, ...].shape)
            pred_dis0 = self.compute_sdf(out_seg0[:, 0, ...], out_seg0[:, 0, ...].shape)
            pred_dis1 = self.compute_sdf(out_seg1[:, 0, ...], out_seg1[:, 0, ...].shape)

        loss_seg0 = self.compute_seg_loss(out_seg0, label_batch)
        loss_seg1 = self.compute_seg_loss(out_seg1, label_batch)
        loss_seg = loss_seg0 + loss_seg1

        loss_sdf, sdf_consistency_loss = self.compute_sdf_loss(label_batch, out_tanh, pred_dis1, gt_dis1)

        precision_a = torch.exp(-self.log_var_a)
        precision_b = torch.exp(-self.log_var_b)

        consistency_weight = self.get_current_consistency_weight(self.iter_num//250)
        self.iter_num = self.iter_num + 1
        supervised_loss = precision_a * (loss_seg) + 2 * self.log_var_a + precision_b * loss_sdf + self.log_var_b
        loss = supervised_loss + consistency_weight * sdf_consistency_loss

        return loss

    def training_step(self, batch, batch_idx: int, dataloader_idx: int = 0, **kwargs):
        x, y = batch
        out_seg0, out_seg1, out_tanh = self(x)
        loss = self.loss_function(batch, batch_idx, out_seg0, out_seg1, out_tanh)
        for m in self.metrics:
            m.update(out_seg1[:, 0, ...], y)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            [p for p in self.parameters()] + [self.log_var_a] + [self.log_var_b], lr=1e-3)
        return optimizer

    def configure_metrics(self):
        return [DiceCoefficient(), JaccardIndex(), AverageSurfaceDistance(), HausdorffDistance95()]

if __name__ == '__main__':
    model = VNet(n_channels=1, n_classes=2)
