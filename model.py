import torch
import torch.nn as nn
import torch.nn.functional as F



def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    """
    batch = kernel.size(0)
    channel = kernel.size(1)
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    out = out.view(batch, channel, out.size(2), out.size(3))
    return out

class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3, hidden_kernel_size=5):
        super(DepthwiseXCorr, self).__init__()
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)
        search = self.conv_search(search)
        feature = xcorr_depthwise(search, kernel)
        out = self.head(feature)
        return out

class AlexNet(nn.Module):
    def __init__(self):
        configs = [3, 96, 256, 384, 384, 256]
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(configs[0], configs[1], kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer2 = nn.Sequential(
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            )
        self.layer3 = nn.Sequential(
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            )
        self.layer4 = nn.Sequential(
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
            )
        self.feature_size = configs[5]  # 256

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

class RpnHead(nn.Module):

    def __init__(self, anchor_num):
        super(RpnHead, self).__init__()
        self.K = anchor_num

        self.cls_z = nn.Conv2d(256, 256 * 2 * self.K, kernel_size=3, stride=1, padding=0)
        self.reg_z = nn.Conv2d(256, 256 * 4 * self.K, kernel_size=3, stride=1, padding=0)

        self.cls_x = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.reg_x = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)

    def forward(self, z_ft, x_ft):
        N = z_ft.shape[0]

        cls_z = self.cls_z(z_ft)  # [N, 2K*256, 4, 4]
        reg_z = self.reg_z(z_ft)  # [N, 4K*256, 4, 4]

        cls_x = self.cls_x(x_ft)  # [N, 256, 20, 20]
        reg_x = self.reg_x(x_ft)  # [N, 256, 20, 20]

        # cross-correlation
        cls_z = cls_z.view(-1, 256, 4, 4)  # [N*2K, 256, 4, 4]
        cls_x = cls_x.view(1, -1, 20, 20)  # [1, N*256, 20, 20]
        pred_cls = F.conv2d(cls_x, cls_z, groups=N)  # [1, N*2K, 17, 17]
        pred_cls = pred_cls.view(N, -1, pred_cls.shape[2], pred_cls.shape[3])  # [N, 2K, 17, 17]

        reg_z = reg_z.view(-1, 256, 4, 4)  # [N*4K, 256, 4, 4]
        reg_x = reg_x.view(1, -1, 20, 20)  # [1, N*256, 20, 20]
        pred_reg = F.conv2d(reg_x, reg_z, groups=N)  # [1, N*4K, 17, 17]
        pred_reg = pred_reg.view(N, -1, pred_reg.shape[2], pred_reg.shape[3])  # [N, 4K, 17, 17]

        # return {'pred_cls': pred_cls, 'pred_reg': pred_reg}
        return pred_cls, pred_reg

class RpnHeadPysot(nn.Module):
    def __init__(self, anchor_num=5, in_channels=256, out_channels=256):
        super(RpnHeadPysot, self).__init__()
        self.cls = DepthwiseXCorr(in_channels, out_channels, 2 * anchor_num)
        self.reg = DepthwiseXCorr(in_channels, out_channels, 4 * anchor_num)

    def forward(self, z_f, x_f):
        cls = self.cls(z_f, x_f)  # (N, 2K, 17, 17)
        reg = self.reg(z_f, x_f)  # (N, 4K, 17, 17)
        return cls, reg


class SiamRPN(nn.Module):
    def __init__(self, anchor_num):
        super(SiamRPN, self).__init__()
        self.K = anchor_num
        self.backbone = AlexNet()
        self.rpnhead = RpnHead(self.K)
        # self.rpnhead = RpnHeadPysot(anchor_num=self.K)

    def forward(self, z, x):
        z_ft = self.backbone(z)
        x_ft = self.backbone(x)

        pred_dict = self.rpnhead(z_ft, x_ft)
        return pred_dict

    def template(self, z):
        self.z_ft = self.backbone(z)

    def track(self, x):
        """
        Args:
            x: cropped x patch of
               subsequent frame
        Returns:
            predict bbox:[x, y, w, h]
        """
        x_ft = self.backbone(x)
        cls, reg = self.rpnhead(self.z_ft, x_ft)

        return {
                'cls': cls,
                'reg': reg
               }


if __name__ == '__main__':
    # z = torch.randn(16, 3, 127, 127).to('cuda')
    # x = torch.randn(16, 3, 255, 255).to('cuda')
    siamrpn = SiamRPN(anchor_num=5)
    siamrpn.cuda()
    from config.config import cfg
    trainable_params = []
    trainable_params += [{'params': filter(lambda x: x.requires_grad,
                                           siamrpn.backbone.parameters()),
                          'lr': cfg.BACKBONE.LAYERS_LR * cfg.TRAIN.BASE_LR}]

    trainable_params += [{'params': siamrpn.rpnhead.parameters(),
                          'lr': cfg.TRAIN.BASE_LR}]

    optimizer = torch.optim.SGD(trainable_params,
                                momentum=cfg.TRAIN.MOMENTUM,
                                weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    from lr_scheduler import  LogScheduler

    scheduler = LogScheduler(optimizer,
                             start_lr=cfg.TRAIN.LR.START,
                             end_lr=cfg.TRAIN.LR.END,
                             epochs=cfg.TRAIN.EPOCH)
    lr = []
    for i in range(1, 51):
        lr.append(optimizer.param_groups[0]['lr'])
        optimizer.step()
        scheduler.step()
    print(lr)
    # for param_group in optimizer.param_groups:
    #     print(param_group['lr'])
    # print(type(optimizer.param_groups))
    # import cv2
    # from torchsummary import summary
    # tic = cv2.getTickCount()
    # pred = siamrpn(z, x)
    # time = (cv2.getTickCount() - tic) / cv2.getTickFrequency()
    # print('time is: ', time)  # 0.005782215 / 0.006382104

    # print("pred_cls shape: ", pred['pred_cls'].shape)  # torch.Size([1, 10, 17, 17])
    # print("pred_reg shape: ", pred['pred_reg'].shape)  # torch.Size([1, 20, 17, 17])

    # summary(siamrpn, input_size=[(3, 127, 127), (3, 255, 255)])  # Estimated Total Size (MB): 36164.68 / 36096.29

