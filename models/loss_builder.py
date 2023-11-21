import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from util.img_generate import *

topil = transforms.ToPILImage()


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class WMF_Loss(nn.Module):
    def __init__(self):
        super(WMF_Loss, self).__init__()
        self.ssim = SSIM().cuda()

    def l1_loss(self, input, target):
        # loss = F.l1_loss(input=input, target=target, reduction='mean')
        abs_diff = torch.abs(input - target)
        # loss = abs_diff.mean(1, keepdim=True)
        loss = abs_diff.mean()
        return loss

    def wmf(self, input, target):


    def build_loss(self, args, inputs, outputs):
        losses = {}

        # L1 Loss
        if args.use_l1:
            l1_l = self.l1_loss(outputs['pred_disp'], inputs['gt_disp'])
            # For additional loss function in the future
            losses['l1'] = l1_l * args.w_l1

        # Total Loss
        losses['total_loss'] = sum(losses.values())

        return losses
