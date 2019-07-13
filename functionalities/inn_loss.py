import torch
from torch import nn
from functionalities import loss


class INN_loss(nn.Module):
    def __init__(self, num_classes, sigma, device):
        super(INN_loss, self).__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device


    def forward(self, lat_img, target, model):
        binary_label = lat_img.new_zeros(lat_img.size(0), self.num_classes)
        idx = torch.arange(target.size(0), dtype=torch.long)
        binary_label[idx, target] = 1
        l = loss.loss_max_likelihood(lat_img, torch.cat([binary_label,
                torch.randn(lat_img[:, self.num_classes:].shape).to(self.device)], dim=1), model, self.num_classes)

        return l