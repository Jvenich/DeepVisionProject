import torch
from torch import nn
from functionalities import loss


class INN_loss(nn.Module):
    def __init__(self, num_classes, a_y, a_z, a_x, a_rec, device):
        super(INN_loss, self).__init__()
        self.num_classes = num_classes
        self.a_y = a_y
        self.a_z = a_z
        self.a_x = a_x
        self.a_rec = a_rec
        self.device = device

        self.CE_loss = nn.CrossEntropyLoss()


    def forward(self, z, v, z_, target):
        l_y = self.a_y * self.CE_loss(v[:, :self.num_classes], target)
        gauss = v.new_empty((v.size(0), v.size(1) - self.num_classes)).normal_()
        #l_z = self.a_z * loss.MMD_multiscale(v[:, self.num_classes:], gauss, self.device)
        l_z = self.a_z * loss.l2_loss(v[:, self.num_classes:], torch.zeros(v[:, self.num_classes:].shape).to(self.device))
        #l_x = self.a_x * loss.MMD_multiscale(z.view(z.size(0), -1), z_.view(z_.size(0), -1), self.device)
        l_x =
        l_rec = self.a_rec * loss.l1_loss(z, z_)

        l = l_y + l_z + l_x + l_rec

        return [l, l_y, l_z, l_x, l_rec]