from torch import nn
from functionalities import loss


class INN_loss(nn.Module):
    def __init__(self, num_classes, a_class, a_noise, a_input, device):
        super(INN_loss, self).__init__()
        self.num_classes = num_classes
        self.a_class = a_class
        self.a_noise = a_noise
        self.a_input = a_input
        self.device = device

        self.CE_loss = nn.CrossEntropyLoss()


    def forward(self, z, v, z_, target):
        l_y = self.a_class * self.CE_loss(v[:, :self.num_classes], target)
        gauss = v.new_empty((v.size(0), v.size(1) - self.num_classes)).normal_()
        l_z = self.a_noise * loss.MMD_multiscale(v[:, self.num_classes:], gauss, self.device)
        l_x = self.a_input * loss.MMD_multiscale(z, z_)

        l = l_y + l_z + l_x

        return [l, l_y, l_z, l_x]