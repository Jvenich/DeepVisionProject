import torch
from torch import nn
from functionalities import loss


class INN_loss(nn.Module):
    def __init__(self, num_classes, sigma, device, batch_size, likelihood, classification):
        super(INN_loss, self).__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        self.device = device
        self.batch_size = batch_size
        self.likelihood = likelihood
        self.classification = classification

        self.CE = nn.CrossEntropyLoss()


    def forward(self, img, lat_img, target, model):
        binary_label = lat_img.new_zeros(lat_img.size(0), self.num_classes)
        idx = torch.arange(target.size(0), dtype=torch.long)
        binary_label[idx, target] = 1

        if self.likelihood:
            l = loss.loss_max_likelihood(lat_img, torch.cat([binary_label,
                torch.randn(lat_img[:, self.num_classes:].shape).to(self.device)], dim=1), model, self.num_classes, self.sigma)
        elif self.classification:
            l = self.CE(lat_img[:, :self.num_classes], target)
        else:
            cat_inputs = [lat_img[:, :self.num_classes] + 5e-2 * loss.noise_batch(self.batch_size, self.num_classes, self.device)]
            cat_inputs.append(lat_img[:, self.num_classes:] + 2e-2 * loss.noise_batch(self.batch_size, 28*28 - self.num_classes, self.device))

            cat_inputs = torch.cat(cat_inputs, dim=1)
            cat_inputs = torch.cat([cat_inputs[:, :self.num_classes + 32], binary_label, torch.zeros(self.batch_size, 28*28 - self.num_classes - 32 - 10).cuda()], dim=1)

            x_reconstructed = model(cat_inputs, rev=True)


            y = torch.cat((binary_label, loss.noise_batch(self.batch_size, 28*28-self.num_classes, self.device)), dim=1)
            y_0 = torch.cat([y[:, :self.num_classes + 32], binary_label, torch.zeros(self.batch_size, 28*28 - self.num_classes - 32 - 10).cuda()], dim=1)


            x_samples = model(y_0, rev=True)


            l = ( loss.l2_loss(x_reconstructed, img)
                  + loss.MMD_multiscale(img.view(img.size(0), -1), x_samples.view(x_samples.size(0), -1), self.device)
                  + loss.l2_loss(lat_img[:, :self.num_classes], y[:, :self.num_classes])
                  + loss.MMD_multiscale(lat_img, y, self.device) )

        return l