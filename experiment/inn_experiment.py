import torch
from torch import nn


class inn_experiment:
    """
    Class for training INN models
    """


    def __init__(self, num_epoch, batch_size, lr_init, milesstones, get_model, modelname, device='cpu',
                 weight_decay=1e-6):
        """
        Init class with pretraining setup.

        :param num_epoch: number of training epochs
        :param batch_size: Batch Size to use during training.
        :param lr_init: Starting learning rate. Will be decrease with adaptive learning.
        :param milesstones: list of training epochs at which the learning rate will be reduce
        :param get_model: function that returns a model for training
        :param modelname: model name under which the model will be saved
        :param device: device on which to do the computation (CPU or CUDA). Please use get_device() to get device
        variable, if using multiple GPU's.
        :param weight_decay: weight decay (L2 penalty) for adam optimizer
               """
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.modelname = modelname
        self.device = device

        self.model = get_model().to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_init, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milesstones, gamma=0.1)

        optimizer = torch.optim.Adam(model_params, lr=lr_init, betas=(0.8, 0.8), eps=1e-04, weight_decay=l2_reg)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


    def init_param(self, sigma=0.1):
        for key, param in self.model.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = sigma * torch.randn(param.data.shape).cuda()
                if split[3][-1] == '3':  # last convolution in the coeff func
                    param.data.fill_(0.)