import numpy as np
import torch
import torchvision
from functionalities import inn_loss as il
from functionalities import dataloader as dl
from functionalities import filemanager as fm
from functionalities import plot as pl
from tqdm import tqdm_notebook as tqdm


class inn_experiment:
    """
    Class for training INN models
    """


    def __init__(self, num_epoch, batch_size, lr_init, milestones, get_model, modelname, device='cpu',
                 weight_decay=1e-6, a_class=1, a_noise=1, a_input=1, a_rec=1):
        """
        Init class with pretraining setup.

        :param num_epoch: number of training epochs
        :param batch_size: Batch Size to use during training.
        :param lr_init: Starting learning rate. Will be decrease with adaptive learning.
        :param milestones: list of training epochs at which the learning rate will be reduce
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
        self.a_class = a_class
        self.a_noise = a_noise
        self.a_input = a_input
        self.a_rec = a_rec

        self.model = get_model().to(self.device)
        self.init_param()

        self.model_params = []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                self.model_params.append(parameter)

        self.optimizer = torch.optim.Adam(self.model_params, lr=lr_init, betas=(0.8, 0.8), eps=1e-04,
                                          weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)


    def get_dataset(self, dataset, pin_memory=True, drop_last=True):
        """
        Init train-, testset and train-, testloader for experiment. Furthermore criterion will be initialized.

        :param dataset: string that describe which dataset to use for training. Current Options: "mnist", "cifar"
        :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them
        :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
        """
        if dataset == "mnist":
            self.trainset, self.testset, self.classes = dl.load_mnist()
            self.num_classes = len(self.classes)
            self.criterion = il.INN_loss(self.num_classes, self.a_class, self.a_noise, self.a_input, self.a_rec,
                                         self.device)
            self.trainloader = dl.get_loader(self.trainset, self.batch_size, pin_memory, drop_last)
            self.testloader = dl.get_loader(self.testset, self.batch_size, pin_memory, drop_last)
        elif dataset == "cifar":
            self.trainset, self.testset, self.classes = dl.load_cifar()
            self.num_classes = len(self.classes)
            self.criterion = il.INN_loss(self.num_classes, self.a_class, self.a_noise, self.a_input, self.a_rec,
                                         self.device)
            self.trainloader = dl.get_loader(self.trainset, self.batch_size, pin_memory, drop_last)
            self.testloader = dl.get_loader(self.testset, self.batch_size, pin_memory, drop_last)
        else:
            print("The requested dataset is not implemented yet.")


    def get_accuracy(self, loader):
        """
        Evaluate accuracy of current model on given loader.

        :param loader: pytorch loader for a dataset
        :return: accuracy
        """
        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(loader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                outputs = outputs.view(outputs.size(0), -1)
                _, predicted = torch.max(outputs[:, :self.num_classes], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total


    def train(self):
        """
        Train INN model.
        """

        self.train_acc_log = []
        self.test_acc_log = []
        self.tot_loss_log = []
        self.lx_loss_log = []
        self.ly_loss_log = []
        self.lz_loss_log = []
        self.lrec_loss_log = []

        for epoch in range(self.num_epoch):
            self.scheduler.step()
            self.model.train()

            losses = np.zeros(5, dtype=np.double)

            print("Epoch: {}".format(epoch + 1))
            print("Training:")

            for i, data in enumerate(tqdm(self.trainloader), 0):
                img, labels = data
                img, labels = img.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                lat_img = self.model(img)
                self.lat_shape = lat_img.shape
                self.lat_img = lat_img.view(lat_img.size(0), -1)
                binary_label = lat_img.new_zeros(lat_img.size(0), self.num_classes)
                idx = torch.arange(labels.size(0), dtype=torch.long)
                binary_label[idx, labels] = 1
                lat_img_mod = torch.cat([binary_label, lat_img[:, self.num_classes:]], dim=1)
                lat_img_mod = lat_img_mod.view(self.lat_shape)
                output = self.model(lat_img_mod, rev=True)
                batch_loss = self.criterion(img, lat_img, output, labels)
                batch_loss[0].backward()
                self.optimizer.step()

                for i in range(len(batch_loss)):
                    losses[i] += batch_loss[i].item()

            print("Evaluating:")
            self.model.eval()

            train_acc = self.get_accuracy(self.trainloader)
            test_acc = self.get_accuracy(self.testloader)

            print("Loss: {:.3f} \t L_y: {:.3f} \t L_z: {:.3f} \t L_x: {:.3f} \t L_rec: {:.3f}".format(
                losses[0], losses[1], losses[2], losses[3], losses[4]))
            print("Train_acc: {:.3f} \t Test_acc: {:.3f}".format(train_acc, test_acc))

            self.train_acc_log.append(train_acc)
            self.test_acc_log.append(test_acc)
            self.tot_loss_log.append(losses[0])
            self.ly_loss_log.append(losses[1])
            self.lz_loss_log.append(losses[2])
            self.lx_loss_log.append(losses[3])
            self.lrec_loss_log.append(losses[4])

        print(80 * "-")
        print("Final Test Accuracy:", self.test_acc_log[-1])

        fm.save_model(self.model, '{}'.format(self.modelname))
        fm.save_weight(self.model, '{}'.format(self.modelname))
        fm.save_variable([self.train_acc_log, self.test_acc_log], '{}_acc'.format(self.modelname))
        fm.save_variable([self.tot_loss_log, self.ly_loss_log, self.lz_loss_log, self.lx_loss_log, self.lrec_loss_log],
                         '{}_loss'.format(self.modelname))


    def init_param(self, sigma=0.1):
        """
        Initialize weights for INN models.

        :param sigma: standard deviation for gaussian
        :return: None
        """
        for key, param in self.model.named_parameters():
            split = key.split('.')
            if param.requires_grad:
                param.data = sigma * torch.randn(param.data.shape).cuda()
                if split[3][-1] == '3':  # last convolution in the coeff func
                    param.data.fill_(0.)


    def load_model(self):
        """
        Load pre-trained model based on modelname.

        :return: None
        """
        self.model = fm.load_model('{}'.format(self.modelname))


    def load_weights(self):
        """
        Load pre-trained weights based on modelname.

        :return: None
        """
        self.model = fm.load_weight(self.model, '{}'.format(self.modelname))


    def plot_accuracy(self, sub_dim=None, figsize=(15, 10), font_size=24, y_log_scale=False):
        """
        Plot train and test accuracy during training.

        :param sub_dim: dimensions of subplots. Only required, if the dimension of both x and y are 2.
        :param figsize: the size of the generated plot
        :param font_size: font size of labels
        :param y_log_scale: y axis will have log scale instead of linear
        :return: None
        """

        pl.plot([x for x in range(1, self.num_epoch+1)], [self.train_acc_log, self.test_acc_log], 'Epoch', 'Accuracy',
                ['train', 'test'], "Train and Test Accuracy History {}".format(self.modelname),
                "train_test_acc_{}".format(self.modelname), sub_dim, figsize, font_size, y_log_scale)


    def plot_loss(self, sub_dim=None, figsize=(15, 10), font_size=24, y_log_scale=False):
        """
        Plot train and test loss during training.

        :param sub_dim: dimensions of subplots. Only required, if the dimension of both x and y are 2.
        :param figsize: the size of the generated plot
        :param font_size: font size of labels
        :param y_log_scale: y axis will have log scale instead of linear
        :return: None
        """

        pl.plot([x for x in range(1, self.num_epoch+1)], [self.tot_loss_log, self.ly_loss_log, self.lz_loss_log,
                self.lx_loss_log, self.lrec_loss_log], 'Epoch', 'Loss',
                ['total', 'ly', 'lz', 'lx', 'l_rec'], "Train Loss History {}".format(self.modelname),
                "train_loss_{}".format(self.modelname), sub_dim, figsize, font_size, y_log_scale)


    def generate(self, label, num_img=100, row_size=10, figsize=(30, 30)):
        """
        Generate images based on given label. Only works after INN model was trained on classification.

        :param label: label of class to generate images from
        :param num_img: number of images to generate
        :param row_size: number of images to show in each row
        :param figsize: the size of the generated plot
        :return: None
        """

        self.load_model()

        binary_label = torch.zeros(self.batch_size, self.num_classes)
        idx = torch.arange(self.batch_size)
        binary_label[idx, label] = 1

        gauss = torch.zeros(self.batch_size, self.lat_img.shape[1] - self.num_classes).normal()

        lat_img = torch.cat([binary_label, gauss], dim=1).view(self.lat_shape)
        gen_img = self.model(lat_img, rev=True)

        pl.imshow(torchvision.utils.make_grid(gen_img[:num_img].detach(), row_size), figsize,
                  self.modelname + "generate_{}".format(self.classes[label]))




