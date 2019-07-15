import torch
from torch import nn
from functionalities import dataloader as dl
from functionalities import filemanager as fm
from functionalities import plot as pl
from tqdm import tqdm_notebook as tqdm

class classic_experiment:
    """
    Class for training classical models
    """


    def __init__(self, num_epoch, batch_size, lr_init, milestones, model, modelname, device='cpu',
                 weight_decay=1e-5):
        """
        Init class with pretraining setup.

        :param num_epoch: number of training epochs
        :param batch_size: Batch Size to use during training.
        :param lr_init: Starting learning rate. Will be decrease with adaptive learning.
        :param milestones: list of training epochs at which the learning rate will be reduce
        :param get_model: model for training
        :param modelname: model name under which the model will be saved
        :param device: device on which to do the computation (CPU or CUDA). Please use get_device() to get device
        variable, if using multiple GPU's.
        :param weight_decay: weight decay (L2 penalty) for adam optimizer
        :return: None
        """
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.modelname = modelname
        self.device = device

        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_init, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        print("Device used for further computation is:", self.device)


    def get_dataset(self, dataset, pin_memory=True, drop_last=True, ratio=0.2):
        """
        Init train-, testset and train-, testloader for experiment.

        :param dataset: string that describe which dataset to use for training. Current Options: "mnist", "cifar"
        :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them
        :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
        :param ratio: Only necessary if dataset is "artset". Ratio of train- and testset by which Painter by Numbers
        dataset should be divided by.
        :return: None
        """
        if dataset == "mnist":
            self.trainset, self.testset, self.classes = dl.load_mnist()
            self.trainloader = dl.get_loader(self.trainset, self.batch_size, pin_memory, drop_last)
            self.testloader = dl.get_loader(self.testset, self.batch_size, pin_memory, drop_last)
        elif dataset == "cifar":
            self.trainset, self.testset, self.classes = dl.load_cifar()
            self.trainloader = dl.get_loader(self.trainset, self.batch_size, pin_memory, drop_last)
            self.testloader = dl.get_loader(self.testset, self.batch_size, pin_memory, drop_last)
        elif dataset == "artset":
            self.dataset, self.classes = dl.load_artset()
            self.trainloader, self.testloader = dl.split_dataset(self.dataset, ratio, self.batch_size, pin_memory,
                                                                 drop_last)
        else:
            print("The requested dataset is not implemented yet.")


    def get_accuracy(self, loader):
        """
        Evaluate accuracy of current model on given loader.

        :param loader: pytorch loader for a dataset
        :return: accuracy
        """
        if self.device != torch.device("cuda"):
            print("Warning: GPU is not used for this computation")
            print("device:", self.device)

        correct = 0
        total = 0
        with torch.no_grad():
            for data in tqdm(loader):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)
                self.model = self.model.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total


    def train(self):
        """
        Train classical model.
        """

        self.loss_log = []

        for epoch in range(self.num_epoch):
            self.scheduler.step()
            self.model.train()

            print()
            print(80 * '-')
            print()

            print("Epoch: {}".format(epoch + 1))
            print("Training:")

            if self.device != torch.device("cuda"):
                print("Warning: GPU is not used for this computation")
                print("device:", self.device)

            for data in tqdm(self.trainloader):
                img, labels = data
                img, labels = img.to(self.device), labels.to(self.device)
                self.model = self.model.to(self.device)
                output = self.model(img)
                loss = self.criterion(output, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print('loss: {:.3f}'.format(loss.data.item()))
            self.loss_log.append(loss)

        print()
        print(80 * "#")
        print(80 * "#")
        print()

        print("Evaluating:")
        self.model.eval()
        self.train_acc = self.get_accuracy(self.trainloader)
        self.test_acc = self.get_accuracy(self.testloader)

        print("Final Train Accuracy:", self.train_acc)
        print("Final Test Accuracy:", self.test_acc)

        fm.save_model(self.model, '{}'.format(self.modelname))
        fm.save_weight(self.model, '{}'.format(self.modelname))
        fm.save_variable([self.train_acc, self.test_acc], '{}_acc'.format(self.modelname))
        fm.save_variable([self.loss_log], '{}_loss'.format(self.modelname))


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


    def load_variables(self):
        """
        Load recorded loss and accuracy training history to class variable.

        :return: None
        """
        self.train_acc, self.test_acc = fm.load_variable('{}_acc'.format(self.modelname))
        self.loss_log = fm.load_variable('{}_loss'.format(self.modelname))


    def print_accuracy(self):
        """
        Plot train and test accuracy during training.

        :param sub_dim: dimensions of subplots. Only required, if the dimension of both x and y are 2.
        :param figsize: the size of the generated plot
        :param font_size: font size of labels
        :param y_log_scale: y axis will have log scale instead of linear
        """
        self.load_variables()

        print("Final Train Accuracy:", self.train_acc)
        print("Final Test Accuracy:", self.test_acc)


    def plot_loss(self, sub_dim=None, figsize=(15, 10), font_size=24, y_log_scale=False):
        """
        Plot train and test loss during training.

        :param sub_dim: dimensions of subplots. Only required, if the dimension of both x and y are 2.
        :param figsize: the size of the generated plot
        :param font_size: font size of labels
        :param y_log_scale: y axis will have log scale instead of linear
        """
        self.load_variables()

        pl.plot([x for x in range(1, self.num_epoch+1)], self.loss_log, 'Epoch', 'Loss',
                ['train', 'test'], "Train and Test Loss History {}".format(self.modelname),
                "train_test_loss_{}".format(self.modelname), sub_dim, figsize, font_size, y_log_scale)
