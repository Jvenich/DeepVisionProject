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


    def __init__(self, num_epoch, batch_size, lr_init, milestones, model, modelname, device='cpu',
                 sigma=1, weight_decay=1e-6, use_genre=True, subset=False, likelihood=False, classification=False, zero_pad=None, conditional=False):
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
               """
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.modelname = modelname
        self.device = device
        self.sigma = sigma
        self.use_genre = use_genre
        self.subset = subset
        self.likelihood = likelihood
        self.classification = classification
        self.zero_pad = zero_pad
        self.conditional = conditional

        self.model = model.to(self.device)
        self.init_param()

        self.model_params = []
        for parameter in self.model.parameters():
            if parameter.requires_grad:
                self.model_params.append(parameter)

        self.optimizer = torch.optim.Adam(self.model_params, lr=lr_init, betas=(0.8, 0.8), eps=1e-04,
                                          weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

        print("Device used for further computation is:", self.device)


    def get_dataset(self, dataset, pin_memory=True, drop_last=True):
        """
        Init train-, testset and train-, testloader for experiment. Furthermore criterion will be initialized.

        :param dataset: string that describe which dataset to use for training. Current Options: "mnist", "cifar"
        :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them
        :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
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
            self.dataset, self.classes = dl.load_artset(self.use_genre, self.subset)
            self.trainloader, self.testloader = dl.split_dataset(self.dataset, 0.2, self.batch_size, pin_memory,
                                                                 drop_last)
        else:
            print("The requested dataset is not implemented yet.")

        img, _ = next(iter(self.trainloader))
        img = img.to(self.device)
        lat_img = self.model(img)
        self.lat_shape = lat_img.shape

        self.num_classes = len(self.classes)
        self.criterion = il.INN_loss(self.num_classes, self.sigma, self.device, self.batch_size, self.likelihood, self.classification, self.zero_pad, self.conditional)


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
                outputs = outputs.view(outputs.size(0), -1)
                _, predicted = torch.max(outputs[:, :self.num_classes], 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total


    def update_criterion(self, sigma):
        """
        Update scaling factors for inn_loss.

        :return: None
        """
        self.criterion = il.INN_loss(self.num_classes, sigma, self.device)



    def train(self):
        """
        Train INN model.
        """

        self.loss_log = []


        for epoch in range(self.num_epoch):
            self.scheduler.step()
            self.model.train()

            loss = 0

            print()
            print(80 * '-')
            print()

            print("Epoch: {}".format(epoch + 1))
            print("Training:")

            if self.device != torch.device("cuda"):
                print("Warning: GPU is not used for this computation")
                print("device:", self.device)

            for i, data in enumerate(tqdm(self.trainloader), 0):
                img, labels = data
                img, labels = img.to(self.device), labels.to(self.device)
                self.model = self.model.to(self.device)

                self.optimizer.zero_grad()

                lat_img = self.model(img)
                lat_img = lat_img.view(lat_img.size(0), -1)
                batch_loss = self.criterion(img, lat_img, labels, self.model)
                batch_loss.backward()
                self.optimizer.step()

                loss += batch_loss


            print("Loss: {:.3f}".format(loss))
            self.loss_log.append(loss)
            
            fm.save_model(self.model, '{}_{}'.format(self.modelname, epoch))
            fm.save_weight(self.model, '{}_{}'.format(self.modelname, epoch))

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


    def load_model(self, epoch=None):
        """
        Load pre-trained model based on modelname.

        :return: None
        """
        if epoch is None:
            self.model = fm.load_model('{}'.format(self.modelname))
        else:
            self.model = fm.load_model('{}_{}'.format(self.modelname, epoch))


    def load_weights(self, epoch=None):
        """
        Load pre-trained weights based on modelname.

        :return: None
        """
        if epoch is None:
            self.model = fm.load_weight(self.model, '{}'.format(self.modelname))
        else:
            self.model = fm.load_weight(self.model, '{}_{}'.format(self.modelname, epoch))



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
        :return: None
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
        :return: None
        """

        self.load_variables()

        pl.plot([x for x in range(1, self.num_epoch+1)], self.loss_log, 'Epoch', 'Loss',
                ['loss'], "Train Loss History {}".format(self.modelname),
                "train_loss_{}".format(self.modelname), sub_dim, figsize, font_size, y_log_scale)


    def generate(self, num_img=100, row_size=10, figsize=(30, 30), target=0):
        """
        Generate images based on given label. Only works after INN model was trained on classification.

        :param num_img: number of images to generate
        :param row_size: number of images to show in each row
        :param figsize: the size of the generated plot
        :return: None
        """

        self.load_model()

        img, _ = next(iter(self.trainloader))
        img = img.to(self.device)



        y = self.model(img)
        y = y.view(y.size(0), -1)


        gauss = torch.randn(y[:, self.num_classes:].shape).to(self.device)
        y = torch.cat([y[0].unsqueeze(0) for i in range(self.batch_size)])

        lat_img = torch.cat([y[:, :self.num_classes], gauss], dim=1).to(self.device)


        if self.zero_pad:
            if self.conditional:
                binary_label = lat_img.new_zeros(lat_img.size(0), self.num_classes)
                idx = torch.arange(self.batch_size, dtype=torch.long)
                binary_label[idx, target] = 1
                lat_img = torch.cat([lat_img[:, : self.num_classes + self.zero_pad], binary_label, torch.zeros(self.batch_size, 28 * 28 - self.num_classes - self.zero_pad - 10).cuda()], dim=1)
            else:
                lat_img = torch.cat([lat_img[:, : self.num_classes + self.zero_pad], torch.zeros(self.batch_size ,28 * 28 - self.num_classes - self.zero_pad).cuda()], dim=1)

                
        lat_img = lat_img.view(self.lat_shape)
        gen_img = self.model(lat_img, rev=True)

        print("Original Input Image:")
        pl.imshow(img[0][0].detach())

        print("Generated Images:")
        pl.imshow(torchvision.utils.make_grid(gen_img[:num_img].detach(), row_size), figsize,
                  self.modelname + "_generate")


    def metameric_sampling(self, num_img=100, row_size=10, figsize=(30, 30)):
        """
        Metameric sampling according to https://arxiv.org/pdf/1811.00401.pdf

        :param num_img: number of images to generate
        :param row_size: number of images to show in each row
        :param figsize: the size of the generated plot
        :return: None
        """

        self.load_model()

        img, _ = next(iter(self.trainloader))
        img = img.to(self.device)

        lat = self.model(img)
        lat = lat.view(lat.size(0), -1)

        y = torch.cat([lat[0].unsqueeze(0) for i in range(self.batch_size)])
        z = lat[:, self.num_classes:]

        lat_img = torch.cat([y[:, :self.num_classes], z], dim=1).to(self.device)
        lat_img = lat_img.view(self.lat_shape)
        gen_img = self.model(lat_img, rev=True)

        print("Original Input Image:")
        pl.imshow(img[0][0].detach())

        print("Images to sample z from:")
        pl.imshow(torchvision.utils.make_grid(img[:num_img].detach(), row_size), figsize)

        print("Generated Images from metameric sampling:")
        pl.imshow(torchvision.utils.make_grid(gen_img[:num_img].detach(), row_size), figsize,
                  self.modelname + "_metameric_sampling")



