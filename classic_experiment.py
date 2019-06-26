import dataloader as dl

class classic_experiment:
    """
    Class for training classical models.
    """

    def __init__(self, num_epoch, batch_size, lr_init, milesstones, get_model, modelname, device='cpu'):
        """
        Init class with pretraining setup.

        :param num_epoch: number of training epochs
        :param batch_size: Batch Size to use during training.
        :param lr_init: Starting learning rate. Will be decrease with adaptive learning.
        :param milesstones: list of training epochs at which the learning rate will be reduce
        :param get_model: function that returns a model for training
        :param modelname: model name under which the model will be saved
        :param device: device on which to do the computation (CPU or CUDA). Please use get_device() to get device
        variable, if using multiple GPU's. Default: cpu
        """
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lr_init = lr_init
        self.milestones = milesstones
        self.get_model = get_model
        self.modelname = modelname
        self.device = device

    def get_dataset(self, dataset, pin_memory=True, drop_last=True):
        if dataset == "mnist":
            self.trainset, self.testset, self.classes = dl.load_mnist()
            self.trainloader = dl.get_loader(self.trainset, self.batch_size, pin_memory, drop_last)
            self.testloader = dl.get_loader(self.testset, self.batch_size, pin_memory, drop_last)
        else:
            print("The requested dataset is not implemented yet.")
