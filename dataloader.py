import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms



def load_mnist():
    """
    Check if the MNIST dataset already exists in the directory "./datasets/mnist". If not, the MNIST dataset is
    downloaded. Returns trainset, testset and classes of MNIST.

    :return: trainset, testset, classes of MNIST
    """

    save_path = "./datasets/mnist"

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.MNIST(root=save_path, train=True, transform=transform, download=True)
    testset = datasets.MNIST(root=save_path, train=False, transform=transform, download=True)

    classes = [x for x in range(10)]

    return trainset, testset, classes


def load_cifar():
    """
    Check if the CIFAR10 dataset already exists in the directory "./datasets/cifar". If not, the CIFAR10 dataset is
    downloaded. Returns trainset, testset and classes of CIFAR10.

    :return: trainset, testset, classes of CIFAR10
    """

    save_path = "./datasets/cifar"

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.CIFAR10(root=save_path, train=True, transform=transform, download=True)
    testset = datasets.CIFAR10(root=save_path, train=False, transform=transform, download=True)

    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    return trainset, testset, classes


def get_loader(dataset, batch_size, pin_memory=True, drop_last=True):
    """
    Create loader for a given dataset.

    :param dataset: dataset for which a loader will be created
    :param batch_size: size of the batch the loader will load during training
    :param pin_memory: pin_memory argument for pytorch dataloader, will be simply forwarded
    :return: loader
    """

    loader = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size, drop_last=drop_last)

    return loader