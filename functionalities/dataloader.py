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


def load_artset():
    """
    Before this functions works, do the following steps:
        1) Download Painter by Numbers dataset from Kaggle: https://www.kaggle.com/c/painter-by-numbers/data
        2) Extract painter-by-numbers.zip
        3) Navigate to the extracted folder painter-by-numbers and extract train.zip and
        replacements_for_corrupted_files.zip
        4) Navigate to extracted folder replacements_for_corrupted_files/train and copy all images in there to the
        extracted train folder. Accept replacement of the copied images, when asked by your operating system.
        5) Copy train folder and train_info.csv to this project folder ../DeepVision Project/datasets/artset
        6) Navigate to DeepVision Project folder and run python prepare_artset.py
        7) Delete the original train folder and the train_info.csv file

    After the above steps have been performed. This function returns the dataset of Painter by Numbers containing the
    images and labels. Additionally, a dict mapping labels to classes will be returned.

    :return: dataset, classes
    """

    image_path = './datasets/artset/'

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    try:
        dataset = datasets.ImageFolder(image_path, transform)
    except:
        print(load_artset.__doc__)

    classes = dataset.classes

    return dataset, classes


def get_loader(dataset, batch_size, pin_memory=True, drop_last=True):
    """
    Create loader for a given dataset.

    :param dataset: dataset for which a loader will be created
    :param batch_size: size of the batch the loader will load during training
    :param pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them
    :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
    :return: loader
    """

    loader = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size, drop_last=drop_last)

    return loader


def split_dataset(dataset, ratio, batch_size, pin_memory=True, drop_last=True):
    """
    Split a dataset into two subset. e.g. trainset and validation-/testset
    :param dataset: dataset, which should be split
    :param ratio: the ratio the two splitted datasets should have to each other
    :param batch_size: batch size the returned dataloaders should have
    :param pin_memory: pin_memory argument for pytorch dataloader, will be simply forwarded
    :param drop_last: If true, drop the last incomplete batch, if the dataset is not divisible by the batch size
    :return: dataloader_1, dataloader_2
    """

    indices = torch.randperm(len(dataset))
    idx_1 = indices[:len(indices) - int(ratio * len(indices))]
    idx_2 = indices[len(indices) - int(ratio * len(indices)):]

    dataloader_1 = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_1),
                                               drop_last=drop_last)

    dataloader_2 = torch.utils.data.DataLoader(dataset, pin_memory=pin_memory, batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_2),
                                               drop_last=drop_last)

    return dataloader_1, dataloader_2