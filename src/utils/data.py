import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from .custom_dataset import MNISTOverCifar

def get_mnist_loader(batch_size):
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, '../../data')

    transform = transforms.Compose([transforms.ToTensor(),
                                    # transforms.Normalize((0.1307,), (0.3081,)),
                                    # transforms.Grayscale(3),
                                    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                    transforms.Resize((32, 32),transforms.InterpolationMode.BILINEAR)])

    mnist_train_loader = DataLoader(
        datasets.MNIST(data_path,
                       train=True,
                       download=True,
                       transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    mnist_test_loader = DataLoader(
        datasets.MNIST(data_path,
                       train=False,
                       download=True,
                       transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    return mnist_train_loader, mnist_test_loader

def get_cifar_loader(batch_size):
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, '../../data')

    transform =transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar_train_loader = DataLoader(
        datasets.CIFAR10(data_path,
                         train=True,
                         download=True,
                         transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    cifar_test_loader = DataLoader(
        datasets.CIFAR10(data_path,
                         train=False,
                         download=True,
                         transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    
    return cifar_train_loader, cifar_test_loader

def get_mnist_over_cifar_loader(batch_size):
    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, '../../data')

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_over_cifar_train_loader = DataLoader(
        MNISTOverCifar(data_path,
                       train=True,
                       download=True,
                       transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    
    mnist_over_cifar_test_loader = DataLoader(
        MNISTOverCifar(data_path,
                       train=False,
                       download=True,
                       transform=transform),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)

    return mnist_over_cifar_train_loader, mnist_over_cifar_test_loader