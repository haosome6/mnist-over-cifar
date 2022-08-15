import os
import torch
import torchvision

def get_mnist_images():
    mnist_train = torchvision.datasets.MNIST('./../../../data/',train=True,download=True)


if __name__ == "__main__":
    get_mnist_images()