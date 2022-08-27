import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
from scipy.io import loadmat
import pickle
import numpy as np


class MNISTOverCifar(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        # self.mnist = datasets.MNIST(root, train=train, download=download)
        # self.cifar = datasets.CIFAR10(root, train=train, download=download)

        mnist = loadmat('../../data/original_data/mnist-original.mat')
        self.mnist_data, self.mnist_label = mnist['data'].T, mnist['label'][0]

        self.cifar_data = None
        if train:
            # mnist
            self.mnist_data, self.mnist_label = self.mnist_data[:50000,:], self.mnist_label[:50000]
            # cifar
            for i in range(1, 6):
                with open('../../data/original_data/cifar-10-batches-py/data_batch_{}'.format(i), 'rb') as f:
                    dict = pickle.load(f, encoding='bytes')
                if self.cifar_data is None:
                    self.cifar_data = dict[b'data']
                else:
                    self.cifar_data = np.vstack((self.cifar_data, dict[b'data']))
        else:
            # mnist
            self.mnist_data, self.mnist_label = self.mnist_data[50000:60000,:], self.mnist_label[50000:60000]
            # cifar
            with open('../../data/original_data/cifar-10-batches-py/test_batch', 'rb') as f:
                dict = pickle.load(f, encoding='bytes')
            self.cifar_data = dict[b'data']

    def __len__(self):
        return min(len(self.mnist), len(self.cifar))

    def __getitem__(self, index):
        mnist_img, mnist_label = self.mnist[index]
        cifar_img, cifar_label = self.cifar[index]

        mnist_img, cifar_img = transforms.PILToTensor()(mnist_img), transforms.PILToTensor()(cifar_img)
        mnist_label, cifar_label = torch.tensor(mnist_label), torch.tensor(cifar_label)

        # transform mnist image to the same size as cifar image
        # mnist_img = transforms.Resize((32, 32), transforms.InterpolationMode.NEAREST)(mnist_img)
        edge_padding = torch.nn.ZeroPad2d(2)
        mnist_img = edge_padding(mnist_img)
        mnist_img = transforms.Lambda(lambda x: x.repeat(3, 1, 1))(mnist_img)

        img = torch.where(mnist_img != 0, mnist_img, cifar_img)
        label = mnist_label

        img = transforms.ToPILImage()(img)
        if self.transform:
            img = self.transform(img)
        
        # sample = {'image': img, 'label': label}
        return img, label
