import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class MNISTOverCifar(Dataset):
    def __init__(self, root, train=True, download=True, transform=None):
        self.root = root
        self.transform = transform
        self.mnist = datasets.MNIST(root, train=train, download=download)
        self.cifar = datasets.CIFAR10(root, train=train, download=download)

    def __len__(self):
        return min(len(self.mnist), len(self.cifar))

    def __getitem__(self, index):
        mnist_img, mnist_label = self.mnist[index]
        cifar_img, cifar_label = self.cifar[index]

        mnist_img, cifar_img = transforms.PILToTensor()(mnist_img), transforms.PILToTensor()(cifar_img)
        mnist_label, cifar_label = torch.tensor(mnist_label), torch.tensor(cifar_label)

        # transform mnist image to the same size as cifar image
        mnist_img = transforms.Resize((32, 32), transforms.InterpolationMode.BILINEAR)(mnist_img)
        mnist_img = transforms.Lambda(lambda x: x.repeat(3, 1, 1))(mnist_img)

        img = torch.where(mnist_img != 0, mnist_img, cifar_img)
        label = mnist_label

        if self.transform:
            img = self.transform(img)
        
        sample = {'image': img, 'label': label}
        return sample
