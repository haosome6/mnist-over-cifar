from turtle import forward
import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, input_dim):
        super(Generator, self).__init__()
        self.input_dim = input_dim

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 384)
        )

        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_channels=384,out_channels=192,kernel_size=(4,4),stride=1,padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=192,out_channels=96,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=96,out_channels=48,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=48,out_channels=3,kernel_size=(4,4),stride=2,padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = x.view(-1, self.input_dim)
        x = self.fc(x)
        x = x.view(-1,384,1,1)
        x = self.net(x)
        return x

class Discriminator(nn.Module):

    def __init__(self, num_class_label=10):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=(3,3),stride=1,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )

        self.net_source = nn.Sequential(
            nn.Linear(3*3*512, 1),
            nn.Sigmoid()
        )

        self.net_class = nn.Sequential(
            nn.Linear(3*3*512, num_class_label),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 3*3*512)
        source = self.net_source(x)
        class_label = self.net_class(x)

        return source, class_label