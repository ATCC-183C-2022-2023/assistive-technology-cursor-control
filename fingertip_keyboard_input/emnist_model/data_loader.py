import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
from pytorch_model_class import DEVICE

class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.3):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

def load_emnist_data(train_noise = False, validation_noise = False):
    train_transform = transforms.ToTensor()
    if train_noise:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise()
        ])

    train_data = dsets.EMNIST(root='./data', train=True, download=True, transform=train_transform, split = 'letters')

    validation_transform = transforms.ToTensor()
    if validation_noise:
        validation_transform = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise()
        ])

    validation_data = dsets.EMNIST(root='./data', train=False, download=True, transform=validation_transform, split = 'letters')
    # plt.imshow(validation_data[0][0][0])
    # plt.show()

    return train_data, validation_data

def create_data_loader(train_data, validation_data, train_batch_size = 2000, validation_batch_size = 2000):
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=train_batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_data, batch_size=validation_batch_size, shuffle=True)

    return train_loader, validation_loader


load_emnist_data(validation_noise = True)