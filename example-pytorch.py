import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    return device

device = get_device()

#Read data
df_train = pd.read_csv("data/msnist/mnist_train.csv")
df_test = pd.read_csv("data/msnist/mnist_test.csv")

print(df_test.head())
#Get the image pixel values and labels
train_labels = df_train.iloc[:, 0]
train_images = df_train.iloc[:, 1:]
test_labels = df_test.iloc[:, 0]
test_images = df_test.iloc[:, 1:]

# Define transforms
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)

class MNISTDataset(Dataset):
    def __init__(self, images, labels= None, transforms= None):
        super().__init__()
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X.iloc[i, :]
        data = np.asarray(data).astype(np.int8).reshape(28,28,1)

        if self.transforms:
            data = self.transforms(data)
        
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

train_data = MNISTDataset(train_images, train_labels, transform)
test_data = MNISTDataset(test_images, test_labels, transform)

