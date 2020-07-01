import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from matplotlib import image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch as T 
import torch.nn as nn
import torch.optim as optim

import torchvision  
import torchvision.models as models
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, transforms


#Constants
IMAGES_DIRECTORY = os.path.join("data", "all_images")
device = "cpu"

#Functions
def get_train_data(): 
    path = os.path.join("data", "train.csv")
    return pd.read_csv(path)

def get_test_data():
    path = os.path.join("data", "test.csv")
    return pd.read_csv(path)

def get_images_array(data_set):
    images = [] 

    for filename in data_set['image_names']:
        img_path = os.path.join(IMAGES_DIRECTORY,filename)

        img = image.imread(img_path)
        
        images.append(img)

    return images


#Implementation
train_data = get_train_data()
test_data = get_test_data()
train_images = get_images_array(train_data)
test_images = get_images_array(test_data)

#Load train and validation data
train_X = train_images
train_y = train_data['emergency_or_not']

test_X = test_images

print("Shape of each train image:", train_X[0].shape)
print("Shape of each test image:", test_X[0].shape)

def show_sample_item(dataset_array, item_idx = 0): #random default number
    sample = dataset_array[item_idx]
    img = Image.fromarray(sample)
    img.show()

# show_sample_item(train_X, item_idx = 1389)
# show_sample_item(test_X) 

def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    return device

device = get_device()
print("Device:", device)

#Get the image pixel values and labels
train_images = train_X
train_labels = train_y

# Define transforms
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
)


class VehicleDataset(Dataset):
    def __init__(self, images, labels= None, transforms= None):
        super().__init__()
        self.X = images
        self.y = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        data = self.X[i]
        data = data.astype(np.uint8).reshape(224,224,3)

        if self.transforms:
            data = self.transforms(data)
        
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

train_data = VehicleDataset(train_images, train_labels, transform)
print(train_data[139])

