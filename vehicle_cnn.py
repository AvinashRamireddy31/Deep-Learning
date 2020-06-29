import os
import numpy as np
import pandas as pd

from PIL import Image

from sklearn.model_selection import train_test_split

import torch as T 
import torch.nn as nn
import torch.optim as optim

import torchvision  
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


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

def get_all_images():
    all_images = []

    #Iterate through images folder
    for filename in os.listdir(IMAGES_DIRECTORY):
        if filename.endswith("jpg"):# Load 'jpg' images only
            img = Image.open(os.path.join(IMAGES_DIRECTORY, filename))
            all_images.append(img)

    return all_images

def get_images(data_set):
    images = [] 

    for filename in data_set['image_names']:
        img = Image.open(os.path.join(IMAGES_DIRECTORY,filename))
        images.append(img)

    return images

def get_train_and_test_split(X, y):
    return train_test_split(X, y, test_size = 0.33, random_state = 42)



#Implementation
train_data = get_train_data()
test_data = get_test_data()
train_images = get_images(train_data)
test_images = get_images(test_data)

#Load train and validation data
X = train_images
y = train_data['emergency_or_not']

# Train and validation split
X_train, X_dev, y_train, y_dev = get_train_and_test_split(X, y)
print(len(X_train))