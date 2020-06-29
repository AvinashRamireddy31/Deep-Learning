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
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
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

def train_model(model, data_loaders, criterion, optimizer, epochs = 5):
    for epoch in range(epochs):
        print(f"Epoch {epoch}/ {epochs-1} ")
        print('_'*15)

        for phase in ['train', 'eval']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            correct = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with T.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = T.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                loss_item = loss.item # Tensor to numpy using 'item
                
                running_loss += loss_item * inputs.size(0)
                correct += T.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            epoch_acc = correct.double() / len(data_loaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))















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