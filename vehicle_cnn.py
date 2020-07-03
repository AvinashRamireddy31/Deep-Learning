import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
from matplotlib import image

import torch as T 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision   
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import transforms


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
test_data = VehicleDataset(train_images, train_labels, transform)


#Dataloaders
trainloader = DataLoader(train_data, batch_size = 128, shuffle = True)
testloader = DataLoader(test_data, batch_size = 128, shuffle = True)

#Define Neural net class
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)

        self.fc1   = nn.Linear(in_features=140450, out_features=500)
        self.fc2   = nn.Linear(in_features=500, out_features=10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        
        x = self.fc2(x)

        return x

net = Net().to(device)
print(net)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum = 0.0)

def train(net, trainloader):
    print("Training started")

    for epoch in range(10): # No. of epochs
        running_loss = 0
        for data in trainloader:
            # Data pixels and labels to GPU if available
            inputs = data[0].to(device, non_blocking = True)
            labels = data[1].to(device, non_blocking = True)

            # Set the parameter gradients to zero
            optimizer.zero_grad()
            output = net(inputs)
            loss = criterion(output, labels)

            # Propagate the loss backward
            loss.backward()

            # Update the gradients
            optimizer.step()

            running_loss += loss.item()
        print('[Epoch %d] loss: %.3f' %(epoch+1, running_loss/len(trainloader)))

    print("Done training")

def test(net, testloader):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in testloader:
            inputs = data[0].to(device, non_blocking = True)
            labels = data[1].to(device, non_blocking = True)
            outputs = net(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)

            correct += (predicted == labels).sum().item()

        print('Accuracy of the network on test images: %0.3f'%(100*correct/total))

train(net, trainloader)
test(net, testloader)
