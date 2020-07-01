#reference: https://debuggercafe.com/custom-dataset-and-dataloader-in-pytorch/

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
print("Device:", device)
#Read data
df_train = pd.read_csv("data/msnist/mnist_train.csv", header = None)
df_test = pd.read_csv("data/msnist/mnist_test.csv", header = None)

print(df_train.shape)
print(df_test.shape)

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
        data = np.asarray(data).astype(np.uint8).reshape(28,28,1)

        if self.transforms:
            data = self.transforms(data)
        
        if self.y is not None:
            return (data, self.y[i])
        else:
            return data

train_data = MNISTDataset(train_images, train_labels, transform)
test_data = MNISTDataset(test_images, test_labels, transform)

#Dataloaders
trainloader = DataLoader(train_data, batch_size = 128, shuffle = True)
testloader = DataLoader(test_data, batch_size = 128, shuffle = True)

#Define Neural net class
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)

        self.fc1   = nn.Linear(in_features=800, out_features=500)
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