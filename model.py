# # Import libraries
# import os

# import numpy as np

# import torch 
# import torchvision as tv
# import torchvision.transforms as transforms


# import torch.nn as nn
# from torch.nn import Linear 
# import torch.nn.functional as F

# from torch.utils.data import DataLoader

# import matplotlib.pyplot as plt


# torch.manual_seed(1)  

# X = [i for i in range(1,10)]
# y = [i*2 for i in X]
# print(y)

# plt.plot(X,y)
# plt.show()











# transform = transforms.Compose(
#     [
#         tv.transforms.ToTensor(), 
#         tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
#     ])


# trainset = tv.datasets.CIFAR10(root= "../data", train = True, download = False, transform = transform )

# dataloader = DataLoader(trainset, batch_size = 4, shuffle = False, num_workers = 4)

# images = iter(dataloader).next()


# class FirstCNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool  = nn.MaxPool2d(2,2) # for images
#         self.conv2 = nn.Conv2d(6,16,5)

#         self.fc1 = nn.Linear(16*5*5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10) # last layer




































































































 

# # # Model construction
# # class Net(nn.Module):
# #     def __init__(self):
# #         super().__init__()

# #         self.convol1 = nn.Conv2d(3, 20, 5)
# #         self.pool = nn.MaxPool2d(2,2)
# #         self.convol2 = nn.Conv2d(20, 30, 5)

# #         self.fc1 = nn.Linear(30*5*5, 300, bias= True)
# #         print("Hellow world")

# # netw = Net()































# # Model construction
# class LR(nn.Module): 
#     def __init__(self, input_size, output_size):
#        super().__init__()
#        self.linear = nn.Linear(in_features = input_size, out_features = output_size)
    
#     def forward(self, x):
#         pred =  self.linear(x)
#         return pred 

# #Implementation
# model = LR(2,1) 
# [w,b] = model.parameters() 

# print(w)
# print(w[0][1].item())

# input_features = torch.tensor([[1.0, 2.0], [2.0, 4.5]], dtype=torch.float32)

# result = model.forward(x= input_features)
# print(result)

# # We can also ignore calling 'forward' directly because nn.Module will do the 'callable' for us which calls 'forward' internally
# result = model(input_features)
# print(result)



# # in_features = torch.tensor([1,2,3,4], dtype = torch.float32)

# # weight_matrix = torch.tensor([
# #     [1,2,3,4],
# #     [2,3,4,5],
# #     [3,4,5,6]
# # ], dtype = torch.float32)

# # fc = nn.Linear(in_features = 4, out_features = 3, bias = False)
# # fc.weight  = nn.Parameter(weight_matrix)
# # result = fc(in_features)
# # print(result)










# # torch.manual_seed(1)

# # model = Linear(in_features=1, out_features=1)
# # print(model.bias, model.weight)

# # x = torch.tensor([[2.0], [5.6]])
# # print(model(x))

















# # from torch.utils.data import Dataset, DataLoader, IterableDataset

# # class MyIterableDataset(IterableDataset):
# #     def __init__(self, data):
# #         super().__init__()
# #         self.data = data
    
# #     def __iter__(self): 
# #         return iter(self.data)

# # iterable_dataset = MyIterableDataset(range(1,1000))

# # loader = DataLoader(iterable_dataset, batch_size = 4)

# # for batch in loader:
# #     # print(batch)











# # class MyMapDataset(Dataset):
# #     def __init__(self, data):
# #         super().__init__()
# #         self.data = data

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         return self.data[idx]

# # map_dataset = MyMapDataset(data=range(1,1000))

# # loader = DataLoader(map_dataset, batch_size = 1000)

# # # for batch in loader:
# #     # print(batch)
 






















































# # # class NumbersDataset(Dataset):
# # #     def __init__(self, low, high):
# # #         super().__init__()
# # #         self.samples = list(range(low,high))

# # #     def __len__(self):
# # #         return len(self.samples)
    
# # #     def __getitem__(self, idx):
# # #         return self.samples[idx]

    
# # # if __name__ == "__main__":
# # #     ds = NumbersDataset(1,100)
# # #     print(len(ds))
# # #     print(ds[10])
# # #     print(ds[11:18])

        
