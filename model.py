# Import libraries
import os

import torch
import torch.nn as nn

from torch.nn import Linear 

torch.manual_seed(1)  

# Model construction
class LR(nn.Module): 
    def __init__(self, input_size, output_size):
       super().__init__()
       self.linear = nn.Linear(in_features= input_size, out_features= output_size)
    
    def forward(self, x):
        pred =  self.linear(x)
        return pred 

#Implementation
model = LR(2,1) 
[w,b] = model.parameters()

print(w)
print(w[0][1].item())

# result = model.forward(x=torch.tensor([[1.0, 2.0], [2.0, 4.5]], dtype=torch.float32))
# print(result)




# in_features = torch.tensor([1,2,3,4], dtype = torch.float32)

# weight_matrix = torch.tensor([
#     [1,2,3,4],
#     [2,3,4,5],
#     [3,4,5,6]
# ], dtype = torch.float32)

# fc = nn.Linear(in_features = 4, out_features = 3, bias = False)
# fc.weight  = nn.Parameter(weight_matrix)
# result = fc(in_features)
# print(result)










# torch.manual_seed(1)

# model = Linear(in_features=1, out_features=1)
# print(model.bias, model.weight)

# x = torch.tensor([[2.0], [5.6]])
# print(model(x))

















# from torch.utils.data import Dataset, DataLoader, IterableDataset

# class MyIterableDataset(IterableDataset):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data
    
#     def __iter__(self): 
#         return iter(self.data)

# iterable_dataset = MyIterableDataset(range(1,1000))

# loader = DataLoader(iterable_dataset, batch_size = 4)

# for batch in loader:
#     # print(batch)











# class MyMapDataset(Dataset):
#     def __init__(self, data):
#         super().__init__()
#         self.data = data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         return self.data[idx]

# map_dataset = MyMapDataset(data=range(1,1000))

# loader = DataLoader(map_dataset, batch_size = 1000)

# # for batch in loader:
#     # print(batch)
 






















































# # class NumbersDataset(Dataset):
# #     def __init__(self, low, high):
# #         super().__init__()
# #         self.samples = list(range(low,high))

# #     def __len__(self):
# #         return len(self.samples)
    
# #     def __getitem__(self, idx):
# #         return self.samples[idx]

    
# # if __name__ == "__main__":
# #     ds = NumbersDataset(1,100)
# #     print(len(ds))
# #     print(ds[10])
# #     print(ds[11:18])

        
