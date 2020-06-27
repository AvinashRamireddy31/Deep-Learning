import os
from torch.utils.data import Dataset, DataLoader, IterableDataset


class MyIterableDataset(IterableDataset):
    def __init__(self, data):
        super().__init__()
        self.data = data
    
    def __iter__(self):
        return iter(self.data)

iterable_dataset = MyIterableDataset(range(1,1000))

loader = DataLoader(iterable_dataset, batch_size = 4)

for batch in loader:
    print(batch)











class MyMapDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

map_dataset = MyMapDataset(data=range(1,1000))

loader = DataLoader(map_dataset, batch_size = 1000)

# for batch in loader:
    # print(batch)
 






















































# class NumbersDataset(Dataset):
#     def __init__(self, low, high):
#         super().__init__()
#         self.samples = list(range(low,high))

#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         return self.samples[idx]

    
# if __name__ == "__main__":
#     ds = NumbersDataset(1,100)
#     print(len(ds))
#     print(ds[10])
#     print(ds[11:18])

        
