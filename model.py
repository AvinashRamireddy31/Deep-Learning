import os
from torch.utils.data import Dataset

class NumbersDataset(Dataset):
    def __init__(self, low, high):
        super().__init__()
        self.samples = list(range(low,high))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

    
if __name__ == "__main__":
    ds = NumbersDataset(1,100)
    print(len(ds))
    print(ds[10])
    print(ds[11:18])

        
