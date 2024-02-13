import torch
from torch.utils.data import Dataset

class MNIST(Dataset):

    def __init__(self, X, y, y2=None, si=None):
        self.X = X
        self.y = y
        self.y2 = y2
        self.si = si

    def __getitem__(self, index):
        instances = self.X[index, :]
        labels = self.y[index]
        if self.y2 is not None:
            labels2 = self.y2[index]
        if self.si is not None:
            sis = self.si[index]

        instances = torch.tensor(instances, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        if self.y2 is not None:
            labels2 = torch.tensor(labels2, dtype=torch.float32)
        if self.si is not None:
            sis = torch.tensor(sis, dtype=torch.float32)

        if self.si is not None:
            if self.y2 is not None:
                return instances, labels, labels2, sis
            else:
                return instances, labels, sis

    def __len__(self):
        return len(self.X)