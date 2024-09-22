from torch.utils.data import Dataset
import torch

class QADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = torch.tensor(self.data[idx]["question"], dtype=torch.long)
        answer = torch.tensor(self.data[idx]["answer"], dtype=torch.long)
        return question, answer
