from text_lstm.config import TEMP_PATH

from torch.utils.data import Dataset
import numpy as np
import torch


class MyDataset(Dataset):

    def __init__(self, cate, padding_len):
        self.x = np.load(TEMP_PATH + '/{}.input.npy'.format(cate))[:, :padding_len]
        self.y = np.load(TEMP_PATH + '/{}.target.npy'.format(cate))

    def __getitem__(self, item):
        return torch.tensor(self.x[item], dtype=torch.long), torch.tensor(self.y[item], dtype=torch.long)

    def __len__(self):
        return len(self.x)
