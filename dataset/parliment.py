import os
import torch
import pandas as pd

from utils.file_loader import ParlaMintFileLoader
from utils.filters import DataFilter


class ParlimentDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer):
        self.samples = data
        self.tokenizer = tokenizer
        self.data_classes = self.samples["Party_orientation"].to_dict()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
