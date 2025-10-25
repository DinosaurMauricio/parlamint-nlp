import torch

from utils.file_loader import ParlaMintFileLoader

from utils.constants import CONLLU_EXT


class ParlimentDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config
        loader = ParlaMintFileLoader(config)
        self.samples, _ = loader.load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
