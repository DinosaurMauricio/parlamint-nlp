import os
import torch
import pandas as pd

from utils.file_loader import ParlaMintFileLoader
from utils.filters import DataFilter


class ParlimentDataset(torch.utils.data.Dataset):
    def __init__(self, config):
        self.config = config

        if os.path.exists(config.paths.preprocessed_data) and config.load_preprocessed:
            samples = pd.read_parquet(config.paths.preprocessed_data)
        else:
            loader = ParlaMintFileLoader(config)
            samples, _ = loader.load_samples()

        self.samples = (
            DataFilter(samples)
            .select_columns(
                ["sent_id", "ID_meta", "text", "Party_orientation", "Words"]
            )
            .replace_hyphen_with_undefined("Party_orientation")
            .drop_duplicate_texts()
            .filter_nonempty_rows()
            .filter_by_threshold(
                "Words", config.dataset.word_count.min, config.dataset.word_count.max
            )
            .apply()
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
