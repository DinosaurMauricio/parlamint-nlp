import torch
from utils.label_encoder import LabelEncoder


class ParlimentDataset(torch.utils.data.Dataset):

    def __init__(self, data, tokenizer):
        self.samples = data
        self.tokenizer = tokenizer
        # LabelEncoder needs to be preconfigured before creating this Dataset class
        # LabelEncoder is set as a singleton, so it can be initialized outside this class
        # else it will throw RuntimeError with LabelEncoder not configured
        self._orientation_label_encoder = LabelEncoder()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples.iloc[idx]
        tokenized_text = self.tokenizer(sample.text, return_tensors="pt")

        orientation_label = self._orientation_label_encoder.encode(
            sample.Party_orientation
        )

        result = {
            "tokenized_text": tokenized_text,
            "label": orientation_label,
        }

        return result
