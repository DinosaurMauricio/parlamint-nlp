import torch


class ParlimentDataset(torch.utils.data.Dataset):
    _orientation_labels = None

    def __init__(self, data, tokenizer):
        self.samples = data
        self.tokenizer = tokenizer

        if ParlimentDataset._orientation_labels is None:
            ParlimentDataset._orientation_labels = {
                value: i
                for i, value in enumerate(self.samples["Party_orientation"].unique())
            }

    @property
    def orientation_labels(self):
        assert ParlimentDataset._orientation_labels is not None

        return ParlimentDataset._orientation_labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples.iloc[idx]
        tokenized_text = self.tokenizer(sample.text, return_tensors="pt")

        assert ParlimentDataset._orientation_labels is not None

        orientation_label = ParlimentDataset._orientation_labels[
            sample.Party_orientation
        ]

        return tokenized_text, orientation_label
