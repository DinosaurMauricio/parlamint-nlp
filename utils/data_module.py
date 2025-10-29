from torch.utils.data import DataLoader
from dataset.parliment import ParlimentDataset
from functools import partial


class ParliamentDataModule:
    def __init__(self, config, data, tokenizer, collate_fn):
        self.config = config
        self.tokenizer = tokenizer
        self.data = data
        self.collate_fn = collate_fn
        self.batch_size = config.training.batch_size

        dataset_partial = partial(ParlimentDataset, tokenizer=self.tokenizer)

        self.datasets = {split: dataset_partial(data[split]) for split in data.keys()}

    def get_dataloader(self, split="train"):
        return DataLoader(
            self.datasets[split],
            batch_size=self.batch_size,
            shuffle=True if split == "train" else False,
            collate_fn=self.collate_fn,
        )

    def get_dataloaders(self):
        """Get dataloaders for all splits (train, val, test)."""
        return {split: self.get_dataloader(split) for split in self.data.keys()}

    @property
    def orientation_labels(self):
        return self.datasets["train"].orientation_labels
