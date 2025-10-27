from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
from dataset.parliment import ParlimentDataset
from functools import partial


class ParliamentDataModule:
    def __init__(self, config, data, collate_fn):
        self.config = config
        # TODO: Decouple this so any tokenizer can be passed in, instead of hardcoding it here
        self.tokenizer = RobertaTokenizer.from_pretrained(config.llm.model)
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
