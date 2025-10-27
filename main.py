import os
from omegaconf import OmegaConf
from functools import partial
from utils.data import load_data, DataPipeline
from dataset.parliment import ParlimentDataset
from transformers import RobertaTokenizer
from torch.utils.data import DataLoader


if __name__ == "__main__":

    PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
    config = OmegaConf.load(PATH_PROJECT + "/config.yaml")

    print(f"Config:\n\n{OmegaConf.to_yaml(config)}")

    print("Loading dataset...")
    raw_data = load_data(config)
    print(f"Loaded dataset... Samples loaded: {len(raw_data)} ")

    pipeline = DataPipeline(config)
    data = pipeline.prepare_dataset(raw_data)

    tokenizer = RobertaTokenizer.from_pretrained(config.llm.model)
    dataset_partial = partial(ParlimentDataset, tokenizer=tokenizer)

    train_dataset = dataset_partial(data["train"])
    val_dataset = dataset_partial(data["val"])
    test_dataset = dataset_partial(data["test"])

    train_loader = DataLoader(
        train_dataset, batch_size=config.train.batch_size, shuffle=False
    )

    it_train_loader = iter(train_loader)
    sample = next(it_train_loader)
