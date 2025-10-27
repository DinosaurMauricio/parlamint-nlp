import os
import torch
import torch.nn as nn

from transformers import AutoModel
from omegaconf import OmegaConf
from tqdm import tqdm

from utils.data import load_data, DataPipeline
from utils.collate import collate_fn
from model.classification import ClassificationParlamint
from utils.data_module import ParliamentDataModule


if __name__ == "__main__":
    PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
    config = OmegaConf.load(PATH_PROJECT + "/config.yaml")

    print(f"Config:\n\n{OmegaConf.to_yaml(config)}")

    print("Loading dataset...")
    raw_data = load_data(config)
    print(f"Loaded dataset... Samples loaded: {len(raw_data)} ")

    pipeline = DataPipeline(config)
    data, class_weights = pipeline.prepare_dataset(raw_data)

    data_module = ParliamentDataModule(config, data, collate_fn)

    print("Prepearing data loaders...")
    data_loaders = data_module.get_dataloaders()

    encoder = AutoModel.from_pretrained(config.llm.model)

    model = ClassificationParlamint(encoder, len(data_module.orientation_labels))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
    optimizer = torch.optim.AdamW(lr=config.training.lr, params=model.parameters())
