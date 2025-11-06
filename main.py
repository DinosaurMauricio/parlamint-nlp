import os
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from transformers import RobertaTokenizer, AutoModel, get_linear_schedule_with_warmup

from utils.dataset_builder import load_data, DatasetBuilder
from utils.collate import collate_fn
from utils.seed import set_seed
from utils.label_encoder import LabelEncoder
from utils.data_loader_builder import ParliamentDataLoaderBuilder
from model.classification import ClassificationParlamint
from training.model_trainer import ModelTrainer


if __name__ == "__main__":
    PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
    config = OmegaConf.load(PATH_PROJECT + "/config.yaml")
    print(f"Config:\n\n{OmegaConf.to_yaml(config)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    set_seed(config.training.seed)
    print(f"Seed: {config.training.seed}")

    print("Loading dataset...")
    raw_data = load_data(config)
    print(f"Loaded dataset... Samples loaded: {len(raw_data)} ")

    dataset_builder = DatasetBuilder(config)
    data = dataset_builder.prepare_dataset(raw_data)

    print(f"Train samples: {len(data['train'])}")
    print(f"Val samples: {len(data['val'])}")
    print(f"Test samples: {len(data['test'])}")

    print("Loading Labels...")
    label_encoder = LabelEncoder()

    class_weights = dataset_builder.compute_class_weights(
        label_encoder.classes, data["train"]
    )

    print("Loading Encoder... ")
    from dummy_classes import DummyEncoder, DummyTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(config.llm.model)
    # tokenizer = DummyTokenizer()
    encoder = AutoModel.from_pretrained(config.llm.model)
    # encoder = DummyEncoder()

    dataloader_builder = ParliamentDataLoaderBuilder(
        config, data, tokenizer, collate_fn
    )

    print("Prepearing data loaders...")
    data_loaders = dataloader_builder.get_dataloaders()

    print("Loading model...")
    model = ClassificationParlamint(
        encoder,
        len(label_encoder),
        config,
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total parameters: {total_params} \nTrainable parameters: {trainable_params}"
    )

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights).to(device).float(),
        ignore_index=tokenizer.pad_token_id,
    )
    optimizer = torch.optim.AdamW(
        lr=config.training.lr,
        params=model.parameters(),
        weight_decay=config.training.weight_decay,
    )
    num_training_steps = len(data_loaders["train"]) * config.training.epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    args = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": loss_fn,
        "device": device,
    }

    trainer = ModelTrainer(**args)
    trainer.train(data_loaders, config.training.epochs)
