import os
import torch
import torch.nn as nn

from transformers import AutoModel
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import RobertaTokenizer
from transformers import get_linear_schedule_with_warmup

from utils.data import load_data, DataPipeline
from utils.collate import collate_fn
from model.classification import ClassificationParlamint
from utils.data_module import ParliamentDataModule


if __name__ == "__main__":
    PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
    config = OmegaConf.load(PATH_PROJECT + "/config.yaml")
    print(f"Config:\n\n{OmegaConf.to_yaml(config)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading dataset...")
    raw_data = load_data(config)
    print(f"Loaded dataset... Samples loaded: {len(raw_data)} ")

    pipeline = DataPipeline(config)
    data, class_weights = pipeline.prepare_dataset(raw_data)

    print("Loading encoder... ")
    tokenizer = RobertaTokenizer.from_pretrained(config.llm.model)
    # tokenizer = DummyTokenizer()
    encoder = AutoModel.from_pretrained(config.llm.model)
    # encoder = DummyEncoder()

    data_module = ParliamentDataModule(config, data, tokenizer, collate_fn)

    print("Prepearing data loaders...")
    data_loaders = data_module.get_dataloaders()

    model = ClassificationParlamint(encoder, len(data_module.orientation_labels))
    model.to(device)

    # TODO: Set ignore index, set pad token in config or constnts.
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device).float())
    optimizer = torch.optim.AdamW(
        lr=config.training.lr,
        params=model.parameters(),
        weight_decay=config.training.weight_decay,
    )
    num_training_steps = len(data_loaders["train"]) * config.training.epochs
    num_warmup_steps = int(0.05 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # TODO: Move this to a class...
    # TODO: Add early stopping
    train_loss_list = []
    val_loss_list = []
    for epoch in range(0, config.training.epochs):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(
            data_loaders["train"],
            desc=f"Epoch {epoch } Training",
            total=(len(data_loaders["train"])),
        )
        for batch in train_bar:
            optimizer.zero_grad()

            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            labels = labels.to(device)

            outputs = model(**inputs)

            loss = loss_fn(outputs, labels)
            train_loss += loss.item()
            loss.backward()

            optimizer.step()
            train_bar.set_postfix(train=f"Loss: {loss.item():.4f}")
        avg_train_loss = train_loss / len(data_loaders["train"])
        train_loss_list.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(
                data_loaders["val"],
                desc=f"Epoch {epoch} Validation",
                total=len(data_loaders["val"]),
                colour="green",
            )
            for batch in val_bar:
                inputs, labels = batch
                inputs = {k: v.to(device) for k, v in inputs.items()}
                labels = labels.to(device)

                outputs = model(**inputs)

                loss = loss_fn(outputs, labels)
                val_loss += loss.item()

                val_bar.set_postfix(val=f"Loss: {loss.item():.4f}")

        avg_val_loss = val_loss / len(data_loaders["val"])
        val_loss_list.append(avg_val_loss)
        scheduler.step()
