import torch
import torch.nn as nn

from transformers import (
    RobertaTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    RobertaForSequenceClassification,
)

if torch.cuda.is_available():
    from peft import LoraConfig, get_peft_model, TaskType

from callbacks.optuna_callback import OptunaCallback
from model.custom_classifier import CustomClassifier
from utils.dataset_builder import DatasetBuilder
from utils.file_loader import ParlaMintFileLoader
from utils.label_encoder import LabelEncoder


def configure_optimizer(model, lr, weigth_decay):

    optimizer = torch.optim.AdamW(
        lr=lr,
        params=model.parameters(),
        weight_decay=weigth_decay,
    )

    return optimizer


def configure_loss(model_type, pad_token_id, weight=None):
    loss_fn = None
    if model_type == "custom":
        loss_fn = nn.CrossEntropyLoss(
            # weight=torch.tensor(class_weights).to(device).float(),
            ignore_index=pad_token_id,
        )
    return loss_fn


def configure_scheduler(train_size, num_epochs, optimizer):
    num_training_steps = train_size * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return scheduler


def configure_encoder_tokenizer(model_name, model_type):
    print("Loading Encoder... ")
    if model_type == "pretrained":
        encoder = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=11
        )
    else:
        encoder = AutoModel.from_pretrained(model_name)

    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    return encoder, tokenizer

    # dummy classes to run on local and verify it can run
    from dummy_classes import DummyEncoder, DummyTokenizer

    return DummyEncoder(), DummyTokenizer()


def create_model(config, encoder, label_encoder=None):
    model = None
    model_type = config.training.model_type
    if model_type == "custom":
        if label_encoder is None:
            raise ValueError("label_encoder required for CustomClassifier")
        model = CustomClassifier(encoder, len(label_encoder), config)
    elif model_type == "pretrained":
        model = encoder
    else:
        raise ValueError(f"Unknown model: {model_type}")

    return model


def apply_lora(model, r=32, lora_alpha=32, target_modules=None, lora_dropout=0.05):
    if target_modules is None:
        target_modules = ["query", "value"]

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )

    model = get_peft_model(model, lora_config)  # type: ignore
    return model


def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total parameters: {total_params} \nTrainable parameters: {trainable_params}"
    )
