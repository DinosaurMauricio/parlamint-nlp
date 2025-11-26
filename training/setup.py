import torch

from omegaconf import OmegaConf

from utils.model import (
    configure_encoder_tokenizer,
    configure_loss,
    configure_optimizer,
    configure_scheduler,
    create_model,
    print_model_stats,
    apply_lora,
)
from utils.config import (
    OptunaManager,
    WandbManager,
    DataManager,
)

from utils.collate import collate_fn
from utils.seed import set_seed
from utils.label_encoder import LabelEncoder
from utils.data_loader_builder import ParliamentDataLoaderBuilder
from utils.constants import PROJECT_NAME, PATH_PROJECT, WANDB_ID
from training.model_trainer import ModelTrainer


def setup_training(config, optuna_callback=None, wandb_callback=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    set_seed(config.training.seed)
    print(f"Seed: {config.training.seed}")

    encoder, tokenizer = configure_encoder_tokenizer(
        config.llm.model, config.training.model_type
    )
    data_manager = DataManager(config)
    data, _ = data_manager.setup_datasets()

    print("Prepearing data loaders...")
    dataloader_builder = ParliamentDataLoaderBuilder(
        config, data, tokenizer, collate_fn
    )
    data_loaders = dataloader_builder.get_dataloaders()

    model = create_model(config, encoder, LabelEncoder())
    # model = apply_lora(model)
    print_model_stats(model)

    loss_fn = configure_loss(config.training.model_type, tokenizer.pad_token_id)
    optimizer = configure_optimizer(
        model, config.training.lr, config.training.weight_decay
    )
    scheduler = configure_scheduler(
        len(data_loaders["train"]), config.training.epochs, optimizer
    )

    trainer = ModelTrainer(
        hpo_callback=optuna_callback,
        log_callback=wandb_callback,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
    )

    return trainer, data_loaders


def train_model(config, use_wandb=False):
    wandb_callback = WandbManager(
        config, PROJECT_NAME, WANDB_ID, use_wandb
    ).setup_callback()

    trainer, data_loaders = setup_training(config, wandb_callback=wandb_callback)
    return trainer.train(data_loaders, config.training.epochs)


def optuna_objective(trial, use_wandb=False):
    config = OmegaConf.load(PATH_PROJECT + "/config.yaml")

    optuna_callback = OptunaManager(config).setup_trial_callback(trial)
    wandb_callback = WandbManager(
        config, PROJECT_NAME, WANDB_ID, use_wandb
    ).setup_callback()

    trainer, data_loaders = setup_training(
        config, optuna_callback=optuna_callback, wandb_callback=wandb_callback
    )

    results = trainer.train(data_loaders, config.training.epochs)
    return min(results["val_losses"])
