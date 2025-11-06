import os
import torch
import torch.nn as nn
import wandb
import argparse

from omegaconf import OmegaConf
from wandb.sdk.lib.runid import generate_id

from utils.config import (
    setup_optuna_trial,
    setup_logging,
    setup_optuna_study,
    setup_model,
    setup_encoder_tokenizer,
    setup_scheduler,
    setup_datasets,
)
from utils.dataset_builder import load_data
from utils.collate import collate_fn
from utils.seed import set_seed
from utils.label_encoder import LabelEncoder
from utils.data_loader_builder import ParliamentDataLoaderBuilder

from training.model_trainer import ModelTrainer

PATH_PROJECT = os.path.dirname(os.path.abspath(__file__))
PROJECT_NAME = "ParlaParla"
WANDB_ID = generate_id()
parser = argparse.ArgumentParser(description="ParlaParla")

parser.add_argument("--log", action="store_true", help="Log to wandb")
parser.add_argument("--optuna", action="store_true", help="Run optuna trials")
parser.add_argument(
    "--number_trials", type=int, default=5, help="Number of trial for optuna"
)


def single_run(trial=None, use_wandb=False):
    config = OmegaConf.load(PATH_PROJECT + "/config.yaml")
    print(f"Config:\n\n{OmegaConf.to_yaml(config)}")

    optuna_callback = setup_optuna_trial(config, trial)
    wandb_callback = setup_logging(config, PROJECT_NAME, WANDB_ID, use_wandb)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    set_seed(config.training.seed)
    print(f"Seed: {config.training.seed}")

    print("Loading data...")
    raw_data = load_data(config)
    print(f"Loaded dataset... Samples loaded: {len(raw_data)} ")

    print("Loading Labels...")
    label_encoder = LabelEncoder()

    data, class_weights = setup_datasets(config, raw_data, label_encoder)

    encoder, tokenizer = setup_encoder_tokenizer(config)

    dataloader_builder = ParliamentDataLoaderBuilder(
        config, data, tokenizer, collate_fn
    )

    print("Prepearing data loaders...")
    data_loaders = dataloader_builder.get_dataloaders()

    model = setup_model(config, encoder, label_encoder)
    model.to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights).to(device).float(),
        ignore_index=tokenizer.pad_token_id,
    )
    optimizer = torch.optim.AdamW(
        lr=config.training.lr,
        params=model.parameters(),
        weight_decay=config.training.weight_decay,
    )
    scheduler = setup_scheduler(
        len(data_loaders["train"]), config.training.epochs, optimizer
    )

    args = {
        "hpo_callback": optuna_callback,
        "log_callback": wandb_callback,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "loss_fn": loss_fn,
        "device": device,
    }

    trainer = ModelTrainer(**args)
    results = trainer.train(data_loaders, config.training.epochs)

    # optuna requires a val to maximize/minimize
    # so in case a trial is sent we return this value
    if trial:
        return min(results["val_losses"])
    else:
        return results


def run_hpo(num_trials, log):
    # hyperparameter optimization (hpo)
    objective = lambda trial: single_run(trial, use_wandb=log)

    study = setup_optuna_study(objective, num_trials)
    print("Best params:")
    print(study.best_params)
    return study


if __name__ == "__main__":
    args = parser.parse_args()
    if args.optuna:
        print("Starting optuna trials...")
        run_hpo(args.number_trials, args.log)
    else:
        print("Single run...")
        results = single_run(use_wandb=args.log)

    if args.log:
        wandb.finish()
