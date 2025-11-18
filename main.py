import wandb
import argparse

from omegaconf import OmegaConf

from utils.config import (
    OptunaManager,
)

from utils.constants import PATH_PROJECT
from training.setup import optuna_objective, train_model


def main():
    parser = argparse.ArgumentParser(description="ParlaParla")
    parser.add_argument("--log", action="store_true", help="Log to wandb")
    parser.add_argument("--optuna", action="store_true", help="Run optuna trials")
    parser.add_argument(
        "--number_trials", type=int, default=5, help="Number of trial for optuna"
    )
    args = parser.parse_args()

    if args.optuna:
        print("Starting optuna trials...")
        # hyperparameter optimization (hpo)
        objective = lambda trial: optuna_objective(trial, use_wandb=args.log)
        study = OptunaManager.create_study(objective, args.number_trials)
        print("Best params:", study.best_params)
    else:
        print("Single run...")
        config = OmegaConf.load(PATH_PROJECT + "/config.yaml")
        print(f"Config:\n\n{OmegaConf.to_yaml(config)}")
        results = train_model(config, use_wandb=args.log)

    if args.log:
        wandb.finish()


if __name__ == "__main__":
    main()
