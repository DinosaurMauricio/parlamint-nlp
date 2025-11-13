import wandb
import json
import optuna
import os
import pandas as pd

from omegaconf import OmegaConf

from callbacks.optuna_callback import OptunaCallback
from utils.dataset_builder import DatasetBuilder
from utils.file_loader import ParlaMintFileLoader
from utils.label_encoder import LabelEncoder


class OptunaManager:

    def __init__(self, config):
        self.config = config

    def setup_trial_callback(self, trial):
        if not trial:
            return None

        self.config.training.lr = trial.suggest_float(
            "lr", self.config.optuna.lr.min, self.config.optuna.lr.max, log=True
        )

        self.config.training.batch_size = trial.suggest_categorical(
            "batch_size", self.config.optuna.batch_size
        )

        self.config.training.weight_decay = trial.suggest_float(
            "weight_decay",
            self.config.optuna.weight_decay.min,
            self.config.optuna.weight_decay.max,
            log=True,
        )

        self.config.training.classifier.hidden_dim = trial.suggest_categorical(
            "classifier_hidden_dim", self.config.optuna.classifier.hidden_dim
        )

        return OptunaCallback(trial)

    @staticmethod
    def create_study(objective, n_trials):
        study = optuna.create_study(
            study_name="ParlaParla",
            direction="minimize",
        )

        study.optimize(objective, n_trials=n_trials)

        return study


class WandbManager:
    def __init__(self, config, project_name, group_id, enbaled):
        self.config = config
        self.name = project_name
        self.group_id = group_id
        self.enabled = enbaled

    def setup_callback(self):

        if not self.enabled:
            return None

        config_container = OmegaConf.to_container(self.config, resolve=True)
        cfg_json = json.loads(json.dumps(config_container))
        wandb.init(
            project=self.name,
            group=self.group_id,
            config=cfg_json,
            reinit="finish_previous",  # Allow to write a new logs on Wandb
        )
        # TODO: Depending how this progress it could be moved to a class
        # use magic method __call__ and set other configs
        callback = lambda metrics: wandb.log(metrics)

        return callback


class DataManager:
    def __init__(self, config):
        self.config = config
        self._data = None

        print("Loading data...")
        self._load_data()
        print(f"Loaded dataset... Samples loaded: {len(self._data)} ")  # type: ignore

    def _load_data(self):
        if self._data is not None:
            return

        # load parquet file
        if os.path.exists(self.config.paths.preprocessed_data):
            self._data = pd.read_parquet(self.config.paths.preprocessed_data)
            return

        # load raw data
        loader = ParlaMintFileLoader(self.config)
        samples, _ = loader.load_samples()
        self._data = pd.DataFrame(samples)

    def setup_datasets(self):
        if self._data is None:
            raise ValueError("data is empty, cannot setup datasets")

        dataset_builder = DatasetBuilder(self.config)
        data = dataset_builder.prepare_dataset(self._data)

        print(f"Train samples: {len(data['train'])}")
        print(f"Val samples: {len(data['val'])}")
        print(f"Test samples: {len(data['test'])}")

        class_weights = dataset_builder.compute_class_weights(
            LabelEncoder().classes, data["train"]
        )

        return data, class_weights
