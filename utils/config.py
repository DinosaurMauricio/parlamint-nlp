import wandb
import json
import optuna

from omegaconf import OmegaConf
from transformers import (
    RobertaTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
    RobertaForSequenceClassification,
)

from callbacks.optuna_callback import OptunaCallback
from model.custom_classifier import CustomClassifier
from utils.dataset_builder import DatasetBuilder

# TODO: Could be a class...


def setup_optuna_trial(config, trial):
    optuna_callback = None
    if trial:
        config.training.lr = trial.suggest_float(
            "lr", config.optuna.lr.min, config.optuna.lr.max, log=True
        )

        config.training.batch_size = trial.suggest_categorical(
            "batch_size", config.optuna.batch_size
        )
        config.training.weight_decay = trial.suggest_float(
            "weight_decay",
            config.optuna.weight_decay.min,
            config.optuna.weight_decay.max,
            log=True,
        )

        config.training.classifier.hidden_dim = trial.suggest_categorical(
            "classifier_hidden_dim", config.optuna.classifier.hidden_dim
        )

        optuna_callback = OptunaCallback(trial)

    return optuna_callback


def setup_logging(config, project_name, group_id, use_wandb):
    wandb_callback = None

    if use_wandb:
        config_container = OmegaConf.to_container(config, resolve=True)
        cfg_json = json.loads(json.dumps(config_container))
        wandb.init(
            project=project_name,
            group=group_id,
            config=cfg_json,
            reinit="finish_previous",  # Allow to write a new logs on Wandb
        )
        # TODO: Depending how this progress it could be moved to a class
        # use magic method __call__ and set other configs
        wandb_callback = lambda metrics: wandb.log(metrics)

    return wandb_callback


def setup_optuna_study(objective, number_trials=0):
    study = optuna.create_study(
        study_name="ParlaParla",
        direction="minimize",
    )
    study.optimize(
        objective,
        n_trials=number_trials,
    )
    return study


def get_model_class(model_name):
    models = {
        "custom": CustomClassifier,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}.")

    return models[model_name]


def create_model(config, encoder, label_encoder=None):
    ModelClass = get_model_class(config.training.model)

    if ModelClass is CustomClassifier:
        if label_encoder is None:
            raise ValueError("label_encoder cannot be None for CustomClassifier")
        model = ModelClass(encoder, len(label_encoder), config)

    return model


def print_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total parameters: {total_params} \nTrainable parameters: {trainable_params}"
    )


def setup_encoder_tokenizer(config):
    print("Loading Encoder... ")
    if config.training.model == "pretrained":
        encoder = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", labels=11
        )
    else:
        encoder = AutoModel.from_pretrained(config.llm.model)

    tokenizer = RobertaTokenizer.from_pretrained(config.llm.model)
    return encoder, tokenizer

    # dummy classes to run on local and verify it can run
    from dummy_classes import DummyEncoder, DummyTokenizer

    return DummyEncoder(), DummyTokenizer()


def setup_scheduler(train_size, num_epochs, optimizer):
    num_training_steps = train_size * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    return scheduler


def setup_datasets(config, raw_data, label_encoder):
    dataset_builder = DatasetBuilder(config)
    data = dataset_builder.prepare_dataset(raw_data)

    print(f"Train samples: {len(data['train'])}")
    print(f"Val samples: {len(data['val'])}")
    print(f"Test samples: {len(data['test'])}")

    class_weights = dataset_builder.compute_class_weights(
        label_encoder.classes, data["train"]
    )

    return data, class_weights
