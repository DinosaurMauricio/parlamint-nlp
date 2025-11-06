import optuna


class OptunaCallback:

    def __init__(self, trial, metric="val_loss") -> None:
        self.trial = trial
        self.metric = metric

    def __call__(self, metrics, epoch):
        self.trial.report(metrics[self.metric], epoch)

        # Handle pruning based on the intermediate value.
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()
