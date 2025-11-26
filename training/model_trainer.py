import torch
from tqdm import tqdm
from sklearn.metrics import classification_report


class ModelTrainer:

    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        loss_fn,
        device,
        log_callback=None,
        hpo_callback=None,  # hyperparameter optimzer callback (using optuna)
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.log_callback = log_callback
        self.hpo_callback = hpo_callback

        self.scaler = torch.GradScaler(device=device)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        batch_labels = []
        batch_preds = []

        train_bar = tqdm(
            dataloader, desc=f"Epoch {epoch} Training", total=len(dataloader)
        )

        for batch in train_bar:
            self.optimizer.zero_grad()

            inputs, labels = batch
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            labels = labels.to(self.device)

            with torch.autocast(
                device_type=self.device,
                dtype=torch.bfloat16,
            ):
                # loss_fn is None when using a pretrained model only
                if self.loss_fn is None:
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                else:
                    logits = self.model(**inputs)
                    loss = self.loss_fn(logits, labels)

            self._apply_gradient(loss)

            total_loss += loss.item()

            batch_labels.append(labels.cpu())
            batch_preds.append(torch.argmax(logits, dim=1).cpu())

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        return avg_loss, batch_labels, batch_preds

    def validate_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0.0
        batch_preds = []
        batch_labels = []

        with torch.no_grad():
            val_bar = tqdm(
                dataloader,
                desc=f"Epoch {epoch} Validation",
                total=len(dataloader),
                colour="green",
            )

            for batch in val_bar:
                inputs, labels = batch
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = labels.to(self.device)

                # loss_fn is None when using a pretrained model only
                if self.loss_fn is None:
                    outputs = self.model(**inputs, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits
                else:
                    logits = self.model(**inputs)
                    loss = self.loss_fn(logits, labels)

                total_loss += loss.item()

                batch_labels.append(labels.cpu())
                batch_preds.append(torch.argmax(logits, dim=1).cpu())

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)

        return avg_loss, batch_labels, batch_preds

    def train(self, data_loaders, epochs):
        for epoch in range(epochs):
            train_loss, train_labels, train_preds = self.train_epoch(
                data_loaders["train"], epoch
            )
            val_loss, val_labels, val_preds = self.validate_epoch(
                data_loaders["val"], epoch
            )

            train_report = classification_report(
                torch.cat(train_labels),
                torch.cat(train_preds),
                output_dict=True,
                zero_division=0,
            )

            val_report = classification_report(
                torch.cat(val_labels),
                torch.cat(val_preds),
                output_dict=True,
                zero_division=0,
            )

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            self.scheduler.step()

            metrics_stats = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_report["accuracy"],  # type: ignore
                "train_precision_avg": train_report["macro avg"][  # type:ignore
                    "precision"
                ],
                "train_recall_avg": train_report["macro avg"]["recall"],  # type:ignore
                "train_f1_avg": train_report["macro avg"]["f1-score"],  # type:ignore
                "val_acc": val_report["accuracy"],  # type: ignore
                "val_precision_avg": val_report["macro avg"][  # type:ignore
                    "precision"
                ],
                "val_recall_avg": val_report["macro avg"]["recall"],  # type:ignore
                "val_f1_avg": val_report["macro avg"]["f1-score"],  # type:ignore
            }

            if self.log_callback:
                self.log_callback(metrics_stats)

            if self.hpo_callback:
                self.hpo_callback(metrics_stats, epoch)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def _apply_gradient(self, loss):
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            # need to unscale the gradients of optimizer's before clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
