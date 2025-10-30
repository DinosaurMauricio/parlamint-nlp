import torch
from tqdm import tqdm


class Trainer:

    def __init__(self, model, optimizer, scheduler, loss_fn, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device

        self.scaler = torch.GradScaler(device=device)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0.0
        tot_correct_predictions = 0
        total_samples = 0

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
                enabled=self.scaler.is_enabled(),
            ):
                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)

            self._apply_gradient(loss)

            total_loss += loss.item()

            tot_correct_predictions += self._calcualte_accuracy(outputs, labels)
            total_samples += labels.size(0)

            train_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = tot_correct_predictions / total_samples
        return avg_loss, avg_accuracy

    def validate_epoch(self, dataloader, epoch):
        self.model.eval()
        total_loss = 0.0
        tot_correct_predictions = 0
        total_samples = 0

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

                outputs = self.model(**inputs)
                loss = self.loss_fn(outputs, labels)

                total_loss += loss.item()

                tot_correct_predictions += self._calcualte_accuracy(outputs, labels)
                total_samples += labels.size(0)

                val_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        avg_accuracy = tot_correct_predictions / total_samples
        return avg_loss, avg_accuracy

    def train(self, data_loaders, epochs):
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(data_loaders["train"], epoch)
            val_loss, val_acc = self.validate_epoch(data_loaders["val"], epoch)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.scheduler.step()

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accuracies": self.train_accuracies,
            "val_accuracies": self.val_accuracies,
        }

    def _apply_gradient(self, loss):
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def _calcualte_accuracy(self, outputs, labels):
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions = (predictions == labels).sum().item()
        return correct_predictions
