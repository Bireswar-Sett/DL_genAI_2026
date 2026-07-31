import torch
from tqdm import tqdm

class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        scheduler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(dataloader, desc="Training", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader, metric_fn):
        self.model.eval()
        total_loss = 0.0
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()

                all_logits.append(outputs.logits.cpu())
                all_labels.append(labels.cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)
        score = metric_fn(logits, labels)

        return {
            "loss": total_loss / len(dataloader),
            "MAP@3": score
        }

    def fit(self, train_loader, val_loader, epochs, metric_fn):
        best_score = -1.0
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, metric_fn)

            print(
                f"Train Loss : {train_loss:.4f}\n"
                f"Val Loss   : {val_metrics['loss']:.4f}\n"
                f"MAP@3      : {val_metrics['MAP@3']:.4f}\n"
            )

            if val_metrics["MAP@3"] > best_score:
                best_score = val_metrics["MAP@3"]
                torch.save(self.model.state_dict(), "best_model.pt")
