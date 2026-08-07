import torch
from tqdm import tqdm
import wandb
from transformers import Trainer, TrainingArguments, TrainerCallback
from torch.optim.lr_scheduler import CosineAnnealingLR


class PredictionResult:
    def __init__(self, predictions):
        self.predictions = predictions


class WandbMetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not wandb.run:
            return

        if "loss" in logs and "epoch" in logs and "eval_loss" not in logs:
            wandb.log({
                "epoch": logs.get("epoch"),
                "learning_rate": logs.get("learning_rate"),
                "train_loss": logs.get("loss"),
            }, step=state.global_step)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None or not wandb.run:
            return

        wandb.log({
            "epoch": state.epoch,
            "learning_rate": metrics.get("learning_rate"),
            "val_loss": metrics.get("eval_loss"),
            "accuracy": metrics.get("eval_accuracy"),
            "MAP@3": metrics.get("eval_MAP@3"),
        }, step=state.global_step)

class MCQTrainer:
    def __init__(
        self, model, optimizer, criterion, device, scheduler, wandb_run,
        checkpoint_path="best_custom_model.pt",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.wandb_run = wandb_run
        self.checkpoint_path = checkpoint_path
        self.globalStep=0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0

        for batch in tqdm(loader):
            q_ids = batch["question_ids"].to(self.device)
            q_mask = batch["question_mask"].to(self.device)
            opt_ids = batch["option_ids"].to(self.device)
            opt_mask = batch["option_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            self.global_step += 1

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.wandb_run.log(
                {
                    "train_loss": loss.item(),
                    "learning_rate": current_lr,
                },
                step=self.global_step,
            )
            scores = self.model(
                q_ids,
                q_mask,
                opt_ids,
                opt_mask,
            )

            loss = self.criterion(scores, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0
        map3_score = 0

        with torch.no_grad():
            for batch in tqdm(loader):
                q_ids = batch["question_ids"].to(self.device)
                q_mask = batch["question_mask"].to(self.device)
                opt_ids = batch["option_ids"].to(self.device)
                opt_mask = batch["option_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                scores = self.model(
                    q_ids,
                    q_mask,
                    opt_ids,
                    opt_mask,
                )

                loss = self.criterion(scores, labels)
                total_loss += loss.item()

                preds = torch.argmax(scores, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                top3 = torch.topk(scores, k=3, dim=1).indices

                for pred, label in zip(top3, labels):
                    pred = pred.tolist()
                    label = label.item()

                    if label in pred:
                        rank = pred.index(label) + 1
                        map3_score += 1 / rank

        return {
            "loss": total_loss / len(loader),
            "accuracy": correct / total,
            "MAP@3": map3_score / total,
        }

    def predict(self, loader):
        self.model.eval()
        all_logits = []

        with torch.no_grad():
            for batch in tqdm(loader):
                q_ids = batch["question_ids"].to(self.device)
                q_mask = batch["question_mask"].to(self.device)
                opt_ids = batch["option_ids"].to(self.device)
                opt_mask = batch["option_mask"].to(self.device)

                scores = self.model(
                    q_ids,
                    q_mask,
                    opt_ids,
                    opt_mask,
                )
                all_logits.append(scores.cpu())

        return PredictionResult(torch.cat(all_logits, dim=0).numpy())

    def fit(self, train_loader, val_loader, epochs):
        best_map = 0
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")

            train_loss = self.train_epoch(train_loader)

            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            print(f"Train Loss : {train_loss:.4f}")

            print(
                f"""
Val Loss   : {val_metrics['loss']:.4f}
Accuracy   : {val_metrics['accuracy']:.4f}
MAP@3      : {val_metrics['MAP@3']:.4f}
"""
            )
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "val_loss": val_metrics["loss"],
                    "accuracy": val_metrics["accuracy"],
                    "MAP@3": val_metrics["MAP@3"],
                },
                step=self.global_step,
            )

            if val_metrics["MAP@3"] > best_map:
                best_map = val_metrics["MAP@3"]
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print("Saved best model")


def build_training_args(
    output_dir="./mcq_model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    run_name=None,
):
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="MAP@3",
        greater_is_better=True,
        fp16=True,
        report_to="wandb",
        run_name=run_name,
        logging_steps=10,
    )


def build_custom_scheduler(optimizer, epochs):
    return CosineAnnealingLR(optimizer, T_max=epochs)


def build_pretrained_trainer(model, train_dataset, val_dataset, training_args, compute_metrics):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[WandbMetricsCallback()],
    )

