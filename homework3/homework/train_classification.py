
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from homework.models import load_model, save_model
from homework.metrics import AccuracyMetric
from homework.datasets.classification_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    num_workers: int = 2,
    train_transform: str = "default",
    val_transform: str = "default",
    weight_decay: float = 1e-4,
    **kwargs,
):
    # -------------------------
    # device
    # -------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA/MPS not available, using CPU")
        device = torch.device("cpu")

    # -------------------------
    # reproducibility
    # -------------------------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # -------------------------
    # logging
    # -------------------------
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # -------------------------
    # model
    # -------------------------
    model = load_model(model_name, **kwargs).to(device)

    # -------------------------
    # data
    # -------------------------
    train_data = load_data(
        "classification_data/train",
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_pipeline=train_transform,
    )

    val_data = load_data(
        "classification_data/val",
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        transform_pipeline=val_transform,
    )

    # -------------------------
    # loss / optimizer
    # -------------------------
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # optional but often helpful
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(num_epoch // 3, 1),
        gamma=0.5,
    )

    global_step = 0
    train_metric = AccuracyMetric()
    val_metric = AccuracyMetric()

    best_val_acc = 0.0

    # -------------------------
    # training loop
    # -------------------------
    for epoch in range(num_epoch):
        model.train()
        train_metric.reset()
        running_train_loss = 0.0

        for img, label in train_data:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()

            logits = model(img)
            loss = loss_func(logits, label)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * img.size(0)

            preds = logits.argmax(dim=1)
            train_metric.add(preds, label)

            logger.add_scalar("train/loss_step", loss.item(), global_step)
            logger.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)

            global_step += 1

        # epoch-level train metrics
        train_results = train_metric.compute()
        epoch_train_acc = train_results["accuracy"]
        epoch_train_loss = running_train_loss / len(train_data.dataset)

        # -------------------------
        # validation
        # -------------------------
        model.eval()
        val_metric.reset()
        running_val_loss = 0.0

        with torch.inference_mode():
            for img, label in val_data:
                img = img.to(device)
                label = label.to(device)

                logits = model(img)
                loss = loss_func(logits, label)

                preds = logits.argmax(dim=1)
                val_metric.add(preds, label)

                running_val_loss += loss.item() * img.size(0)

        val_results = val_metric.compute()
        epoch_val_acc = val_results["accuracy"]
        epoch_val_loss = running_val_loss / len(val_data.dataset)

        # -------------------------
        # log epoch metrics
        # -------------------------
        logger.add_scalar("epoch/train_loss", epoch_train_loss, epoch)
        logger.add_scalar("epoch/train_accuracy", epoch_train_acc, epoch)
        logger.add_scalar("epoch/val_loss", epoch_val_loss, epoch)
        logger.add_scalar("epoch/val_accuracy", epoch_val_acc, epoch)

        # print on first, last, every 5th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 5 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_loss={epoch_train_loss:.4f} "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_loss={epoch_val_loss:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

        # save best model for grading
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")
            print(f"Saved new best model with val_acc={best_val_acc:.4f}")

        scheduler.step()

    # save final checkpoint copy in log dir
    torch.save(model.state_dict(), log_dir / f"{model_name}_final.th")
    print(f"Final model copy saved to {log_dir / f'{model_name}_final.th'}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="classifier")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--train_transform", type=str, default="default")
    parser.add_argument("--val_transform", type=str, default="default")
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    train(**vars(parser.parse_args()))
