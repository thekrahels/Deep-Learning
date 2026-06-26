"""
Usage:
    python3 -m homework.train_planner --your_args here
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .datasets.road_dataset import load_data
from .metrics import PlannerMetric
from .models import load_model, save_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_smooth_l1_loss(preds, labels, mask):
    """
    preds:  (B, n_waypoints, 2)
    labels: (B, n_waypoints, 2)
    mask:   (B, n_waypoints)
    """
    mask = mask.float()
    loss = F.smooth_l1_loss(preds, labels, reduction="none")
    loss = loss * mask[..., None]

    return loss.sum() / (mask.sum() * 2.0 + 1e-8)


def move_batch_to_device(batch, device):
    output = {}

    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value

    return output


def run_epoch(model, loader, optimizer, device, train: bool):
    metric = PlannerMetric()

    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        labels = batch["waypoints"]
        labels_mask = batch["waypoints_mask"]

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            preds = model(**batch)
            loss = masked_smooth_l1_loss(preds, labels, labels_mask)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        metric.add(preds.detach(), labels.detach(), labels_mask.detach())

    results = metric.compute()
    results["loss"] = total_loss / max(total_batches, 1)

    return results


def train(
    model_name: str,
    dataset_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    num_workers: int,
):
    device = get_device()
    print(f"Using device: {device}")

    dataset_dir = Path(dataset_dir)

    if model_name in ["mlp_planner", "transformer_planner"]:
        transform_pipeline = "state_only"
    elif model_name == "cnn_planner":
        transform_pipeline = "default"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    train_path = dataset_dir / "train"
    val_path = dataset_dir / "val"

    train_loader = load_data(
        train_path,
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
    )

    val_loader = load_data(
        val_path,
        transform_pipeline=transform_pipeline,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
    )

    model = load_model(model_name)
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_score = float("inf")
    best_epoch = -1

    for epoch in range(1, epochs + 1):
        train_results = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            train=True,
        )

        val_results = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            train=False,
        )

        val_score = val_results["longitudinal_error"] + val_results["lateral_error"]

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss={train_results['loss']:.4f} "
            f"train long={train_results['longitudinal_error']:.4f} "
            f"train lat={train_results['lateral_error']:.4f} | "
            f"val loss={val_results['loss']:.4f} "
            f"val long={val_results['longitudinal_error']:.4f} "
            f"val lat={val_results['lateral_error']:.4f}"
        )

        if val_score < best_score:
            best_score = val_score
            best_epoch = epoch
            output_path = save_model(model)
            print(f"  Saved best model to {output_path}")

    print(f"Best epoch: {best_epoch}, best val score: {best_score:.4f}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="mlp_planner",
        choices=["mlp_planner", "transformer_planner", "cnn_planner"],
    )

    parser.add_argument("--dataset_dir", type=str, default="drive_data")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Use 0 on Windows / VS Code if multiprocessing causes issues.
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    train(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()


print("Time to train")
