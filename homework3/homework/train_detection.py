import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from homework.models import load_model, save_model
from homework.metrics import DetectionMetric
from homework.datasets.road_dataset import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "detector",
    num_epoch: int = 40,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    num_workers: int = 2,
    weight_decay: float = 1e-4,
    **kwargs,
):
    # -------------------------
    # device
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # seed
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
        "drive_data/train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val_data = load_data(
        "drive_data/val",
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # -------------------------
    # loss functions
    # -------------------------
    
    seg_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, 5.0, 5.0], device=device)
    )

    depth_criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=8,
        gamma=0.5,
    )


    lambda_depth = 0.15

    global_step = 0
    best_score = -1.0

    train_metric = DetectionMetric()
    val_metric = DetectionMetric()

    # -------------------------
    # training loop
    # -------------------------
    for epoch in range(num_epoch):
        model.train()
        train_metric.reset()

        for batch in train_data:
            image = batch["image"].to(device)
            depth = batch["depth"].to(device)
            track = batch["track"].to(device)

            optimizer.zero_grad()

            seg_logits, pred_depth = model(image)

            seg_loss = seg_criterion(seg_logits, track)
            depth_loss = depth_criterion(pred_depth, depth)

            loss = seg_loss + lambda_depth * depth_loss
            ##loss = seg_loss + 0.5 * depth_loss

            loss.backward()
            optimizer.step()

            preds = seg_logits.argmax(dim=1)

            train_metric.add(
                preds,
                track,
                pred_depth.detach(),
                depth,
            )

            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        train_results = train_metric.compute()

        # -------------------------
        # validation
        # -------------------------
        model.eval()
        val_metric.reset()

        with torch.inference_mode():
            for batch in val_data:
                image = batch["image"].to(device)
                depth = batch["depth"].to(device)
                track = batch["track"].to(device)

                seg_logits, pred_depth = model(image)

                preds = seg_logits.argmax(dim=1)

                val_metric.add(
                    preds,
                    track,
                    pred_depth,
                    depth,
                )

        val_results = val_metric.compute()

        # -------------------------
        # logging
        # -------------------------
        logger.add_scalar("epoch/train_iou", train_results["iou"], epoch)
        logger.add_scalar("epoch/val_iou", val_results["iou"], epoch)

        logger.add_scalar("epoch/val_depth_error", val_results["abs_depth_error"], epoch)
        logger.add_scalar("epoch/val_tp_depth_error", val_results["tp_depth_error"], epoch)

        print(
            f"Epoch {epoch+1:2d} | "
            f"IoU={val_results['iou']:.4f} | "
            f"Depth={val_results['abs_depth_error']:.4f} | "
            f"LaneDepth={val_results['tp_depth_error']:.4f}"
        )

        # -------------------------
        # save best model
        # -------------------------
        score = val_results["iou"]

        if score > best_score:
            best_score = score
            save_model(model)
            torch.save(model.state_dict(), log_dir / f"{model_name}_best.th")
            print("Saved best model")

        scheduler.step()
    logger.close()
    print("Best IoU:", best_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="detector")
    parser.add_argument("--num_epoch", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=2)

    train(**vars(parser.parse_args()))