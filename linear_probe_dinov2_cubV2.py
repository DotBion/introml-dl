#!/usr/bin/env python
import os
import argparse
from typing import Tuple

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import timm
import torchvision.transforms as T


# ============================================================
# Dataset
# ============================================================

class CubCsvDataset(Dataset):
    """
    Dataset for the prepared CUB-200 data.

    Works with the original CSVs from prepare_cub200_for_kaggle.py
    without modifying them:

      - train_labels.csv: ['id', 'class_id']
      - val_labels.csv:   ['id', 'class_id']
      - test_images.csv:  ['id']

    We mainly rely on column positions:
      - column 0: image id / filename
      - column 1: label (for train/val only)
    """

    def __init__(self, csv_file: str, root_dir: str, mode: str = "train",
                 transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.mode = mode  # "train", "val", or "test"
        self.transform = transform

        # Log for sanity
        print(f"[CubCsvDataset] mode={self.mode}, csv={csv_file}")
        print(f"[CubCsvDataset] columns={list(self.df.columns)}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]

        # --------- IMAGE NAME ----------
        # Prefer named columns, fall back to first column
        if "image" in self.df.columns:
            img_name = row["image"]
        elif "id" in self.df.columns:
            img_name = row["id"]
        else:
            img_name = row.iloc[0]  # first column as fallback

        img_path = os.path.join(self.root_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # --------- LABEL ----------
        if self.mode == "test":
            label = -1  # dummy
        else:
            if "label" in self.df.columns:
                label = int(row["label"])
            elif "class_id" in self.df.columns:
                label = int(row["class_id"])
            else:
                # Fallback: assume second column is label
                if len(self.df.columns) > 1:
                    label = int(row.iloc[1])
                else:
                    raise RuntimeError(
                        f"No label column found for train/val CSV "
                        f"(columns: {list(self.df.columns)})"
                    )

        return img, label


# ============================================================
# Model
# ============================================================

class LinearClassifier(nn.Module):
    """
    Linear classifier head on top of a frozen backbone.
    """

    def __init__(self, backbone: nn.Module, num_classes: int = 200):
        super().__init__()
        self.backbone = backbone

        # Infer embedding dimension on the same device as backbone
        device = next(backbone.parameters()).device
        example = torch.zeros(1, 3, 96, 96, device=device)
        with torch.no_grad():
            feat = self.backbone(example)
        if feat.ndim > 2:
            feat = feat.mean(dim=(-2, -1))
        in_dim = feat.shape[-1]

        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.backbone(x)
        if feats.ndim > 2:
            feats = feats.mean(dim=(-2, -1))
        logits = self.head(feats)
        return logits


def build_backbone(ckpt_path: str, device: torch.device) -> nn.Module:
    """
    Build a ViT-S/14 DINOv2-like backbone and load weights from checkpoint.

    Assumes your checkpoint has either:
      - a 'student' key with the model state_dict, or
      - is itself a state_dict.

    Also strips leading "0." in keys (common when checkpoint stores a module list).
    """
    print(f"Loading DINO student from: {ckpt_path}")

    # Create the backbone model (no classifier head)
    backbone = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
        img_size=96,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "student" in ckpt:
        state_dict = ckpt["student"]
    else:
        state_dict = ckpt

    # Strip leading "0." prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("0."):
            new_k = k.split(".", 1)[1]  # drop "0."
        else:
            new_k = k
        new_state_dict[new_k] = v

    missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
    print("Backbone loaded. Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    backbone.to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone


# ============================================================
# Training / Evaluation
# ============================================================

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        running_correct += (preds == y).sum().item()
        total += x.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Val", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)

            running_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            total += x.size(0)

    epoch_loss = running_loss / total
    epoch_acc = running_correct / total
    return epoch_loss, epoch_acc


def predict_test(model, loader, device, test_dataset, out_csv_path: str):
    """
    Run inference on the test set and write a Kaggle submission CSV.

    We use the original CUB ids from test_images.csv and
    produce columns: id, class_id
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Test", leave=False)
        for x, _ in pbar:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)

    # test_dataset.df first column is the original id
    if "id" in test_dataset.df.columns:
        ids = test_dataset.df["id"].tolist()
    elif "image" in test_dataset.df.columns:
        ids = test_dataset.df["image"].tolist()
    else:
        ids = test_dataset.df.iloc[:, 0].tolist()  # first column fallback

    if len(ids) != len(all_preds):
        raise RuntimeError(
            f"Mismatch between number of test ids ({len(ids)}) "
            f"and predictions ({len(all_preds)})"
        )

    out_df = pd.DataFrame({"id": ids, "class_id": all_preds})
    out_df.to_csv(out_csv_path, index=False)
    print(f"Saved submission to: {out_csv_path}")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Linear probe on CUB-200 using DINOv2 ViT-S/14 backbone"
    )
    parser.add_argument("--pretrained_ckpt", type=str, required=True,
                        help="Path to DINO checkpoint (.pth).")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="CUB data dir with train/val/test and CSVs.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints and submission.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mode", type=str, default="train_and_predict",
                        choices=["train", "predict", "train_and_predict"])
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # ----------------- Data -----------------
    train_root = os.path.join(args.data_dir, "train")
    val_root = os.path.join(args.data_dir, "val")
    test_root = os.path.join(args.data_dir, "test")

    train_csv = os.path.join(args.data_dir, "train_labels.csv")
    val_csv   = os.path.join(args.data_dir, "val_labels.csv")
    test_csv  = os.path.join(args.data_dir, "test_images.csv")

    # Transforms
    train_transform = T.Compose([
        T.Resize(96),
        T.RandomResizedCrop(96, scale=(0.8, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    eval_transform = T.Compose([
        T.Resize(96),
        T.CenterCrop(96),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CubCsvDataset(
        csv_file=train_csv,
        root_dir=train_root,
        mode="train",
        transform=train_transform,
    )
    val_dataset = CubCsvDataset(
        csv_file=val_csv,
        root_dir=val_root,
        mode="val",
        transform=eval_transform,
    )
    test_dataset = CubCsvDataset(
        csv_file=test_csv,
        root_dir=test_root,
        mode="test",
        transform=eval_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ----------------- Model -----------------
    backbone = build_backbone(args.pretrained_ckpt, device)
    model = LinearClassifier(backbone, num_classes=200).to(device)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_acc = 0.0
    best_ckpt_path = os.path.join(args.output_dir, "linear_probe_best.pth")

    # ----------------- Training -----------------
    if args.mode in ("train", "train_and_predict"):
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")

            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            print(f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f}")

            val_loss, val_acc = evaluate(model, val_loader, device)
            print(f"Val   loss: {val_loss:.4f}, acc: {val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_acc": best_val_acc,
                    },
                    best_ckpt_path,
                )
                print(f"New best val acc: {best_val_acc:.4f} -> saved {best_ckpt_path}")

    # ----------------- Prediction on test -----------------
    if args.mode in ("predict", "train_and_predict"):
        if os.path.exists(best_ckpt_path):
            print(f"Loading best linear head from {best_ckpt_path}")
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state"])
        else:
            print("Warning: best_ckpt_path not found, using current model weights.")

        submission_path = os.path.join(args.output_dir, "submission_linear_dinov2.csv")
        predict_test(model, test_loader, device, test_dataset, submission_path)


if __name__ == "__main__":
    main()
