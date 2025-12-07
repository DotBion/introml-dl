#!/usr/bin/env python
import os
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms as T
from PIL import Image
import pandas as pd
from tqdm.auto import tqdm
import timm

# -----------------------------
# DINO head + backbone
# -----------------------------
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1)
        x = self.last_layer(x)
        return x


def create_dinov2_vits14_student(out_dim=65536):
    backbone = timm.create_model(
        "vit_small_patch14_dinov2",
        pretrained=False,
        img_size=224,
        num_classes=0,
    )
    embed_dim = backbone.num_features
    head = DINOHead(embed_dim, out_dim=out_dim)
    student = nn.Sequential(backbone, head)
    return student, embed_dim

# -----------------------------
# CUB dataset using CSVs
# -----------------------------
class CubCsvDataset(Dataset):
    def __init__(self, csv_path, root_dir, is_train=True):
        """
        csv format:
          train/val: columns ['image', 'label']
          test:      column  ['image']
        image paths in csv are relative to root_dir.
        """
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.is_train = is_train

        if is_train:
            # standard augments for linear probe
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                    T.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row["image"])
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)

        if "label" in row and not pd.isna(row["label"]):
            y = int(row["label"])
            return x, y
        else:
            return x, row["image"]  # return filename/id for test


# -----------------------------
# Linear probe model
# -----------------------------
class LinearProbeModel(nn.Module):
    def __init__(self, backbone, embed_dim, num_classes=200):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            feats = self.backbone(x)  # [B, D] because num_classes=0
        logits = self.fc(feats)
        return logits


# -----------------------------
# Training / eval loops
# -----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += x.size(0)
        pbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / total
    acc = total_correct / total
    return avg_loss, acc


def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Val", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total += x.size(0)

    avg_loss = total_loss / total
    acc = total_correct / total
    return avg_loss, acc


def predict_test(model, loader, device):
    model.eval()
    all_ids = []
    all_preds = []
    with torch.no_grad():
        for x, ids in tqdm(loader, desc="Test", leave=False):
            x = x.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_ids.extend(list(ids))
            all_preds.extend(list(preds))
    return all_ids, all_preds


# -----------------------------
# Main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser("Linear probe DINOv2-S on CUB-200")
    p.add_argument("--pretrained_ckpt", type=str, required=True,
                   help="Path to DINO pretrain checkpoint (.pth) with 'student'")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to CUB data folder from prepare_cub200_for_kaggle.py")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Where to save linear-probe checkpoint and submission")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mode", type=str, choices=["train", "predict", "train_and_predict"],
                   default="train_and_predict")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Recreate student and load DINO weights
    print("Loading DINO student from:", args.pretrained_ckpt)
    student, embed_dim = create_dinov2_vits14_student(out_dim=65536)
    ckpt = torch.load(args.pretrained_ckpt, map_location=device)
    student.load_state_dict(ckpt["student"])
    backbone = student[0]  # timm ViT backbone
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # 2) Build linear probe model
    num_classes = 200
    model = LinearProbeModel(backbone, embed_dim, num_classes=num_classes).to(device)

    train_csv = os.path.join(args.data_dir, "train_labels.csv")
    val_csv = os.path.join(args.data_dir, "val_labels.csv")
    test_csv = os.path.join(args.data_dir, "test_images.csv")

    train_root = os.path.join(args.data_dir, "train")
    val_root = os.path.join(args.data_dir, "val")
    test_root = os.path.join(args.data_dir, "test")

    linear_ckpt_path = os.path.join(args.output_dir, "linear_probe_best.pth")

    if args.mode in ["train", "train_and_predict"]:
        # 3) Data loaders
        train_ds = CubCsvDataset(train_csv, train_root, is_train=True)
        val_ds = CubCsvDataset(val_csv, val_root, is_train=False)

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)

        best_val_acc = 0.0
        for epoch in range(args.epochs):
            print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
            val_loss, val_acc = eval_one_epoch(model, val_loader, device)
            print(
                f"Train loss {train_loss:.4f}, acc {train_acc:.4f} | "
                f"Val loss {val_loss:.4f}, acc {val_acc:.4f}"
            )
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(
                    {
                        "epoch": epoch,
                        "state_dict": model.state_dict(),
                        "val_acc": val_acc,
                    },
                    linear_ckpt_path,
                )
                print(f"  -> New best val acc {val_acc:.4f}, saved {linear_ckpt_path}")

    if args.mode in ["predict", "train_and_predict"]:
        # reload best linear probe
        print("Loading best linear-probe checkpoint:", linear_ckpt_path)
        ckpt_lin = torch.load(linear_ckpt_path, map_location=device)
        model.load_state_dict(ckpt_lin["state_dict"])
        model.to(device)
        model.eval()

        # test loader
        test_ds = CubCsvDataset(test_csv, test_root, is_train=False)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        ids, preds = predict_test(model, test_loader, device)

        # build submission following sample_submission.csv structure
        sample_sub = pd.read_csv(os.path.join(args.data_dir, "sample_submission.csv"))
        # assume columns: ['image', 'label']
        sub_df = sample_sub.copy()
        pred_map = {img_id: p for img_id, p in zip(ids, preds)}
        sub_df["label"] = sub_df["image"].map(pred_map)

        out_path = os.path.join(args.output_dir, "submission_linear_dinov2.csv")
        sub_df.to_csv(out_path, index=False)
        print("Wrote submission to:", out_path)


if __name__ == "__main__":
    main()
