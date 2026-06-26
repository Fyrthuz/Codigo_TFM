"""Training script for the 2D UNet on LGG MRI dataset."""
import argparse
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm

from src.models.unet import UNet
from src.utils.dataset import LGGSegmentationDataset, split_by_patient, save_test_indices


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, indices, augment=True):
        self.base = base_dataset
        self.indices = indices
        self.augment = augment
        self.train_tfm = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.base[self.indices[idx]]
        if self.augment:
            seed = torch.randint(0, 2**31, (1,)).item()
            torch.manual_seed(seed)
            img = self.train_tfm(img)
            torch.manual_seed(seed)
            mask = self.train_tfm(mask)
        return img, mask


def dice_loss(preds, targets, smooth=1.0):
    preds = torch.sigmoid(preds)
    intersection = (preds * targets).sum()
    return 1 - (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


def combined_loss(preds, targets, bce_weight=0.5):
    bce = nn.BCEWithLogitsLoss()(preds, targets)
    dice = dice_loss(preds, targets)
    return bce_weight * bce + (1 - bce_weight) * dice


def compute_iou(preds, targets, threshold=0.5):
    preds = torch.sigmoid(preds)
    preds = (preds > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection / union).item() if union > 0 else 1.0


def validate(model, dataloader, device):
    model.eval()
    val_loss, val_iou = 0.0, 0.0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = combined_loss(outputs, targets)
            val_loss += loss.item()
            val_iou += compute_iou(outputs, targets)
    model.train()
    return val_loss / len(dataloader), val_iou / len(dataloader)


def train(model, train_loader, val_loader, optimizer, device, num_epochs=60, patience=15):
    best_val_iou = 0.0
    best_state = None
    wait = 0

    for epoch in range(num_epochs):
        running_loss, running_iou = 0.0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = combined_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_iou += compute_iou(outputs, targets)
            pbar.set_postfix(loss=running_loss / (pbar.n + 1), iou=running_iou / (pbar.n + 1))

        val_loss, val_iou = validate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {running_loss/len(train_loader):.4f}, "
              f"Train IoU: {running_iou/len(train_loader):.4f}, "
              f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best Val IoU: {best_val_iou:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train UNet on LGG MRI dataset")
    parser.add_argument("--data-root", type=str, default="./MRI/filtered_data")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save-path", type=str, default="unet_model.pth")
    parser.add_argument("--test-indices", type=str, default="test_indices.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNet(in_channels=3, out_channels=1, init_features=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    base_dataset = LGGSegmentationDataset(args.data_root)

    # Patient-level split
    train_idx, val_idx, test_idx = split_by_patient(
        base_dataset.patient_ids, train_ratio=0.70, val_ratio=0.15, seed=42
    )
    print(f"Patients: {len(set(base_dataset.patient_ids))} total")
    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples, Test: {len(test_idx)} samples")

    save_test_indices(test_idx, args.test_indices)
    print(f"Test indices saved to {args.test_indices}")

    train_ds = AugmentedDataset(base_dataset, train_idx, augment=True)
    val_ds = AugmentedDataset(base_dataset, val_idx, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    train(model, train_loader, val_loader, optimizer, device, num_epochs=args.epochs)
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")

    # Evaluate on test set
    test_ds = AugmentedDataset(base_dataset, test_idx, augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loss, test_iou = validate(model, test_loader, device)
    print(f"\nTest set — Loss: {test_loss:.4f}, IoU: {test_iou:.4f}")


if __name__ == "__main__":
    main()
