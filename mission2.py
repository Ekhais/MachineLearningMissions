# deep_plant_classification.py
import os
import random
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
from torchvision import transforms, models

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# -------------------------
# Config
# -------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
BATCH_SIZE = 32
EPOCHS = 12           # 每个 fold 训练轮次（数据少时不需要很多 epoch）
IMG_SIZE = 224
N_FOLDS = 5
NUM_WORKERS = 4
DATA_ROOT = "/kaggle/input/streetgrandfasuccess/neu-plant-seedling-classification-num2-2025/dataset-for-task2/dataset-for-task2"  # 修改为你的路径
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")
OUTPUT_CSV = "/kaggle/working/submission.csv"
MODEL_SAVE_DIR = "/kaggle/working/models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# -------------------------
# Reproducibility
# -------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)
seed_everything(SEED)

# -------------------------
# Dataset
# -------------------------
def make_image_list(train_dir):
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    image_paths = []
    labels = []
    for cls in classes:
        for p in glob(os.path.join(train_dir, cls, "*.png")):
            image_paths.append(p)
            labels.append(cls)
    return image_paths, labels, classes

class PlantDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is None:
            return img, os.path.basename(p)  # for test: return image and filename
        else:
            return img, self.labels[idx]

# -------------------------
# Transforms
# -------------------------
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    transforms.RandomErasing(p=0.2)
])

valid_transform = transforms.Compose([
    transforms.Resize(int(IMG_SIZE*1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

test_transform = valid_transform

# -------------------------
# Model factory
# -------------------------
def build_model(num_classes=NUM_CLASSES, pretrained=True):
    model = models.resnet34(pretrained=pretrained)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

# -------------------------
# Train & Validate functions
# -------------------------
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    preds_list = []
    targets_list = []
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)
        out = model(imgs)
        loss = criterion(out, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1).detach().cpu().numpy()
        preds_list.append(preds)
        targets_list.append(targets.detach().cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)
    f1 = f1_score(targets, preds, average='micro')
    return avg_loss, f1

def validate_one_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, targets)
            running_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1).detach().cpu().numpy()
            preds_list.append(preds)
            targets_list.append(targets.detach().cpu().numpy())
    avg_loss = running_loss / len(loader.dataset)
    preds = np.concatenate(preds_list)
    targets = np.concatenate(targets_list)
    f1 = f1_score(targets, preds, average='micro')
    return avg_loss, f1, preds, targets

# -------------------------
# Main CV training
# -------------------------
def run_training_cv(image_paths, labels, classes):
    le = LabelEncoder()
    y_enc = le.fit_transform(labels)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    fold_results = {}
    for fold, (train_idx, val_idx) in enumerate(skf.split(image_paths, y_enc)):
        print(f"\n=== Fold {fold+1}/{N_FOLDS} ===")
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        train_labels = [y_enc[i] for i in train_idx]
        val_labels = [y_enc[i] for i in val_idx]

        train_ds = PlantDataset(train_paths, train_labels, transform=train_transform)
        val_ds = PlantDataset(val_paths, val_labels, transform=valid_transform)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        model = build_model(num_classes=len(classes), pretrained=True).to(DEVICE)

        # Optionally freeze backbone for first few epochs (warmup)
        # for name, param in model.named_parameters():
        #     if "fc" not in name:
        #         param.requires_grad = False

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_f1 = 0.0
        best_model_path = os.path.join(MODEL_SAVE_DIR, f"resnet34_fold{fold}.pth")
        for epoch in range(EPOCHS):
            tr_loss, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_f1, _, _ = validate_one_epoch(model, val_loader, criterion)
            scheduler.step()
            print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={tr_loss:.4f} train_f1={tr_f1:.4f} | val_loss={val_loss:.4f} val_f1={val_f1:.4f}")
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), best_model_path)
        print(f"Fold {fold} best val_f1 = {best_val_f1:.4f}")
        fold_results[fold] = {"best_val_f1": best_val_f1, "model_path": best_model_path}

    mean_f1 = np.mean([fold_results[f]["best_val_f1"] for f in fold_results])
    print(f"\nCV mean F1_micro = {mean_f1:.4f}")
    return le, fold_results, mean_f1

# -------------------------
# Final train on all data & predict test
# -------------------------
def train_final_and_predict(image_paths, labels, classes, le, best_model_path=None):
    y_enc = le.transform(labels)
    ds_all = PlantDataset(image_paths, y_enc, transform=train_transform)
    loader_all = DataLoader(ds_all, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(num_classes=len(classes), pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # If best_model_path provided, load and continue fine-tune
    if best_model_path is not None and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
        print("Loaded pretrained fold model, fine-tuning on full data...")

    for epoch in range(6):  # short fine-tune
        tr_loss, tr_f1 = train_one_epoch(model, loader_all, criterion, optimizer)
        scheduler.step()
        print(f"Final train epoch {epoch+1}/6 loss={tr_loss:.4f} f1={tr_f1:.4f}")

    # predict test
    test_paths = sorted(glob(os.path.join(TEST_DIR, "*.png")))
    test_ds = PlantDataset(test_paths, labels=None, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model.eval()
    preds_all = []
    filenames = []
    with torch.no_grad():
        for imgs, names in test_loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().numpy()
            preds_all.extend(preds)
            filenames.extend(names)

    pred_labels = le.inverse_transform(preds_all)
    # Build submission dataframe
    ids_with_ext = [f"{os.path.splitext(fn)[0]}.png" for fn in filenames]
    df = pd.DataFrame({"ID": ids_with_ext, "Category": pred_labels})
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved submission to {OUTPUT_CSV}")
    return df

# -------------------------
# Run everything
# -------------------------
if __name__ == "__main__":
    print("Collecting training images ...")
    image_paths, labels, classes = make_image_list(TRAIN_DIR)
    print(f"Found {len(image_paths)} images, classes: {classes}")

    le, fold_results, mean_f1 = run_training_cv(image_paths, labels, classes)

    # choose best fold model for final fine-tune or None
    best_fold = max(fold_results.items(), key=lambda kv: kv[1]["best_val_f1"])[0]
    best_model_path = fold_results[best_fold]["model_path"]
    print(f"Using best fold model: {best_model_path} to initialize final training.")

    df_submit = train_final_and_predict(image_paths, labels, classes, le, best_model_path=best_model_path)
    print(df_submit.head())
