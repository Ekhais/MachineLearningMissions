"""
Fovea Localization Model for Kaggle Competition
混合方法：高斯热图回归 + 坐标精修 (Hybrid Heatmap + Coordinate Refinement)
解决Y坐标预测瓶颈问题
"""

import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
import math
import random
import warnings
warnings.filterwarnings('ignore')

# Set device and enable GPU optimizations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

# Paths
BASE_PATH = '/kaggle/input/mission4/detection'
TRAIN_IMG_PATH = os.path.join(BASE_PATH, 'train')
TEST_IMG_PATH = os.path.join(BASE_PATH, 'test')
TRAIN_LOCATION_PATH = os.path.join(BASE_PATH, 'train_location')
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'fovea_localization_train_GT.csv')
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, 'sample_submission.csv')

# Hyperparameters - 优化后
IMG_SIZE = 448  # 更大的输入图像以获得更多细节
HEATMAP_SIZE = 112  # 更高分辨率热图 (IMG_SIZE / 4)
SIGMA = 4  # 更小的sigma使高斯分布更集中
BATCH_SIZE = 6
EPOCHS = 300
LEARNING_RATE = 2e-4
MIN_LR = 1e-7
EARLY_STOP_PATIENCE = 80
WEIGHT_DECAY = 5e-5
USE_MIXED_PRECISION = True
NUM_WORKERS = 2
WARMUP_EPOCHS = 8
VAL_SPLIT = 0.15


def generate_gaussian_heatmap(center_x, center_y, heatmap_size, sigma):
    """
    生成以(center_x, center_y)为中心的高斯热图
    center_x, center_y: 归一化坐标 [0, 1]
    """
    x = np.arange(0, heatmap_size, dtype=np.float32)
    y = np.arange(0, heatmap_size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # Convert normalized coordinates to heatmap coordinates
    cx = center_x * heatmap_size
    cy = center_y * heatmap_size

    # Gaussian distribution
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    return heatmap


def get_coords_from_heatmap(heatmap, orig_w, orig_h, temperature=20.0):
    """
    从热图中提取坐标，使用改进的soft-argmax方法
    更高的temperature使分布更加集中
    """
    heatmap = heatmap.squeeze()
    h, w = heatmap.shape
    device = heatmap.device

    # Apply softmax with higher temperature for sharper distribution
    heatmap_flat = heatmap.reshape(-1)
    heatmap_softmax = F.softmax(heatmap_flat * temperature, dim=0)
    heatmap_softmax = heatmap_softmax.reshape(h, w)

    # Create coordinate grids
    x_coords = torch.arange(w, dtype=torch.float32, device=device)
    y_coords = torch.arange(h, dtype=torch.float32, device=device)

    # Soft-argmax with improved weighting
    x = (heatmap_softmax.sum(dim=0) * x_coords).sum()
    y = (heatmap_softmax.sum(dim=1) * y_coords).sum()

    # Normalize and convert to original image coordinates
    x_norm = x / w
    y_norm = y / h

    x_pixel = x_norm * orig_w
    y_pixel = y_norm * orig_h

    return x_pixel, y_pixel


class FoveaHeatmapDataset(Dataset):
    """Dataset that generates Gaussian heatmaps as targets with coordinate labels"""

    def __init__(self, image_paths, labels=None, img_sizes=None, transform=None,
                 img_size=448, heatmap_size=112, sigma=4, is_train=False):
        self.image_paths = image_paths
        self.labels = labels
        self.img_sizes = img_sizes
        self.transform = transform
        self.img_size = img_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.is_train = is_train

    def __len__(self):
        return len(self.image_paths)

    def apply_augmentation(self, image, label, orig_w, orig_h):
        """Apply augmentations optimized for macula detection"""

        # 1. Random horizontal flip (50% chance)
        if random.random() < 0.5:
            image = cv2.flip(image, 1)
            label[0] = orig_w - label[0]

        # 2. Random scale (92% - 108%) - more conservative
        if random.random() < 0.5:
            scale = random.uniform(0.92, 1.08)
            new_w, new_h = int(orig_w * scale), int(orig_h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            label[0] *= scale
            label[1] *= scale
            orig_w, orig_h = new_w, new_h

        # 3. Random small rotation (-8 to 8 degrees) - more conservative
        if random.random() < 0.4:
            angle = random.uniform(-8, 8)
            center = (orig_w / 2, orig_h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, M, (orig_w, orig_h), borderMode=cv2.BORDER_REFLECT)
            rad = math.radians(-angle)
            cos_a, sin_a = math.cos(rad), math.sin(rad)
            cx, cy = center
            x, y = label[0] - cx, label[1] - cy
            label[0] = x * cos_a - y * sin_a + cx
            label[1] = x * sin_a + y * cos_a + cy

        # 4. Random translation (-3% to 3%) - more conservative for Y
        if random.random() < 0.3:
            tx = random.uniform(-0.03, 0.03) * orig_w
            ty = random.uniform(-0.02, 0.02) * orig_h  # Smaller for Y
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (orig_w, orig_h), borderMode=cv2.BORDER_REFLECT)
            label[0] += tx
            label[1] += ty

        # 5. Brightness/Contrast adjustment
        if random.random() < 0.5:
            alpha = random.uniform(0.85, 1.15)
            beta = random.randint(-15, 15)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # 6. CLAHE for edge enhancement
        if random.random() < 0.4:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=random.uniform(1.5, 2.5), tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        # 7. Gaussian blur (mild)
        if random.random() < 0.2:
            ksize = random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # Ensure label is within bounds
        label[0] = np.clip(label[0], 0, orig_w - 1)
        label[1] = np.clip(label[1], 0, orig_h - 1)

        return image, label, orig_w, orig_h

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        if self.labels is not None:
            label = self.labels[idx].copy()

            if self.is_train:
                image, label, orig_w, orig_h = self.apply_augmentation(image, label, orig_w, orig_h)

            # Resize image
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

            if self.transform:
                image = self.transform(image)

            # Normalize coordinates to [0, 1] range
            norm_x = label[0] / orig_w
            norm_y = label[1] / orig_h

            # Generate Gaussian heatmap
            heatmap = generate_gaussian_heatmap(norm_x, norm_y, self.heatmap_size, self.sigma)
            heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

            # Also return normalized coordinates for hybrid loss
            coords = torch.tensor([norm_x, norm_y], dtype=torch.float32)

            return image, heatmap, coords, torch.tensor([orig_w, orig_h], dtype=torch.float32)
        else:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            if self.transform:
                image = self.transform(image)
            return image, torch.tensor([orig_w, orig_h], dtype=torch.float32)


class CoordinateAttention(nn.Module):
    """Coordinate Attention module for better spatial awareness"""
    def __init__(self, in_channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mid_channels = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, in_channels, 1)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, 1)

    def forward(self, x):
        identity = x
        n, c, h, w = x.shape

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        return identity * a_h * a_w


class HeatmapDecoderWithCoordRefine(nn.Module):
    """Decoder head with heatmap generation and coordinate refinement"""

    def __init__(self, in_channels, heatmap_size):
        super().__init__()
        self.heatmap_size = heatmap_size

        # Main decoder path
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            CoordinateAttention(512, reduction=16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            CoordinateAttention(256, reduction=16),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Heatmap output
        self.heatmap_conv = nn.Conv2d(64, 1, 1)

        # Coordinate refinement head - predicts offset from heatmap center
        self.coord_refine = nn.Sequential(
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # Output: (dx, dy) offset
            nn.Tanh()  # Bound offset to [-1, 1]
        )

    def forward(self, x):
        features = self.decoder(x)

        # Heatmap branch
        heatmap = self.heatmap_conv(features)
        heatmap = F.interpolate(heatmap, size=(self.heatmap_size, self.heatmap_size),
                               mode='bilinear', align_corners=False)

        # Coordinate refinement branch
        coord_offset = self.coord_refine(features)

        return heatmap, coord_offset


class FoveaHybridNet(nn.Module):
    """Hybrid model: ResNet backbone + Heatmap + Coordinate Refinement"""

    def __init__(self, pretrained=True, heatmap_size=112):
        super().__init__()

        # Use ResNet50 backbone
        resnet = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )

        # Freeze early layers (only train layer3, layer4)
        for name, child in self.backbone.named_children():
            if name in ['0', '1', '2', '3', '4', '5']:  # conv1, bn1, relu, maxpool, layer1, layer2
                for param in child.parameters():
                    param.requires_grad = False

        # Hybrid decoder head
        self.head = HeatmapDecoderWithCoordRefine(2048, heatmap_size)
        self.heatmap_size = heatmap_size

    def forward(self, x):
        features = self.backbone(x)
        heatmap, coord_offset = self.head(features)
        return heatmap, coord_offset

    def get_refined_coords(self, heatmap, coord_offset, orig_w, orig_h):
        """Get final coordinates by combining heatmap and refinement"""
        batch_size = heatmap.shape[0]
        results = []

        for i in range(batch_size):
            # Get base coordinates from heatmap
            base_x, base_y = get_coords_from_heatmap(heatmap[i], orig_w[i], orig_h[i])

            # Apply refinement offset (scaled by a small factor)
            offset_scale = 30.0  # Maximum offset in pixels
            dx = coord_offset[i, 0] * offset_scale
            dy = coord_offset[i, 1] * offset_scale

            refined_x = base_x + dx
            refined_y = base_y + dy

            # Clamp to image bounds
            refined_x = torch.clamp(refined_x, 0, orig_w[i] - 1)
            refined_y = torch.clamp(refined_y, 0, orig_h[i] - 1)

            results.append([refined_x, refined_y])

        return results


# Data transforms
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class HybridLoss(nn.Module):
    """Combined loss: Heatmap MSE + Coordinate MSE with adaptive weighting"""

    def __init__(self, heatmap_weight=1.0, coord_weight=2.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred_heatmap, target_heatmap, pred_offset, target_coords, heatmap_size):
        # 1. Weighted heatmap loss
        heatmap_loss = self.mse(pred_heatmap, target_heatmap)
        # Weight more heavily near the peak
        weight = 1.0 + target_heatmap * 5.0
        heatmap_loss = (heatmap_loss * weight).mean()

        # 2. Coordinate refinement loss
        # Get predicted coords from heatmap
        batch_size = pred_heatmap.shape[0]
        pred_coords = []
        for i in range(batch_size):
            hm = pred_heatmap[i].squeeze()
            h, w = hm.shape
            hm_flat = hm.reshape(-1)
            hm_softmax = F.softmax(hm_flat * 20, dim=0).reshape(h, w)

            x_coords = torch.arange(w, dtype=torch.float32, device=hm.device)
            y_coords = torch.arange(h, dtype=torch.float32, device=hm.device)

            x = (hm_softmax.sum(dim=0) * x_coords).sum() / w
            y = (hm_softmax.sum(dim=1) * y_coords).sum() / h
            pred_coords.append(torch.stack([x, y]))

        pred_coords = torch.stack(pred_coords)

        # Combined coordinate: heatmap prediction + offset
        offset_scale = 0.03  # Scale offset relative to image size
        final_pred = pred_coords + pred_offset * offset_scale

        coord_loss = F.mse_loss(final_pred, target_coords)

        # 3. Offset regularization (prefer small offsets)
        offset_reg = (pred_offset ** 2).mean() * 0.1

        total_loss = self.heatmap_weight * heatmap_loss + self.coord_weight * coord_loss + offset_reg

        return total_loss, heatmap_loss, coord_loss


def load_training_data():
    df = pd.read_csv(TRAIN_CSV_PATH)
    image_paths = []
    labels = []
    img_sizes = []

    for _, row in df.iterrows():
        img_id = str(int(row['data'])).zfill(4)
        img_path = os.path.join(TRAIN_IMG_PATH, f'{img_id}.jpg')

        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                h, w = img.shape[:2]
                image_paths.append(img_path)
                labels.append([row['Fovea_X'], row['Fovea_Y']])
                img_sizes.append([w, h])

    return image_paths, np.array(labels), img_sizes


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    total_loss = 0
    batch_count = 0

    for images, heatmaps, coords, _ in dataloader:
        images = images.to(device)
        heatmaps = heatmaps.to(device)
        coords = coords.to(device)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                pred_heatmap, pred_offset = model(images)
                loss, _, _ = criterion(pred_heatmap, heatmaps, pred_offset, coords, HEATMAP_SIZE)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred_heatmap, pred_offset = model(images)
            loss, _, _ = criterion(pred_heatmap, heatmaps, pred_offset, coords, HEATMAP_SIZE)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        batch_count += 1

    return total_loss / batch_count


def validate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0
    batch_count = 0

    all_preds_pixel = []
    all_labels_pixel = []

    with torch.no_grad():
        for images, heatmaps, coords, sizes in dataloader:
            images = images.to(device)
            heatmaps_device = heatmaps.to(device)
            coords_device = coords.to(device)

            if use_amp:
                with autocast():
                    pred_heatmap, pred_offset = model(images)
                    loss, _, _ = criterion(pred_heatmap, heatmaps_device, pred_offset, coords_device, HEATMAP_SIZE)
            else:
                pred_heatmap, pred_offset = model(images)
                loss, _, _ = criterion(pred_heatmap, heatmaps_device, pred_offset, coords_device, HEATMAP_SIZE)

            total_loss += loss.item()
            batch_count += 1

            # Extract coordinates
            for i in range(pred_heatmap.shape[0]):
                orig_w, orig_h = sizes[i, 0].item(), sizes[i, 1].item()

                # Get refined coordinates
                pred_x, pred_y = get_coords_from_heatmap(pred_heatmap[i], orig_w, orig_h)

                # Apply offset refinement
                offset_scale = 30.0
                dx = pred_offset[i, 0].item() * offset_scale
                dy = pred_offset[i, 1].item() * offset_scale

                pred_x = np.clip(pred_x.cpu().item() + dx, 0, orig_w - 1)
                pred_y = np.clip(pred_y.cpu().item() + dy, 0, orig_h - 1)

                # Get target coordinates
                target_x, target_y = get_coords_from_heatmap(heatmaps[i].to(device), orig_w, orig_h)

                all_preds_pixel.append([pred_x, pred_y])
                all_labels_pixel.append([target_x.cpu().item(), target_y.cpu().item()])

    all_preds_pixel = np.array(all_preds_pixel)
    all_labels_pixel = np.array(all_labels_pixel)

    mse_x = np.mean((all_preds_pixel[:, 0] - all_labels_pixel[:, 0]) ** 2)
    mse_y = np.mean((all_preds_pixel[:, 1] - all_labels_pixel[:, 1]) ** 2)
    real_mse = (mse_x + mse_y) / 2.0

    distances = np.sqrt(np.sum((all_preds_pixel - all_labels_pixel) ** 2, axis=1))
    avg_distance = np.mean(distances)

    avg_loss = total_loss / batch_count

    print(f"  Val Loss: {avg_loss:.6f}")
    print(f"  📊 MSE (pixels²): {real_mse:.2f} | MSE_X: {mse_x:.2f}, MSE_Y: {mse_y:.2f}")
    print(f"  📏 Avg Distance: {avg_distance:.2f} pixels")
    print(f"  🔍 Pred range: X=[{all_preds_pixel[:, 0].min():.1f}-{all_preds_pixel[:, 0].max():.1f}], Y=[{all_preds_pixel[:, 1].min():.1f}-{all_preds_pixel[:, 1].max():.1f}]")
    print(f"  🔍 Label range: X=[{all_labels_pixel[:, 0].min():.1f}-{all_labels_pixel[:, 0].max():.1f}], Y=[{all_labels_pixel[:, 1].min():.1f}-{all_labels_pixel[:, 1].max():.1f}]")

    return avg_loss, real_mse


# Main training code
print("Loading training data...")
train_paths, train_labels, train_sizes = load_training_data()
print(f"Total training images: {len(train_paths)}")

print(f"\nLabel statistics:")
print(f"  Fovea_X: min={train_labels[:, 0].min():.2f}, max={train_labels[:, 0].max():.2f}, mean={train_labels[:, 0].mean():.2f}")
print(f"  Fovea_Y: min={train_labels[:, 1].min():.2f}, max={train_labels[:, 1].max():.2f}, mean={train_labels[:, 1].mean():.2f}")

# Split data
indices = np.arange(len(train_paths))
train_idx, val_idx = train_test_split(indices, test_size=VAL_SPLIT, random_state=42)

train_paths_split = [train_paths[i] for i in train_idx]
val_paths_split = [train_paths[i] for i in val_idx]
train_labels_split = train_labels[train_idx]
val_labels_split = train_labels[val_idx]
train_sizes_split = [train_sizes[i] for i in train_idx]
val_sizes_split = [train_sizes[i] for i in val_idx]

print(f"\nTraining set: {len(train_paths_split)} images")
print(f"Validation set: {len(val_paths_split)} images")

# Create datasets
train_dataset = FoveaHeatmapDataset(
    train_paths_split, train_labels_split, train_sizes_split,
    transform=train_transform, img_size=IMG_SIZE, heatmap_size=HEATMAP_SIZE,
    sigma=SIGMA, is_train=True
)
val_dataset = FoveaHeatmapDataset(
    val_paths_split, val_labels_split, val_sizes_split,
    transform=test_transform, img_size=IMG_SIZE, heatmap_size=HEATMAP_SIZE,
    sigma=SIGMA, is_train=False
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=NUM_WORKERS, pin_memory=True)

# Initialize model
model = FoveaHybridNet(pretrained=True, heatmap_size=HEATMAP_SIZE).to(device)
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: ResNet50 + Hybrid Heatmap + Coord Refinement")
print(f"Trainable params: {trainable_params:,} / {total_params:,}")
print(f"Heatmap size: {HEATMAP_SIZE}x{HEATMAP_SIZE}, Sigma: {SIGMA}")

# Loss function
criterion = HybridLoss(heatmap_weight=1.0, coord_weight=2.0)

# Optimizer with different LR for backbone and head
backbone_params = []
head_params = []
for name, param in model.named_parameters():
    if param.requires_grad:
        if 'head' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

optimizer = optim.AdamW([
    {'params': backbone_params, 'lr': LEARNING_RATE * 0.1, 'initial_lr': LEARNING_RATE * 0.1},
    {'params': head_params, 'lr': LEARNING_RATE, 'initial_lr': LEARNING_RATE}
], weight_decay=WEIGHT_DECAY)

# Cosine annealing with warm restarts - longer cycles
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2, eta_min=MIN_LR)

# Mixed precision
scaler = GradScaler() if USE_MIXED_PRECISION else None
print(f"Mixed Precision: {'Enabled' if USE_MIXED_PRECISION else 'Disabled'}")

print(f"\n{'='*60}")
print("🚀 Starting training with Hybrid Heatmap + Coord Refinement...")
print(f"{'='*60}")

best_mse = float('inf')
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    print(f'\n{"="*60}')
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'{"="*60}')

    # Warmup learning rate
    if epoch < WARMUP_EPOCHS:
        warmup_factor = (epoch + 1) / WARMUP_EPOCHS
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * warmup_factor
        print(f"  🔥 Warmup: LR factor = {warmup_factor:.2f}")

    train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, USE_MIXED_PRECISION)
    print(f"  Train Loss: {train_loss:.6f}")

    val_loss, val_mse = validate(model, val_loader, criterion, device, USE_MIXED_PRECISION)

    # Update scheduler after warmup
    if epoch >= WARMUP_EPOCHS:
        scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']
    print(f"  Learning Rate: {current_lr:.6e}")

    # Save best model
    if val_mse < best_mse:
        best_mse = val_mse
        epochs_without_improvement = 0
        torch.save(model.state_dict(), 'best_fovea_model.pth')
        print(f"  ✅ Best model saved! MSE: {val_mse:.2f}")
    else:
        epochs_without_improvement += 1
        print(f"  ⚠️ No improvement for {epochs_without_improvement} epochs (Best: {best_mse:.2f})")

    if epochs_without_improvement >= EARLY_STOP_PATIENCE:
        print(f'\n🛑 Early stopping after {epoch+1} epochs')
        break

print(f"\n{'='*60}")
print(f"✅ Training completed! Best MSE: {best_mse:.2f}")
print(f"{'='*60}")

# Load best model and generate predictions with TTA
print("\nGenerating predictions on test set (with TTA)...")
model.load_state_dict(torch.load('best_fovea_model.pth'))
model.eval()

test_images = sorted([f for f in os.listdir(TEST_IMG_PATH) if f.endswith('.jpg')])
test_paths = [os.path.join(TEST_IMG_PATH, img) for img in test_images]

predictions = []

with torch.no_grad():
    for img_path in test_paths:
        image = cv2.imread(img_path)
        orig_h, orig_w = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        all_preds_x = []
        all_preds_y = []

        # Original prediction
        img_resized = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        img_tensor = test_transform(img_resized).unsqueeze(0).to(device)
        heatmap1, offset1 = model(img_tensor)
        x1, y1 = get_coords_from_heatmap(heatmap1[0], orig_w, orig_h)
        x1 = x1.cpu().item() + offset1[0, 0].item() * 30.0
        y1 = y1.cpu().item() + offset1[0, 1].item() * 30.0
        all_preds_x.append(x1)
        all_preds_y.append(y1)

        # Flipped prediction (TTA)
        img_flipped = cv2.flip(image, 1)
        img_flipped_resized = cv2.resize(img_flipped, (IMG_SIZE, IMG_SIZE))
        img_flipped_tensor = test_transform(img_flipped_resized).unsqueeze(0).to(device)
        heatmap2, offset2 = model(img_flipped_tensor)
        heatmap2_flipped = torch.flip(heatmap2, dims=[3])
        x2, y2 = get_coords_from_heatmap(heatmap2_flipped[0], orig_w, orig_h)
        x2 = x2.cpu().item() - offset2[0, 0].item() * 30.0  # Flip offset sign
        y2 = y2.cpu().item() + offset2[0, 1].item() * 30.0
        all_preds_x.append(x2)
        all_preds_y.append(y2)

        # Multi-scale TTA
        for scale in [0.9, 1.1]:
            scaled_size = int(IMG_SIZE * scale)
            img_scaled = cv2.resize(image, (scaled_size, scaled_size))
            # Center crop or pad to IMG_SIZE
            if scale > 1:
                start = (scaled_size - IMG_SIZE) // 2
                img_cropped = img_scaled[start:start+IMG_SIZE, start:start+IMG_SIZE]
            else:
                pad = (IMG_SIZE - scaled_size) // 2
                img_cropped = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
                img_cropped[pad:pad+scaled_size, pad:pad+scaled_size] = img_scaled

            img_tensor = test_transform(img_cropped).unsqueeze(0).to(device)
            heatmap, offset = model(img_tensor)
            x, y = get_coords_from_heatmap(heatmap[0], orig_w, orig_h)
            x = x.cpu().item() + offset[0, 0].item() * 30.0
            y = y.cpu().item() + offset[0, 1].item() * 30.0
            all_preds_x.append(x)
            all_preds_y.append(y)

        # Average predictions
        pred_x = np.clip(np.mean(all_preds_x), 0, orig_w - 1)
        pred_y = np.clip(np.mean(all_preds_y), 0, orig_h - 1)
        predictions.append([pred_x, pred_y])

# Create submission file
submission_df = pd.read_csv(SAMPLE_SUB_PATH)

for i, (x, y) in enumerate(predictions):
    img_id = int(test_images[i].split('.')[0])

    x_idx = submission_df[submission_df['ImageID'] == f'{img_id}_Fovea_X'].index[0]
    submission_df.loc[x_idx, 'value'] = x

    y_idx = submission_df[submission_df['ImageID'] == f'{img_id}_Fovea_Y'].index[0]
    submission_df.loc[y_idx, 'value'] = y

submission_df.to_csv('submission.csv', index=False)
print("\n✅ Submission saved to submission.csv")
print(submission_df.head(10))