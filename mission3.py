"""
快速优化版 - 大幅提速，保持精度
主要改进：
1. 使用 ResNet34 替代 EfficientNet-B3 (更快，参数少)
2. 简化分类器 (直接 512→6)
3. 移除复杂预处理 (CLAHE等)
4. 减少 epoch 到 25
5. TTA 从 5 降到 3
6. 增加 batch_size 到 128
7. 优化数据增强策略

预计训练时间: 1-1.5小时 (原来4小时)
目标准确率: 55-60%
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from collections import Counter
import torch.nn.functional as F

# ==================== Kaggle环境配置 ====================
KAGGLE_INPUT_PATH = "/kaggle/input/mission3"
TRAIN_DIR = os.path.join(KAGGLE_INPUT_PATH, "fer_data/fer_data/train")
TEST_DIR = os.path.join(KAGGLE_INPUT_PATH, "fer_data/fer_data/test")
OUTPUT_DIR = "/kaggle/working"

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

set_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

emotion_dict = {
    'Angry': 0, 'Fear': 1, 'Happy': 2, 'Sad': 3, 'Surprise': 4, 'Neutral': 5
}

# ==================== MixUp数据增强 ====================
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ==================== 数据集类（简化版） ====================
class FERDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.images = []
        self.labels = []

        if is_train:
            for emotion_name, emotion_label in emotion_dict.items():
                emotion_dir = os.path.join(root_dir, emotion_name)
                if os.path.exists(emotion_dir):
                    for img_name in os.listdir(emotion_dir):
                        if img_name.endswith('.jpg'):
                            self.images.append(os.path.join(emotion_dir, img_name))
                            self.labels.append(emotion_label)
        else:
            for img_name in os.listdir(root_dir):
                if img_name.endswith('.jpg'):
                    self.images.append(os.path.join(root_dir, img_name))
                    self.labels.append(0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            return image, self.labels[idx]
        else:
            return image, os.path.basename(img_path)

    def get_labels(self):
        return self.labels

# ==================== 优化的数据增强 ====================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),  # 减少到10度
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),  # 减轻
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # 减轻
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==================== 轻量级模型（ResNet34） ====================
class FastEmotionClassifier(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(FastEmotionClassifier, self).__init__()

        # 使用 ResNet34 (比 ResNet50/EfficientNet-B3 快得多)
        backbone = models.resnet34(pretrained=pretrained)

        # 移除最后的全连接层和池化层
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # 自适应池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # 简化的分类器 (512 -> 6, 无中间层)
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==================== Label Smoothing ====================
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        loss = -log_preds.sum(dim=-1).mean()
        nll = F.nll_loss(log_preds, target, reduction='mean')
        return self.smoothing * loss / n_classes + (1 - self.smoothing) * nll

# ==================== 训练函数（优化版） ====================
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0
    best_model_wts = model.state_dict()
    patience = 7  # 降低patience
    patience_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # ==================== 训练阶段 ====================
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in tqdm(train_loader, desc='训练'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # 30% 概率使用 MixUp (降低复杂度)
            use_mixup = random.random() > 0.7

            if use_mixup:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_corrects += (lam * torch.sum(preds == labels_a.data).float() +
                                   (1 - lam) * torch.sum(preds == labels_b.data).float()).long()
            else:
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            running_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects.double() / total

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())

        print(f'训练 Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='验证'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
                val_total += inputs.size(0)

        val_loss = val_loss / val_total
        val_acc = val_corrects.double() / val_total

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

        print(f'验证 Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

        # ==================== 保存最佳模型 ====================
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = model.state_dict()
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc.item(),
                'epoch': epoch
            }, os.path.join(OUTPUT_DIR, 'best_emotion_model.pth'))
            print(f'✓ 最佳模型已保存，准确率: {best_acc:.4f}')
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'Patience: {patience_counter}/{patience}')

        if patience_counter >= patience:
            print(f'\n早停触发于第 {epoch+1} 轮')
            break

        scheduler.step(val_acc)

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'history': history
            }, os.path.join(OUTPUT_DIR, f'checkpoint_epoch_{epoch+1}.pth'))

    print(f'\n✓ 训练完成！最佳验证准确率: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model

# ==================== TTA预测（减少到3次） ====================
def predict_with_tta(model, test_loader, num_tta=3):
    model.eval()
    all_predictions = []
    image_names = None

    for tta_idx in range(num_tta):
        predictions_probs = []
        names = []

        with torch.no_grad():
            for inputs, batch_names in tqdm(test_loader, desc=f'TTA {tta_idx+1}/{num_tta}'):
                inputs = inputs.to(device)

                # 只在TTA>1时做水平翻转
                if tta_idx == 1:
                    inputs = torch.flip(inputs, dims=[3])

                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)
                predictions_probs.append(probs.cpu().numpy())
                names.extend(batch_names)

        all_predictions.append(np.concatenate(predictions_probs, axis=0))
        if image_names is None:
            image_names = names

    avg_predictions = np.mean(all_predictions, axis=0)
    final_predictions = np.argmax(avg_predictions, axis=1)
    return image_names, final_predictions

# ==================== 主程序 ====================
def main():
    print("=" * 70)
    print("快速优化版 - 面部情感分类")
    print("=" * 70)

    print("\n加载数据集...")
    full_train_dataset = FERDataset(TRAIN_DIR, transform=train_transform, is_train=True)
    labels = full_train_dataset.get_labels()
    label_counts = Counter(labels)

    print("\n类别分布:")
    for emotion_name, emotion_label in emotion_dict.items():
        print(f"  {emotion_name}: {label_counts[emotion_label]}")

    # 90/10 分割
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    test_dataset = FERDataset(TEST_DIR, transform=test_transform, is_train=False)

    print(f"\n数据集大小:")
    print(f"  训练集: {len(train_dataset)}")
    print(f"  验证集: {len(val_dataset)}")
    print(f"  测试集: {len(test_dataset)}")

    # 加权采样
    train_labels = [full_train_dataset.labels[i] for i in train_dataset.indices]
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / class_counts.get(cls, 1) for cls in range(6)}
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # 增大 batch_size 到 128 (加速训练)
    train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler,
                             num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False,
                           num_workers=2, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)

    print("\n初始化模型 (ResNet34)...")
    model = FastEmotionClassifier(num_classes=6, pretrained=True).to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")

    # Label Smoothing 损失函数
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)

    # 优化器
    optimizer = optim.AdamW([
        {'params': model.features.parameters(), 'lr': 0.0001},
        {'params': model.classifier.parameters(), 'lr': 0.001}
    ], weight_decay=0.01)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )

    print("\n开始训练...")
    print(f"  Epochs: 25")
    print(f"  Batch Size: 128")
    print(f"  预计训练时间: 1-1.5小时")
    print("=" * 70)

    model = train_model(model, train_loader, val_loader, criterion, optimizer,
                       scheduler, num_epochs=25)

    print("\n使用TTA进行测试集预测 (3次)...")
    image_names, predictions = predict_with_tta(model, test_loader, num_tta=3)

    submission_df = pd.DataFrame({'ID': image_names, 'Emotion': predictions})
    submission_df.to_csv(os.path.join(OUTPUT_DIR, 'submission.csv'), index=False)
    print(f"\n✓ 提交文件已保存: submission.csv")
    print("=" * 70)

if __name__ == '__main__':
    main()