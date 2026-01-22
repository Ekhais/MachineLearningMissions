# -*- coding: utf-8 -*-
"""
眼底图像血管分割 - U-Net模型
用于Kaggle竞赛：task-5-vesselsegmentation-2025
修复版：解决RLE编码问题 + 强化数据增强减少过拟合
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import random
from tqdm import tqdm

# ================== 设置随机种子 ==================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ================== 配置参数 ==================
class Config:
    # 数据路径
    DATA_DIR = '/kaggle/input/task-5-vesselsegmentation-2025/segmentation'
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train/image')
    TRAIN_LABEL_DIR = os.path.join(DATA_DIR, 'train/label')
    TEST_IMG_DIR = os.path.join(DATA_DIR, 'test/image')
    OUTPUT_DIR = '/kaggle/working/predictions'

    # 模型参数
    IMG_SIZE = 512
    IN_CHANNELS = 3
    OUT_CHANNELS = 1
    FEATURES = [64, 128, 256, 512]

    # 训练参数 - 减少过拟合
    BATCH_SIZE = 4  # 小batch增加随机性
    NUM_EPOCHS = 200
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3  # 增强L2正则化

    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # P100优化选项
    USE_AMP = True
    NUM_WORKERS = 2

    # 验证集比例
    VAL_SPLIT = 0.2  # 20%用于验证

config = Config()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# ================== 数据集类 ==================
class VesselDataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None, is_train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.is_train = is_train

        self.images = sorted([f for f in os.listdir(image_dir)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))],
                            key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 加载图像
        image = Image.open(img_path).convert('RGB')
        original_size = image.size  # 保存原始尺寸
        image = image.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.BILINEAR)

        if self.is_train and self.label_dir is not None:
            # 加载标签
            label_path = os.path.join(self.label_dir, img_name)
            label = Image.open(label_path).convert('L')
            label = label.resize((config.IMG_SIZE, config.IMG_SIZE), Image.Resampling.NEAREST)

            # 数据增强
            if self.transform:
                image, label = self.transform(image, label)

            # 转换为张量
            image = TF.to_tensor(image)
            label = TF.to_tensor(label)

            # ★★★ 关键修复 ★★★
            # 实际标签格式（通过检查确认）：
            # 血管=白色(255)，背景=黑色(0)
            # 我们需要：血管=1, 背景=0 (用于训练)
            # 所以：label中值接近1的是血管，接近0的是背景
            label = (label > 0.5).float()  # 血管(白色/高值) -> 1, 背景(黑色/低值) -> 0

            return image, label, img_name
        else:
            image = TF.to_tensor(image)
            return image, img_name

# ================== 强化数据增强 ==================
class StrongTrainTransform:
    """更强的数据增强以减少过拟合"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        # 随机水平翻转
        if random.random() < self.p:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # 随机垂直翻转
        if random.random() < self.p:
            image = TF.vflip(image)
            label = TF.vflip(label)

        # 随机旋转 (任意角度)
        if random.random() < self.p:
            angle = random.uniform(-45, 45)
            image = TF.rotate(image, angle, fill=0)
            label = TF.rotate(label, angle, fill=0)  # 背景填充黑色(0)

        # 90度旋转
        if random.random() < self.p:
            k = random.choice([1, 2, 3])
            image = TF.rotate(image, k * 90)
            label = TF.rotate(label, k * 90)

        # 随机裁剪后resize - 增加概率
        if random.random() < 0.5:
            scale = random.uniform(0.6, 1.0)
            new_size = int(config.IMG_SIZE * scale)
            i = random.randint(0, config.IMG_SIZE - new_size)
            j = random.randint(0, config.IMG_SIZE - new_size)
            image = TF.crop(image, i, j, new_size, new_size)
            label = TF.crop(label, i, j, new_size, new_size)
            image = TF.resize(image, [config.IMG_SIZE, config.IMG_SIZE])
            label = TF.resize(label, [config.IMG_SIZE, config.IMG_SIZE], interpolation=InterpolationMode.NEAREST)

        # 弹性变形模拟（通过仿射变换近似）
        if random.random() < 0.3:
            # 随机仿射变换
            angle = random.uniform(-10, 10)
            translate = [random.uniform(-0.1, 0.1) * config.IMG_SIZE, random.uniform(-0.1, 0.1) * config.IMG_SIZE]
            scale = random.uniform(0.9, 1.1)
            shear = [random.uniform(-5, 5)]
            image = TF.affine(image, angle=angle, translate=translate, scale=scale, shear=shear, fill=[0])
            label = TF.affine(label, angle=angle, translate=translate, scale=scale, shear=shear, fill=[0])

        # 颜色增强（仅对图像）- 更强
        if random.random() < 0.7:
            brightness_factor = random.uniform(0.6, 1.4)
            image = TF.adjust_brightness(image, brightness_factor)

        if random.random() < 0.7:
            contrast_factor = random.uniform(0.6, 1.4)
            image = TF.adjust_contrast(image, contrast_factor)

        if random.random() < 0.5:
            saturation_factor = random.uniform(0.5, 1.5)
            image = TF.adjust_saturation(image, saturation_factor)

        if random.random() < 0.3:
            hue_factor = random.uniform(-0.15, 0.15)
            image = TF.adjust_hue(image, hue_factor)

        # gamma变换
        if random.random() < 0.3:
            gamma = random.uniform(0.7, 1.5)
            image = TF.adjust_gamma(image, gamma)

        # 高斯模糊
        if random.random() < 0.3:
            kernel_size = random.choice([3, 5])
            image = TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size])

        return image, label

# ================== U-Net模型（带Dropout减少过拟合） ==================
class DoubleConv(nn.Module):
    """双卷积块 - 增加Dropout"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net网络架构 - 带Dropout正则化"""
    def __init__(self, in_channels=3, out_channels=1, features=None, dropout_p=0.2):
        super(UNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_p)

        # 编码器
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_p=dropout_p * 0.5))
            in_channels = feature

        # 底部
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_p=dropout_p)

        # 解码器
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature, dropout_p=dropout_p * 0.5))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            x = self.dropout(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)

# ================== Attention U-Net ==================
class AttentionBlock(nn.Module):
    """注意力门模块"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    """带注意力机制的U-Net - 增加Dropout"""
    def __init__(self, in_channels=3, out_channels=1, features=None, dropout_p=0.2):
        super(AttentionUNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_p)

        # 编码器
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, dropout_p=dropout_p * 0.5))
            in_channels = feature

        # 底部
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2, dropout_p=dropout_p)

        # 解码器
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.attentions.append(AttentionBlock(F_g=feature, F_l=feature, F_int=feature // 2))
            self.ups.append(DoubleConv(feature * 2, feature, dropout_p=dropout_p * 0.5))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            x = self.dropout(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            skip_connection = self.attentions[idx // 2](x, skip_connection)
            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        return self.final_conv(x)


# ================== ResBlock for deeper network ==================
class ResBlock(nn.Module):
    """残差块"""
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_p)

        # Skip connection
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class ResUNet(nn.Module):
    """带残差连接的U-Net - 更强的梯度流"""
    def __init__(self, in_channels=3, out_channels=1, features=None, dropout_p=0.2):
        super(ResUNet, self).__init__()
        if features is None:
            features = [64, 128, 256, 512]

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_p)

        # 编码器
        for feature in features:
            self.encoder.append(ResBlock(in_channels, feature, dropout_p=dropout_p * 0.5))
            in_channels = feature

        # 底部
        self.bottleneck = ResBlock(features[-1], features[-1] * 2, dropout_p=dropout_p)

        # 解码器
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(ResBlock(feature * 2, feature, dropout_p=dropout_p * 0.5))

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for enc in self.encoder:
            x = enc(x)
            skip_connections.append(x)
            x = self.pool(x)
            x = self.dropout(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip = skip_connections[idx // 2]

            if x.shape != skip.shape:
                x = TF.resize(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_conv(x)

# ================== 损失函数 ==================
class DiceLoss(nn.Module):
    """Dice损失函数"""
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice

class BCEDiceLoss(nn.Module):
    """BCE + Dice组合损失"""
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

class FocalDiceLoss(nn.Module):
    """Focal Loss + Dice - 更好处理类别不平衡"""
    def __init__(self, gamma=2.0, alpha=0.25, dice_weight=0.5):
        super(FocalDiceLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.dice_weight = dice_weight
        self.dice = DiceLoss()

    def forward(self, pred, target):
        # Focal Loss
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce
        focal_loss = focal_loss.mean()

        # Dice Loss
        dice_loss = self.dice(pred, target)

        return (1 - self.dice_weight) * focal_loss + self.dice_weight * dice_loss

# ================== 评估指标 ==================
def calculate_dice(pred, target, threshold=0.5):
    """计算Dice系数"""
    pred = (torch.sigmoid(pred) > threshold).float()
    smooth = 1e-6

    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    return dice.item()

# ================== TTA (Test Time Augmentation) ==================
def tta_predict(model, image, device, use_amp=False):
    """测试时增强 - 通过多次变换预测取平均"""
    model.eval()
    predictions = []

    transforms_list = [
        lambda x: x,  # 原图
        lambda x: TF.hflip(x),  # 水平翻转
        lambda x: TF.vflip(x),  # 垂直翻转
        lambda x: TF.rotate(x, 90),  # 旋转90度
        lambda x: TF.rotate(x, 180),  # 旋转180度
        lambda x: TF.rotate(x, 270),  # 旋转270度
    ]

    inverse_transforms = [
        lambda x: x,
        lambda x: TF.hflip(x),
        lambda x: TF.vflip(x),
        lambda x: TF.rotate(x, -90),
        lambda x: TF.rotate(x, -180),
        lambda x: TF.rotate(x, -270),
    ]

    with torch.no_grad():
        for transform, inverse in zip(transforms_list, inverse_transforms):
            # 应用变换
            aug_image = transform(image)

            if use_amp:
                with autocast():
                    output = model(aug_image)
            else:
                output = model(aug_image)

            output = torch.sigmoid(output)

            # 逆变换
            output = inverse(output)
            predictions.append(output)

    # 平均所有预测
    final_pred = torch.stack(predictions).mean(dim=0)
    return final_pred

# ================== 训练函数 ==================
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None, use_amp=False):
    model.train()
    total_loss = 0.0
    total_dice = 0.0

    pbar = tqdm(dataloader, desc='Training')
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if use_amp and scaler is not None:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            total_dice += calculate_dice(outputs, labels)

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)

    return avg_loss, avg_dice

def validate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    total_loss = 0.0
    total_dice = 0.0

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if use_amp:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            total_dice += calculate_dice(outputs, labels)

    avg_loss = total_loss / len(dataloader)
    avg_dice = total_dice / len(dataloader)

    return avg_loss, avg_dice

# ================== 预测和保存 ==================
def predict_and_save(model, dataloader, device, output_dir, use_amp=False, use_tta=True):
    """对测试集进行预测并保存分割结果"""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for images, img_names in tqdm(dataloader, desc='Predicting'):
            images = images.to(device, non_blocking=True)

            if use_tta:
                # 使用TTA
                outputs = tta_predict(model, images, device, use_amp)
            else:
                if use_amp:
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                outputs = torch.sigmoid(outputs)

            for i, img_name in enumerate(img_names):
                pred = outputs[i, 0].cpu().numpy()

                # 二值化: pred > 0.5 表示血管
                pred_binary = (pred > 0.5).astype(np.uint8)

                # ★★★ 关键修复 ★★★
                # segmentation_to_csv.py 编码的是 非零像素
                # 题目要求：血管=0(黑色)，背景=255(白色)
                # 我们模型输出：血管=1, 背景=0
                # 所以需要反转：血管变为0，背景变为255
                # 但是！RLE编码的是非零像素！
                # 所以我们需要输出：血管=255(会被RLE编码)，背景=0
                pred_img_array = pred_binary * 255

                pred_img = Image.fromarray(pred_img_array)
                save_path = os.path.join(output_dir, img_name)
                pred_img.save(save_path)

    print(f'Predictions saved to {output_dir}')

# ================== Run-Length Encoding ==================
def rle_encode(mask):
    """
    将mask转换为run-length encoding
    mask: numpy array，血管区域=1，背景=0
    """
    mask = mask.flatten()
    mask = (mask > 0).astype(np.uint8)

    dots = np.where(mask == 1)[0]
    run_lengths = []
    prev = -2

    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b

    return ' '.join([str(r) for r in run_lengths])

def generate_submission(model, dataloader, device, output_path='submission.csv', use_amp=False, use_tta=True):
    """直接生成提交文件"""
    model.eval()
    results = []

    with torch.no_grad():
        for images, img_names in tqdm(dataloader, desc='Generating submission'):
            images = images.to(device, non_blocking=True)

            if use_tta:
                outputs = tta_predict(model, images, device, use_amp)
            else:
                if use_amp:
                    with autocast():
                        outputs = model(images)
                else:
                    outputs = model(images)
                outputs = torch.sigmoid(outputs)

            for i, img_name in enumerate(img_names):
                pred = outputs[i, 0].cpu().numpy()

                # 确保尺寸为512x512
                if pred.shape != (512, 512):
                    pred_img = Image.fromarray((pred * 255).astype(np.uint8))
                    pred_img = pred_img.resize((512, 512), Image.Resampling.BILINEAR)
                    pred = np.array(pred_img) / 255.0

                # 二值化: pred > 0.5 表示血管=1
                pred_binary = (pred > 0.5).astype(np.uint8)

                # ★★★ RLE编码血管区域 ★★★
                # pred_binary: 血管=1, 背景=0
                # rle_encode 会编码值为1的像素
                rle = rle_encode(pred_binary)

                img_id = img_name.split('.')[0]
                results.append({'Id': img_id, 'Predicted': rle})

    # 创建DataFrame并排序
    df = pd.DataFrame(results)
    df['Id'] = df['Id'].astype(int)
    df = df.sort_values('Id')
    df.to_csv(output_path, index=False)
    print(f'Submission saved to {output_path}')

# ================== 主函数 ==================
def train_single_fold(fold_idx, train_indices, val_indices, full_dataset, val_dataset_no_aug, test_loader, config):
    """训练单个fold"""
    print(f"\n{'='*60}")
    print(f"Training Fold {fold_idx + 1}")
    print(f"{'='*60}")

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(val_dataset_no_aug, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 创建模型
    model = ResUNet(
        in_channels=config.IN_CHANNELS,
        out_channels=config.OUT_CHANNELS,
        features=config.FEATURES,
        dropout_p=0.3
    ).to(config.DEVICE)

    criterion = FocalDiceLoss(gamma=2.0, alpha=0.25, dice_weight=0.5)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )

    scaler = GradScaler() if config.USE_AMP else None

    best_val_dice = 0.0
    patience = 30
    patience_counter = 0
    model_path = f'/kaggle/working/best_model_fold{fold_idx}.pth'

    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE,
            scaler=scaler, use_amp=config.USE_AMP
        )

        val_loss, val_dice = validate(
            model, val_loader, criterion, config.DEVICE,
            use_amp=config.USE_AMP
        )

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: Train Dice={train_dice:.4f}, Val Dice={val_dice:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience and epoch >= 50:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"Fold {fold_idx + 1} Best Val Dice: {best_val_dice:.4f}")

    # 加载最佳模型
    model.load_state_dict(torch.load(model_path))
    return model, best_val_dice


def ensemble_predict(models, dataloader, device, use_amp=False, use_tta=True):
    """使用多个模型进行ensemble预测"""
    for model in models:
        model.eval()

    results = []

    with torch.no_grad():
        for images, img_names in tqdm(dataloader, desc='Ensemble predicting'):
            images = images.to(device, non_blocking=True)

            # 收集所有模型的预测
            all_predictions = []
            for model in models:
                if use_tta:
                    pred = tta_predict(model, images, device, use_amp)
                else:
                    if use_amp:
                        with autocast():
                            output = model(images)
                    else:
                        output = model(images)
                    pred = torch.sigmoid(output)
                all_predictions.append(pred)

            # 平均所有模型的预测
            ensemble_pred = torch.stack(all_predictions).mean(dim=0)

            for i, img_name in enumerate(img_names):
                pred = ensemble_pred[i, 0].cpu().numpy()

                if pred.shape != (512, 512):
                    pred_img = Image.fromarray((pred * 255).astype(np.uint8))
                    pred_img = pred_img.resize((512, 512), Image.Resampling.BILINEAR)
                    pred = np.array(pred_img) / 255.0

                # 二值化
                pred_binary = (pred > 0.5).astype(np.uint8)

                # RLE编码
                rle = rle_encode(pred_binary)

                img_id = img_name.split('.')[0]
                results.append({'Id': img_id, 'Predicted': rle})

    return results


def main():
    print(f"Using device: {config.DEVICE}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # 创建数据集
    train_transform = StrongTrainTransform(p=0.5)

    full_train_dataset = VesselDataset(
        image_dir=config.TRAIN_IMG_DIR,
        label_dir=config.TRAIN_LABEL_DIR,
        transform=train_transform,
        is_train=True
    )

    val_dataset_no_aug = VesselDataset(
        image_dir=config.TRAIN_IMG_DIR,
        label_dir=config.TRAIN_LABEL_DIR,
        transform=None,
        is_train=True
    )

    test_dataset = VesselDataset(
        image_dir=config.TEST_IMG_DIR,
        label_dir=None,
        transform=None,
        is_train=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    print(f"Total train samples: {len(full_train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # K折交叉验证
    K_FOLDS = 5
    dataset_size = len(full_train_dataset)
    indices = list(range(dataset_size))
    random.shuffle(indices)

    fold_size = dataset_size // K_FOLDS
    models = []
    fold_scores = []

    for fold in range(K_FOLDS):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < K_FOLDS - 1 else dataset_size
        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        model, score = train_single_fold(
            fold, train_indices, val_indices,
            full_train_dataset, val_dataset_no_aug, test_loader, config
        )
        models.append(model)
        fold_scores.append(score)

    print(f"\n{'='*60}")
    print(f"Cross-validation completed!")
    print(f"Fold scores: {fold_scores}")
    print(f"Mean CV Dice: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    print(f"{'='*60}")

    # Ensemble预测
    print("\nGenerating ensemble submission with TTA...")
    results = ensemble_predict(models, test_loader, config.DEVICE, use_amp=config.USE_AMP, use_tta=True)

    df = pd.DataFrame(results)
    df['Id'] = df['Id'].astype(int)
    df = df.sort_values('Id')
    df.to_csv('/kaggle/working/submission.csv', index=False)
    print(f"Submission saved to /kaggle/working/submission.csv")

    print("\n" + "=" * 60)
    print("All done!")
    print(f"Mean CV Dice: {np.mean(fold_scores):.4f}")

if __name__ == '__main__':
    main()