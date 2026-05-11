# Dataset and DataLoader utilities for ISIC 2018 Skin Lesion Segmentation

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Training data augmentation pipeline (first augment the dataset and then train)
def get_train_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(img_size=256):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class ISICDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 127).astype(np.float32)
        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask'].unsqueeze(0)
        return image, mask

# Builds train, validation, and test DataLoaders
def build_dataloaders(image_dir, mask_dir, img_size=256, batch_size=16, val_split=0.15, test_split=0.15, seed=42):
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    image_paths, mask_paths = [], []
    for f in image_files:
        img_id = os.path.splitext(f)[0]
        mask_f = f"{img_id}_segmentation.png"
        mask_p = os.path.join(mask_dir, mask_f)
        if os.path.exists(mask_p):
            image_paths.append(os.path.join(image_dir, f))
            mask_paths.append(mask_p)
    
    indices = list(range(len(image_paths)))
    train_val_idx, test_idx = train_test_split(indices, test_size=test_split, random_state=seed)
    val_ratio = val_split / (1.0 - test_split)
    train_idx, val_idx = train_test_split(train_val_idx, test_size=val_ratio, random_state=seed)

    def _make_loader(idxs, transform, shuffle):
        ds = ISICDataset([image_paths[i] for i in idxs], [mask_paths[i] for i in idxs], transform)
        return DataLoader(ds, batch_size=batch_size if shuffle else 1, shuffle=shuffle, 
                          num_workers=2, pin_memory=True, persistent_workers=True)

    return (_make_loader(train_idx, get_train_transforms(img_size), True),
            _make_loader(val_idx, get_val_transforms(img_size), False),
            _make_loader(test_idx, get_val_transforms(img_size), False))
