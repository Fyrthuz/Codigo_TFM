import os
import json
from abc import ABC, abstractmethod
from typing import Tuple, List, Optional

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image


def extract_patient_id(case_name: str) -> str:
    """Extract patient ID from a TCGA case directory name.
    E.g., 'TCGA_CS_4941_19960909' -> 'TCGA_CS_4941'
    """
    return "_".join(case_name.split("_")[:3])


class SegmentationDataset(ABC, Dataset):
    @abstractmethod
    def __len__(self) -> int:
        ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ...


class LGGSegmentationDataset(SegmentationDataset):
    def __init__(self, root_dir, image_transform=None, mask_transform=None):
        self.root_dir = root_dir
        self.image_mask_pairs = []
        self.patient_ids = []

        for case in os.listdir(root_dir):
            case_path = os.path.join(root_dir, case)
            if os.path.isdir(case_path):
                patient_id = extract_patient_id(case)
                for file in os.listdir(case_path):
                    if file.lower().endswith(".tif") and "_mask" not in file:
                        image_path = os.path.join(case_path, file)
                        base, ext = os.path.splitext(file)
                        mask_file = base + "_mask" + ext
                        mask_path = os.path.join(case_path, mask_file)
                        if os.path.exists(mask_path):
                            self.image_mask_pairs.append((image_path, mask_path))
                            self.patient_ids.append(patient_id)

        self.image_mask_pairs.sort(key=lambda x: x[0])
        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __len__(self):
        return len(self.image_mask_pairs)

    def __getitem__(self, idx):
        image_path, mask_path = self.image_mask_pairs[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.image_transform is None:
            image = transforms.ToTensor()(image)
        else:
            image = self.image_transform(image)

        if self.mask_transform is None:
            mask = transforms.ToTensor()(mask)
        else:
            mask = self.mask_transform(mask)

        return image, mask


def recover_image_mask_pairs(root_dir):
    pairs = []
    patient_ids = []
    for case in os.listdir(root_dir):
        case_path = os.path.join(root_dir, case)
        if os.path.isdir(case_path):
            patient_id = extract_patient_id(case)
            for file in os.listdir(case_path):
                if file.lower().endswith(".tif") and "_mask" not in file:
                    image_path = os.path.join(case_path, file)
                    base, ext = os.path.splitext(file)
                    mask_file = base + "_mask" + ext
                    mask_path = os.path.join(case_path, mask_file)
                    if os.path.exists(mask_path):
                        pairs.append((image_path, mask_path))
                        patient_ids.append(patient_id)
    return pairs, patient_ids


def split_by_patient(
    patient_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Split dataset indices by patient ID to avoid data leakage.

    Returns:
        (train_indices, val_indices, test_indices)
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

    # Group indices by patient
    patient_to_indices = {}
    for idx, pid in enumerate(patient_ids):
        patient_to_indices.setdefault(pid, []).append(idx)

    patients = list(patient_to_indices.keys())
    random.shuffle(patients)

    n = len(patients)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_patients = patients[:n_train]
    val_patients = patients[n_train:n_train + n_val]
    test_patients = patients[n_train + n_val:]

    train_indices = sorted([i for p in train_patients for i in patient_to_indices[p]])
    val_indices = sorted([i for p in val_patients for i in patient_to_indices[p]])
    test_indices = sorted([i for p in test_patients for i in patient_to_indices[p]])

    return train_indices, val_indices, test_indices


def save_test_indices(test_indices: List[int], path: str):
    with open(path, "w") as f:
        json.dump(test_indices, f)


def load_test_indices(path: str) -> List[int]:
    with open(path) as f:
        return json.load(f)
