"""
PyTorch Dataset Class
======================
Custom Dataset for loading precomputed audio features for training/evaluation.
Uses lazy-loading from individual .pt files to minimize memory usage.
Supports on-the-fly feature extraction as a fallback.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from config import config
from src.preprocessor import AudioPreprocessor
from src.feature_extractor import FeatureExtractor

class VoiceAuthenticityDataset(Dataset):
    """
    Dataset for voice authenticity detection.

    Lazy-loads precomputed feature tensors from individual .pt files.
    Each .pt file contains a dict with 'features', 'label', and 'source'.
    """

    def __init__(self, file_paths: list[str], augment: bool = False):
        """
        Args:
            file_paths: List of absolute paths to individual .pt feature files
            augment: Whether to apply feature-level augmentation
        """
        self.file_paths = file_paths
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = torch.load(self.file_paths[idx], weights_only=False)

        features = item["features"]  # (2, n_mels, time_frames)
        label = torch.tensor(item["label"], dtype=torch.float32)

        if self.augment:
            features = self._augment_features(features.clone())

        return features, label

    def _augment_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply feature-level augmentation."""
        # Frequency masking (SpecAugment-style)
        if np.random.random() < 0.3:
            freq_mask_size = np.random.randint(1, 10)
            freq_start = np.random.randint(0, features.shape[1] - freq_mask_size)
            features[:, freq_start : freq_start + freq_mask_size, :] = 0

        # Time masking (SpecAugment-style)
        if np.random.random() < 0.3:
            time_mask_size = np.random.randint(1, 20)
            time_start = np.random.randint(0, features.shape[2] - time_mask_size)
            features[:, :, time_start : time_start + time_mask_size] = 0

        return features


class RawAudioDataset(Dataset):
    """
    Dataset that processes raw audio files on-the-fly.
    Used for inference or when precomputed features are not available.
    """

    def __init__(self, file_paths: list[str], labels: list[int] = None):
        """
        Args:
            file_paths: List of audio file paths
            labels: Optional list of labels (None for inference)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.preprocessor = AudioPreprocessor()
        self.feature_extractor = FeatureExtractor()

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_path = self.file_paths[idx]

        # Preprocess and extract features
        waveform = self.preprocessor.process(file_path)
        features = self.feature_extractor.extract_combined_features(waveform)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            label = torch.tensor(-1, dtype=torch.float32)  # placeholder

        return features, label


def _get_label_from_filename(filename: str) -> int:
    """Extract label from .pt filename convention: authentic_* = 0, fake_* = 1."""
    if filename.startswith('authentic_'):
        return 0
    elif filename.startswith('fake_'):
        return 1
    else:
        raise ValueError(f"Cannot determine label from filename: {filename}")


def create_data_loaders(
    features_dir: str = None,
    batch_size: int = config.BATCH_SIZE,
    train_split: float = config.TRAIN_SPLIT,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders from individual .pt feature files.

    Args:
        features_dir: Path to the directory containing individual .pt feature files.
                      Falls back to config.FEATURES_PT_PATH if not provided.
        batch_size: Batch size for DataLoaders
        train_split: Fraction for training
        val_split: Fraction for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split

    if features_dir is None:
        features_dir = config.FEATURES_PT_PATH

    if not os.path.isdir(features_dir):
        raise FileNotFoundError(
            f"Features directory not found at {features_dir}. "
            "Please run prepare_dataset.py first."
        )

    # Collect all .pt file paths and their labels
    pt_files = sorted([f for f in os.listdir(features_dir) if f.endswith('.pt')])

    if len(pt_files) == 0:
        raise FileNotFoundError(
            f"No .pt files found in {features_dir}. "
            "Please run prepare_dataset.py first."
        )

    file_paths = [os.path.join(features_dir, f) for f in pt_files]
    labels = [_get_label_from_filename(f) for f in pt_files]

    print(f"  Found {len(file_paths)} feature files in {features_dir}")
    print(f"  Authentic: {labels.count(0)} | Fake: {labels.count(1)}")

    # First split: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        test_size=(1 - train_split),
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    val_ratio = val_split / (val_split + (1 - train_split - val_split))
    val_paths, test_paths = train_test_split(
        temp_paths,
        test_size=(1 - val_ratio),
        stratify=temp_labels,
        random_state=seed,
    )

    print(f"  Dataset splits — Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # Create datasets (lazy-loading from file paths)
    train_dataset = VoiceAuthenticityDataset(train_paths, augment=True)
    val_dataset = VoiceAuthenticityDataset(val_paths, augment=False)
    test_dataset = VoiceAuthenticityDataset(test_paths, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        val_dataset if len(test_paths) == 0 else test_dataset,  # Safety fallback
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
