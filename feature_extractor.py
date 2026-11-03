"""
Feature Extraction Module
==========================
Extracts Log-Mel Spectrograms and MFCCs from preprocessed audio waveforms,
combining them into a dual-channel tensor for the CNN-BiLSTM model.
"""

import torch
import torchaudio
import numpy as np
import os
from config import config

class FeatureExtractor:
    """Extracts spectral features from audio waveforms."""

    def __init__(
        self,
        sample_rate: int = config.SAMPLE_RATE,
        n_mels: int = config.N_MELS,
        n_mfcc: int = config.N_MFCC,
        n_fft: int = config.N_FFT,
        hop_length: int = config.HOP_LENGTH,
        win_length: int = config.WIN_LENGTH,
        f_min: float = config.F_MIN,
        f_max: float = config.F_MAX,
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc

        # Mel Spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
        )

        # MFCC transform
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={
                "n_fft": n_fft,
                "hop_length": hop_length,
                "win_length": win_length,
                "n_mels": n_mels,
                "f_min": f_min,
                "f_max": f_max,
            },
        )

        # Amplitude to dB conversion
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype="power", top_db=80
        )

    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract log-Mel spectrogram from waveform.

        Args:
            waveform: Audio tensor of shape (1, num_samples)

        Returns:
            Log-Mel spectrogram of shape (1, n_mels, time_frames)
        """
        mel_spec = self.mel_transform(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # Normalize to [0, 1] range
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (
            log_mel_spec.max() - log_mel_spec.min() + 1e-8
        )

        return log_mel_spec

    def extract_mfcc(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract MFCC features from waveform.

        Args:
            waveform: Audio tensor of shape (1, num_samples)

        Returns:
            MFCC features of shape (1, n_mfcc, time_frames)
        """
        mfccs = self.mfcc_transform(waveform)

        # Normalize MFCCs (zero mean, unit variance per coefficient)
        mean = mfccs.mean(dim=-1, keepdim=True)
        std = mfccs.std(dim=-1, keepdim=True) + 1e-8
        mfccs = (mfccs - mean) / std
        
        return mfccs

    def extract_combined_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Extract and combine Mel spectrogram and MFCC into a dual-channel tensor.

        The MFCC (n_mfcc features) is padded along the frequency axis to match
        the Mel spectrogram (n_mels features), creating a 2-channel input.

        Args:
            waveform: Audio tensor of shape (1, num_samples)

        Returns:
            Combined features of shape (2, n_mels, time_frames)
        """
        # Extract features
        mel_spec = self.extract_mel_spectrogram(waveform)  # (1, n_mels, T)
        mfccs = self.extract_mfcc(waveform)  # (1, n_mfcc, T)

        # Ensure same time dimension
        min_time = min(mel_spec.shape[-1], mfccs.shape[-1])
        mel_spec = mel_spec[..., :min_time]
        mfccs = mfccs[..., :min_time]

        # Pad MFCCs to match Mel spectrogram height (n_mfcc -> n_mels)
        if self.n_mfcc < self.n_mels:
            pad_size = self.n_mels - self.n_mfcc
            mfccs = torch.nn.functional.pad(mfccs, (0, 0, 0, pad_size))

        # Stack as 2 channels: (2, n_mels, time_frames)
        combined = torch.cat([mel_spec, mfccs], dim=0)

        return combined

    def extract_features_for_visualization(
        self, waveform: torch.Tensor
    ) -> dict:
        """
        Extract features and return them separately for visualization.
        Used by the web UI to display spectrograms.

        Returns:
            Dict with 'mel_spectrogram' and 'mfcc' numpy arrays
        """
        mel_spec = self.extract_mel_spectrogram(waveform)
        mfccs = self.extract_mfcc(waveform)

        return {
            "mel_spectrogram": mel_spec.squeeze().numpy(),
            "mfcc": mfccs.squeeze().numpy(),
        }

def extract_and_save_features(
    processed_dir: str, label: int
) -> list[dict]:
    """
    Extract features from all preprocessed audio tensors in a directory.

    Args:
        processed_dir: Directory with preprocessed .pt files
        label: 0 for authentic, 1 for fake

    Returns:
        List of feature dicts with 'features', 'label', and 'source' keys
    """
    from tqdm import tqdm

    extractor = FeatureExtractor()
    all_data = []

    pt_files = [f for f in os.listdir(processed_dir) if f.endswith(".pt")]

    for filename in tqdm(pt_files, desc=f"Extracting features ({'authentic' if label == 0 else 'fake'})"):
        input_path = os.path.join(processed_dir, filename)

        try:
            data = torch.load(input_path, weights_only=True)
            waveform = data["waveform"]

            features = extractor.extract_combined_features(waveform)

            all_data.append({
                "features": features,
                "label": label,
                "source": filename,
            })
        except Exception as e:
            print(f"  ⚠ Error extracting features from {filename}: {e}")

    return all_data
