"""
Audio Preprocessing Pipeline
=============================
Handles resampling, normalization, silence trimming, and padding/truncation
to produce consistent audio tensors for feature extraction.
"""

import torch
import torchaudio
import numpy as np
import os
from config import config

class AudioPreprocessor:
    """Preprocesses raw audio files into standardized tensors."""

    def __init__(
        self,
        target_sr: int = config.SAMPLE_RATE,
        target_length: int = config.NUM_SAMPLES,
        silence_threshold_db: float = config.SILENCE_THRESHOLD_DB,
    ):
        self.target_sr = target_sr
        self.target_length = target_length
        self.silence_threshold_db = silence_threshold_db

    def load_audio(self, file_path: str) -> tuple[torch.Tensor, int]:
        """
        Load audio from file, supporting multiple formats.
        Returns (waveform, sample_rate).
        """
        try:
            waveform, sr = torchaudio.load(file_path)
        except Exception:
            # Fallback: try with librosa for unsupported formats
            import librosa
            y, sr = librosa.load(file_path, sr=None, mono=True)
            waveform = torch.from_numpy(y).unsqueeze(0)

        return waveform, sr

    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert multi-channel audio to mono by averaging channels."""
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def resample(self, waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample audio to target sample rate."""
        if orig_sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=self.target_sr
            )
            waveform = resampler(waveform)
        return waveform

    def normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize amplitude to [-1, 1] range."""
        max_val = torch.max(torch.abs(waveform))
        if max_val > 0:
            waveform = waveform / max_val
        return waveform

    def trim_silence(self, waveform: torch.Tensor) -> torch.Tensor:
        """Trim leading and trailing silence using energy threshold."""
        # Convert threshold from dB to amplitude
        threshold = 10 ** (self.silence_threshold_db / 20.0)

        audio = waveform.squeeze()
        abs_audio = torch.abs(audio)

        # Find first and last non-silent samples
        mask = abs_audio > threshold
        if mask.any():
            nonzero = torch.nonzero(mask).squeeze()
            if nonzero.dim() == 0:
                return waveform
            start = nonzero[0].item()
            end = nonzero[-1].item() + 1
            waveform = audio[start:end].unsqueeze(0)

        return waveform

    def pad_or_truncate(self, waveform: torch.Tensor) -> torch.Tensor:
        """Pad with zeros or truncate to target length."""
        current_length = waveform.shape[-1]

        if current_length > self.target_length:
            # Truncate: take center portion for center of audio
            start = (current_length - self.target_length) // 2
            waveform = waveform[..., start : start + self.target_length]
        elif current_length < self.target_length:
            # Pad with zeros at the end
            padding = self.target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        return waveform

    def process(self, file_path: str) -> torch.Tensor:
        """
        Full preprocessing pipeline.

        Args:
            file_path: Path to audio file

        Returns:
            Preprocessed waveform tensor of shape (1, target_length)
        """
        # Load
        waveform, sr = self.load_audio(file_path)

        # Convert to mono
        waveform = self.to_mono(waveform)

        # Resample to target sample rate
        waveform = self.resample(waveform, sr)

        # Normalize amplitude
        waveform = self.normalize(waveform)

        # Trim silence
        waveform = self.trim_silence(waveform)

        # Pad or truncate to fixed length
        waveform = self.pad_or_truncate(waveform)

        return waveform

    def process_with_augmentation(
        self, file_path: str, augment: bool = True
    ) -> torch.Tensor:
        """
        Process audio with optional data augmentation for training.

        Augmentations applied randomly:
        - Additive Gaussian noise
        - Random time shift
        """
        waveform = self.process(file_path)

        if augment:
            waveform = self._apply_augmentation(waveform)

        return waveform

    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply random augmentations to waveform."""
        # Additive noise
        if np.random.random() < config.AUGMENT_NOISE_PROB:
            noise = torch.randn_like(waveform) * config.AUGMENT_NOISE_FACTOR
            waveform = waveform + noise
            waveform = torch.clamp(waveform, -1.0, 1.0)

        # Random time shift
        if np.random.random() < config.AUGMENT_TIME_SHIFT_PROB:
            shift = np.random.randint(
                -config.AUGMENT_TIME_SHIFT_MAX, config.AUGMENT_TIME_SHIFT_MAX
            )
            waveform = torch.roll(waveform, shifts=shift, dims=-1)

        return waveform

def process_directory(
    input_dir: str, output_dir: str, label: int
) -> list[dict]:
    """
    Process all audio files in a directory and save as tensors.

    Args:
        input_dir: Directory containing audio files
        output_dir: Directory to save processed tensors
        label: 0 for authentic, 1 for AI-generated

    Returns:
        List of metadata dicts with file paths and labels
    """
    from tqdm import tqdm

    preprocessor = AudioPreprocessor()
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [
        f
        for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in audio_extensions
    ]

    for filename in tqdm(files, desc=f"Processing {'authentic' if label == 0 else 'fake'}"):
        input_path = os.path.join(input_dir, filename)
        output_filename = os.path.splitext(filename)[0] + ".pt"
        output_path = os.path.join(output_dir, output_filename)

        try:
            waveform = preprocessor.process(input_path)
            torch.save({"waveform": waveform, "label": label}, output_path)
            metadata.append(
                {
                    "filename": output_filename,
                    "path": output_path,
                    "label": label,
                    "original": input_path,
                }
            )
        except Exception as e:
            print(f"  ⚠ Error processing {filename}: {e}")

    return metadata
