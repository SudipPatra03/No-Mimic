"""
CNN-BiLSTM with Attention — Voice Authenticity Detection Model
================================================================
Hybrid architecture that combines:
- CNN blocks for spatial/spectral feature extraction from Mel spectrograms
- Bidirectional LSTM for temporal pattern recognition
- Self-attention pooling for focusing on discriminative time segments
"""

import torch
import torch.nn as nn
from config import config

class ConvBlock(nn.Module):
    """A single convolutional block: Conv2d → BatchNorm → ReLU → MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=config.CNN_KERNEL_SIZE,
                stride=1,
                padding=config.CNN_PADDING,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.dropout(x)
        return x


class SelfAttention(nn.Module):
    """
    Self-attention layer for temporal pooling.
    Learns to weight different time steps based on their importance
    for the classification task.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time_steps, features)

        Returns:
            context: (batch, features) — attention-weighted sum over time
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # (batch, time, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # Weighted sum
        context = torch.sum(x * attn_weights, dim=1)  # (batch, features)
        
        return context


class VoiceAuthenticityModel(nn.Module):
    """
    CNN-BiLSTM with Self-Attention for Voice Authenticity Detection.

    Architecture:
        Input (2, 80, T)
        → ConvBlock(2→32) → ConvBlock(32→64) → ConvBlock(64→128)
        → Reshape to (batch, time, features)
        → BiLSTM(2 layers, 128 hidden)
        → Self-Attention pooling
        → FC(256→64) → ReLU → Dropout → FC(64→1) → Sigmoid
    """

    def __init__(self):
        super().__init__()

        # CNN Feature Extractor
        self.conv1 = ConvBlock(
            config.CNN_CHANNELS[0], config.CNN_CHANNELS[1]
        )
        self.conv2 = ConvBlock(
            config.CNN_CHANNELS[1], config.CNN_CHANNELS[2]
        )
        self.conv3 = ConvBlock(
            config.CNN_CHANNELS[2], config.CNN_CHANNELS[3],
            dropout=config.CNN_DROPOUT,
        )

        # After 3 MaxPool2d(2,2) operations:
        # Frequency: 80 → 40 → 20 → 10
        # Time: T → T/2 → T/4 → T/8
        # Channels: 128
        # Feature dim per time step = 128 * 10 = 1280
        self._freq_bins_after_cnn = config.N_MELS // 8  # 80 // 8 = 10
        self._lstm_input_size = config.CNN_CHANNELS[3] * self._freq_bins_after_cnn

        # BiLSTM Temporal Encoder
        self.lstm = nn.LSTM(
            input_size=self._lstm_input_size,
            hidden_size=config.LSTM_HIDDEN_SIZE,
            num_layers=config.LSTM_NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.LSTM_DROPOUT if config.LSTM_NUM_LAYERS > 1 else 0,
        )

        # # GRU Temporal Encoder
        # self.lstm = nn.GRU(
        #     input_size=self._lstm_input_size,
        #     hidden_size=config.GRU_HIDDEN_SIZE,
        #     num_layers=config.GRU_NUM_LAYERS,
        #     batch_first=True,
        #     bidirectional=True,
        #     dropout=config.GRU_DROPOUT if config.GRU_NUM_LAYERS > 1 else 0,
        # )

        # Self-Attention Pooling
        self._lstm_output_size = config.LSTM_HIDDEN_SIZE * 2  # *2 for bidirectional
        self.attention = SelfAttention(self._lstm_output_size)

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self._lstm_output_size, config.FC_HIDDEN_SIZE),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.FC_DROPOUT),
            nn.Linear(config.FC_HIDDEN_SIZE, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 2, n_mels, time_frames)

        Returns:
            Output logits of shape (batch, 1)
        """
        # CNN: extract spatial/spectral features
        x = self.conv1(x)  # (B, 32, 40, T/2)
        x = self.conv2(x)  # (B, 64, 20, T/4)
        x = self.conv3(x)  # (B, 128, 10, T/8)

        # Reshape for LSTM: (B, channels, freq, time) → (B, time, channels*freq)
        batch_size, channels, freq, time = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, time, channels, freq)
        x = x.reshape(batch_size, time, channels * freq)  # (B, time, 1280)

        # BiLSTM: temporal encoding
        x, _ = self.lstm(x)  # (B, time, 256)

        # Self-Attention: weighted temporal pooling
        x = self.attention(x)  # (B, 256)

        # Classification
        x = self.classifier(x)  # (B, 1)

        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability output (sigmoid applied)."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model() -> VoiceAuthenticityModel:
    """Build and return the model, printing parameter count."""
    model = VoiceAuthenticityModel()
    param_count = model.count_parameters()
    print(f"Model built — {param_count:,} trainable parameters")
    return model
