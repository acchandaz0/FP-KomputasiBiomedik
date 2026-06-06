"""
model_cnn.py
============
1D-CNN module untuk ekstraksi fitur ECG dari sinyal multi-channel.

Input  : (batch, 180, 3)  — 3 channel multi-scale dari preprocessing Anadya
Output : (batch, cnn_out_dim)  — representasi fitur flat siap masuk fusion

Arsitektur:
    Conv1D(3→32, k=3) → BN → ReLU → Dropout
    Conv1D(32→64, k=3) → BN → ReLU → Dropout
    Conv1D(64→128, k=3) → BN → ReLU → Dropout
    AdaptiveAvgPool1D(1) → Flatten  → Linear(128, cnn_out_dim) → ReLU
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Satu blok konvolusi: Conv1d → BatchNorm → ReLU → Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size,
                      padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1D(nn.Module):
    """
    1D-CNN untuk ECG.

    Parameters
    ----------
    in_channels   : jumlah channel input (default 3, dari multi-scale Anadya)
    cnn_out_dim   : dimensi vektor keluaran setelah projection (default 256)
    kernel_size   : ukuran kernel konvolusi (default 3, sesuai spesifikasi)
    dropout       : dropout rate (default 0.3)
    """

    def __init__(
        self,
        in_channels: int = 3,
        cnn_out_dim: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()

        # 3 blok konvolusi bertumpuk: channel 3→32→64→128
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels, 32,  kernel_size, dropout),
            ConvBlock(32,          64,  kernel_size, dropout),
            ConvBlock(64,          128, kernel_size, dropout),
        )

        # Global average pooling → vektor (batch, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Projection ke dimensi yang sama dengan BERT output
        self.projection = nn.Sequential(
            nn.Linear(128, cnn_out_dim),
            nn.ReLU(inplace=True),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (batch, 180, 3)
            Output preprocessing Anadya (sudah multi-scale, z-scored)

        Returns
        -------
        Tensor shape (batch, cnn_out_dim)
        """
        # Conv1d butuh (batch, channels, length) — tukar axis
        x = x.permute(0, 2, 1)          # (batch, 3, 180)
        x = self.conv_layers(x)          # (batch, 128, 180)
        x = self.pool(x).squeeze(-1)     # (batch, 128)
        x = self.projection(x)           # (batch, cnn_out_dim)
        return x


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = CNN1D(in_channels=3, cnn_out_dim=256)
    dummy  = torch.randn(8, 180, 3)          # batch=8, sesuai input preprocessing
    out    = model(dummy)
    print(f"[CNN1D] input: {dummy.shape}  →  output: {out.shape}")
    assert out.shape == (8, 256), "Shape mismatch!"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[CNN1D] Total parameters: {total_params:,}")
