"""
model_bert.py
=============
BERT-style Transformer Encoder untuk ECG.

Catatan: Ini adalah *encoder-only* Transformer — tidak ada masked LM pretraining
(sesuai simplifikasi pada risiko antisipasi dokumen pembagian kerja). Tujuannya
adalah menangkap dependensi temporal jangka panjang antar-timestep dalam detak.

Input  : (batch, 180, 3)  — sinyal multi-scale dari preprocessing Anadya
Output : (batch, bert_dim) — representasi [CLS] token setelah encoder

Arsitektur:
    Linear input projection (3 → bert_dim)
    + Learnable Positional Encoding (180 timesteps)
    + [CLS] token prepend
    → 4x TransformerEncoderLayer (4 attention heads, dim=256, FFN=512)
    → Ambil posisi [CLS] sebagai representasi urutan
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding — ditambahkan ke sequence setelah projection.

    Menggunakan pendekatan learnable sebagai alternatif (bisa di-swap):
    di sini memakai versi fixed sinusoidal agar lebih stabil saat data kecil.
    """

    def __init__(self, d_model: int, max_len: int = 182, dropout: float = 0.1):
        # max_len = 180 timestep + 1 CLS + sedikit buffer
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BERTEncoder(nn.Module):
    """
    Transformer Encoder (BERT-style) untuk representasi urutan ECG.

    Parameters
    ----------
    in_channels  : channel input (default 3, multi-scale dari preprocessing)
    bert_dim     : dimensi embedding internal (default 256)
    num_heads    : jumlah attention heads (default 4)
    num_layers   : jumlah encoder layer (default 4)
    ffn_dim      : dimensi feed-forward network (default 512 = 2 × bert_dim)
    dropout      : dropout rate (default 0.1)
    max_seq_len  : panjang sekuens maksimum incl. CLS token (default 182)
    """

    def __init__(
        self,
        in_channels: int = 3,
        bert_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 182,
    ):
        super().__init__()
        self.bert_dim = bert_dim

        # Proyeksikan 3 channel ke bert_dim (linear embedding per timestep)
        self.input_projection = nn.Linear(in_channels, bert_dim)

        # [CLS] token — learnable, di-prepend ke depan sekuens
        self.cls_token = nn.Parameter(torch.zeros(1, 1, bert_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(bert_dim, max_len=max_seq_len,
                                               dropout=dropout)

        # Stack TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=bert_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation='gelu',          # GELU seperti BERT asli
            batch_first=True,           # (batch, seq, dim)
            norm_first=True,            # Pre-LN lebih stabil
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(bert_dim),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor shape (batch, 180, 3)
            Output preprocessing Anadya (multi-scale, z-scored)

        Returns
        -------
        Tensor shape (batch, bert_dim)
            Representasi [CLS] token — ringkasan seluruh sekuens
        """
        batch_size = x.size(0)

        # (batch, 180, 3) → (batch, 180, bert_dim)
        x = self.input_projection(x)

        # Prepend [CLS] token → (batch, 181, bert_dim)
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Tambah positional encoding
        x = self.pos_encoding(x)          # (batch, 181, bert_dim)

        # Self-attention transformer encoder
        x = self.transformer(x)           # (batch, 181, bert_dim)

        # Ambil representasi [CLS] token (posisi 0)
        cls_repr = x[:, 0, :]             # (batch, bert_dim)
        return cls_repr


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    model = BERTEncoder(
        in_channels=3,
        bert_dim=256,
        num_heads=4,
        num_layers=4,
        ffn_dim=512,
    )
    dummy = torch.randn(8, 180, 3)
    out   = model(dummy)
    print(f"[BERTEncoder] input: {dummy.shape}  →  output: {out.shape}")
    assert out.shape == (8, 256), "Shape mismatch!"
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[BERTEncoder] Total parameters: {total_params:,}")
