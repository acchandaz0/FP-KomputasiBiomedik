"""
model_fusion.py
===============
Multimodal Fusion Module — menggabungkan tiga stream:
    1. CNN output        : (batch, 256)  — fitur morfologi lokal ECG
    2. BERT output       : (batch, 256)  — fitur temporal global ECG
    3. Clinical features : (batch, 13)   — fitur klinis dari Anadya

Strategi Fusion:
    ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐
    │  CNN  (256)  │  │  BERT (256)  │  │  Clinical (13)  │
    └──────┬───────┘  └──────┬───────┘  └────────┬────────┘
           │                 │                    │
           └────────Cross-Modal Attention─────────┘
                             │
                    Gated Fusion Layer
                             │
                    MLP Classifier Head
                             │
                      logits (5 kelas)

Cross-Modal Attention:
    CNN dan BERT saling attend satu sama lain (bidirectional cross-attention),
    lalu hasilnya digabung dengan clinical features lewat gated fusion.

Gated Fusion:
    z = sigmoid(W_gate · concat(cnn_attn, bert_attn, clinical_proj))
    fused = z · tanh(W_fuse · concat(cnn_attn, bert_attn, clinical_proj))
"""

import torch
import torch.nn as nn

from model_cnn  import CNN1D
from model_bert import BERTEncoder


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention antara CNN dan BERT.

    CNN attend ke BERT (CNN queries BERT):
        cnn_attn = Attention(Q=cnn, K=bert, V=bert)
    BERT attend ke CNN (BERT queries CNN):
        bert_attn = Attention(Q=bert, K=cnn, V=cnn)
    """

    def __init__(self, dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # CNN → BERT attention
        self.cnn_to_bert = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        # BERT → CNN attention
        self.bert_to_cnn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )

        self.norm_cnn  = nn.LayerNorm(dim)
        self.norm_bert = nn.LayerNorm(dim)

    def forward(
        self,
        cnn_feat: torch.Tensor,   # (batch, dim)
        bert_feat: torch.Tensor,  # (batch, dim)
    ):
        # Perlu menambah dimensi seq_len=1 untuk MultiheadAttention
        cnn_q  = cnn_feat.unsqueeze(1)    # (batch, 1, dim)
        bert_q = bert_feat.unsqueeze(1)   # (batch, 1, dim)

        # CNN queries BERT (CNN minta info dari BERT)
        cnn_attn, _  = self.cnn_to_bert(
            query=cnn_q, key=bert_q, value=bert_q
        )
        cnn_attn = self.norm_cnn(cnn_feat + cnn_attn.squeeze(1))   # residual

        # BERT queries CNN (BERT minta info dari CNN)
        bert_attn, _ = self.bert_to_cnn(
            query=bert_q, key=cnn_q, value=cnn_q
        )
        bert_attn = self.norm_bert(bert_feat + bert_attn.squeeze(1))  # residual

        return cnn_attn, bert_attn   # keduanya (batch, dim)


class GatedFusion(nn.Module):
    """
    Gated fusion untuk menggabungkan tiga modalitas:
        cnn_attn  (batch, dim)
        bert_attn (batch, dim)
        clinical  (batch, clinical_proj_dim)

    z = sigmoid(W_gate · [cnn_attn ‖ bert_attn ‖ clinical])
    fused = z ⊙ tanh(W_fuse · [cnn_attn ‖ bert_attn ‖ clinical])
    """

    def __init__(self, cnn_dim: int = 256, bert_dim: int = 256,
                 clinical_dim: int = 64, fused_dim: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        combined_dim = cnn_dim + bert_dim + clinical_dim

        self.gate = nn.Linear(combined_dim, fused_dim)
        self.fuse = nn.Linear(combined_dim, fused_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm    = nn.LayerNorm(fused_dim)

    def forward(
        self,
        cnn_attn: torch.Tensor,      # (batch, cnn_dim)
        bert_attn: torch.Tensor,     # (batch, bert_dim)
        clinical: torch.Tensor,      # (batch, clinical_dim)
    ) -> torch.Tensor:
        x = torch.cat([cnn_attn, bert_attn, clinical], dim=-1)  # (batch, combined)
        z = torch.sigmoid(self.gate(x))                          # (batch, fused_dim)
        h = torch.tanh(self.fuse(x))                             # (batch, fused_dim)
        out = self.norm(self.dropout(z * h))                     # (batch, fused_dim)
        return out


class ECGBert(nn.Module):
    """
    Model lengkap ECGBert: CNN + BERT + Clinical → Cross-Modal Attention
    → Gated Fusion → Classifier.

    Parameters
    ----------
    num_classes     : jumlah kelas AAMI (default 5: N, S, V, F, Q)
    cnn_out_dim     : dimensi output CNN (default 256)
    bert_dim        : dimensi BERT internal (default 256)
    num_heads       : attention heads (default 4)
    num_bert_layers : jumlah layer Transformer (default 4)
    clinical_in     : dimensi input clinical features (default 13)
    clinical_proj   : dimensi proyeksi clinical (default 64)
    fused_dim       : dimensi fused representation (default 256)
    dropout         : dropout rate (default 0.3)
    """

    def __init__(
        self,
        num_classes: int       = 5,
        cnn_out_dim: int       = 256,
        bert_dim: int          = 256,
        num_heads: int         = 4,
        num_bert_layers: int   = 4,
        clinical_in: int       = 13,
        clinical_proj: int     = 64,
        fused_dim: int         = 256,
        dropout: float         = 0.3,
    ):
        super().__init__()

        # ── Stream 1: CNN ──────────────────────────────────────────────────
        self.cnn = CNN1D(
            in_channels=3,
            cnn_out_dim=cnn_out_dim,
            kernel_size=3,
            dropout=dropout,
        )

        # ── Stream 2: BERT Transformer ────────────────────────────────────
        self.bert = BERTEncoder(
            in_channels=3,
            bert_dim=bert_dim,
            num_heads=num_heads,
            num_layers=num_bert_layers,
            ffn_dim=bert_dim * 2,
            dropout=0.1,
        )

        # ── Stream 3: Clinical features projection ────────────────────────
        self.clinical_proj = nn.Sequential(
            nn.Linear(clinical_in, clinical_proj),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ── Cross-Modal Attention ─────────────────────────────────────────
        self.cross_attn = CrossModalAttention(
            dim=cnn_out_dim,     # harus sama antara CNN dan BERT
            num_heads=num_heads,
            dropout=0.1,
        )

        # ── Gated Fusion ──────────────────────────────────────────────────
        self.gated_fusion = GatedFusion(
            cnn_dim=cnn_out_dim,
            bert_dim=bert_dim,
            clinical_dim=clinical_proj,
            fused_dim=fused_dim,
            dropout=dropout,
        )

        # ── Classifier Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, num_classes),
        )

        self._init_classifier()

    def _init_classifier(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        x_signal: torch.Tensor,    # (batch, 180, 3) — dari preprocessing Anadya
        x_clinical: torch.Tensor,  # (batch, 13)     — dari preprocessing Anadya
    ) -> torch.Tensor:
        """
        Returns
        -------
        logits : Tensor shape (batch, num_classes)
            Raw logits sebelum softmax — langsung masuk ke CrossEntropyLoss
            atau FocalLoss yang diimplementasi Anadya.
        """
        # ── Tiga stream paralel ───────────────────────────────────────────
        cnn_feat      = self.cnn(x_signal)          # (batch, 256)
        bert_feat     = self.bert(x_signal)          # (batch, 256)
        clinical_feat = self.clinical_proj(x_clinical)  # (batch, 64)

        # ── Cross-modal attention ─────────────────────────────────────────
        cnn_attn, bert_attn = self.cross_attn(cnn_feat, bert_feat)

        # ── Gated fusion ──────────────────────────────────────────────────
        fused = self.gated_fusion(cnn_attn, bert_attn, clinical_feat)

        # ── Klasifikasi ───────────────────────────────────────────────────
        logits = self.classifier(fused)              # (batch, 5)
        return logits


# ─── Quick sanity check ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import torch

    model = ECGBert(
        num_classes=5,
        cnn_out_dim=256,
        bert_dim=256,
        num_heads=4,
        num_bert_layers=4,
        clinical_in=13,
        clinical_proj=64,
        fused_dim=256,
        dropout=0.3,
    )

    # Simulasi batch dari output preprocessing Anadya
    x_sig = torch.randn(8, 180, 3)   # batch=8, 180 sampel, 3 channel multi-scale
    x_clin = torch.randn(8, 13)      # batch=8, 13 clinical features

    logits = model(x_sig, x_clin)
    print(f"[ECGBert] x_signal:   {x_sig.shape}")
    print(f"[ECGBert] x_clinical: {x_clin.shape}")
    print(f"[ECGBert] logits:     {logits.shape}")
    assert logits.shape == (8, 5), "Shape mismatch!"

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[ECGBert] Total parameters    : {total_params:,}")
    print(f"[ECGBert] Trainable parameters: {trainable:,}")

    # Cek alur gradient
    loss = logits.sum()
    loss.backward()
    print("\n[ECGBert] Backward pass: OK ✓")
