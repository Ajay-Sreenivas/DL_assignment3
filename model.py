"""
model.py — Transformer Architecture Implementation
DA6401 Assignment 3: "Attention Is All You Need"

Design choice: Pre-LayerNorm (Pre-LN) is used over Post-LayerNorm.
Rationale: Pre-LN normalises inputs BEFORE each sub-layer rather than
after. This keeps gradient magnitudes stable from the very first step,
preventing the early-training divergence that Post-LN transformers
are prone to — especially important when combined with the Noam
warmup schedule on small datasets like Multi30k.

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import re
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    depth = Q.size(-1)
    # Raw attention scores: (..., seq_q, seq_k)
    raw_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(depth)

    # Apply mask: blocked positions become -inf so softmax gives ~0
    if mask is not None:
        raw_scores = raw_scores.masked_fill(mask, float("-inf"))

    attention_weights = F.softmax(raw_scores, dim=-1)

    # Weighted aggregation over values
    context = torch.matmul(attention_weights, V)
    return context, attention_weights


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # Identify pad positions, then expand dims for broadcasting over heads
    pad_positions = (src == pad_idx)               # [batch, src_len]
    return pad_positions.unsqueeze(1).unsqueeze(2) # [batch, 1, 1, src_len]


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    seq_len = tgt.size(1)

    # Padding mask — True for <pad> tokens
    # Shape: [batch, 1, 1, tgt_len] → broadcasts to [batch, 1, tgt_len, tgt_len]
    padding_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal (upper-triangular) mask — True for future positions
    # Shape: [1, 1, tgt_len, tgt_len]
    upper_tri = torch.triu(
        torch.ones(seq_len, seq_len, device=tgt.device, dtype=torch.bool),
        diagonal=1
    ).unsqueeze(0).unsqueeze(0)

    # Union: mask out either PAD or future positions
    return padding_mask | upper_tri


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    NOTE: torch.nn.MultiheadAttention is NOT used.

    Projection matrices use no bias term, matching the paper.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # Individual projection matrices for Q, K, V and the output
        self.proj_q = nn.Linear(d_model, d_model, bias=False)
        self.proj_k = nn.Linear(d_model, d_model, bias=False)
        self.proj_v = nn.Linear(d_model, d_model, bias=False)
        self.proj_o = nn.Linear(d_model, d_model, bias=False)

        self.attn_dropout = nn.Dropout(p=dropout)

        # Store last attention weights for visualisation (Task 2.3)
        self.last_attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [batch, seq, d_model] → [batch, heads, seq, d_k]."""
        batch, seq, _ = x.shape
        # Reshape to (batch, seq, heads, d_k) then transpose to (batch, heads, seq, d_k)
        return x.view(batch, seq, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape [batch, heads, seq, d_k] → [batch, seq, d_model]."""
        batch, _, seq, _ = x.shape
        # Transpose back then flatten head dimension
        return x.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        # Project inputs to Q, K, V spaces and split into heads
        Q = self._split_heads(self.proj_q(query))  # [batch, heads, seq_q, d_k]
        K = self._split_heads(self.proj_k(key))    # [batch, heads, seq_k, d_k]
        V = self._split_heads(self.proj_v(value))  # [batch, heads, seq_k, d_k]

        # Scaled dot-product attention across all heads simultaneously
        attended, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attended: [batch, heads, seq_q, d_k]

        # Optionally apply dropout to attention weights (training regularisation)
        self.last_attn_weights = attn_weights.detach()

        # Merge heads and project back to d_model
        merged   = self._merge_heads(attended)      # [batch, seq_q, d_model]
        output   = self.proj_o(merged)              # [batch, seq_q, d_model]
        return output


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The encoding is pre-computed and stored as a non-trainable buffer.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout)

        # Pre-compute the full table of shape [1, max_len, d_model]
        encoding_table = torch.zeros(max_len, d_model)

        positions = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        # Compute denominators in log-space for numerical stability
        # div_term[i] = 1 / 10000^(2i / d_model)
        dimension_indices = torch.arange(0, d_model, 2, dtype=torch.float)
        div_term = torch.exp(dimension_indices * (-math.log(10000.0) / d_model))

        encoding_table[:, 0::2] = torch.sin(positions * div_term)
        encoding_table[:, 1::2] = torch.cos(positions * div_term)

        # Add batch dimension: [1, max_len, d_model]
        encoding_table = encoding_table.unsqueeze(0)

        # Register as buffer so it moves with .to(device) but isn't a parameter
        self.register_buffer("pe", encoding_table)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape with positional encoding added.
        """
        seq_len = x.size(1)
        # self.pe is [1, max_len, d_model]; slice to current sequence length
        x = x + self.pe[:, :seq_len, :]
        return self.dropout_layer(x)


# ══════════════════════════════════════════════════════════════════════
#  POSITION-WISE FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Applied identically to each position. Inner dimension d_ff is
    typically 4× d_model.

    Args:
        d_model (int)  : Input / output dimensionality.
        d_ff    (int)  : Inner-layer dimensionality.
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout  = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        # W1 + ReLU + dropout + W2
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  (Pre-LayerNorm variant)
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer using PRE-LayerNorm:

        x → LayerNorm → Self-Attention → Residual
          → LayerNorm → FFN            → Residual

    Pre-LN is chosen over the paper's Post-LN because it keeps
    gradient signals stable during the critical early training steps,
    avoiding the need for careful learning-rate warm-up tuning.

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_fwd   = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm_attn  = nn.LayerNorm(d_model)
        self.norm_ff    = nn.LayerNorm(d_model)
        self.drop_attn  = nn.Dropout(p=dropout)
        self.drop_ff    = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Sub-layer 1: self-attention with pre-norm
        residual = x
        normed   = self.norm_attn(x)
        attn_out = self.self_attn(normed, normed, normed, src_mask)
        x        = residual + self.drop_attn(attn_out)

        # Sub-layer 2: FFN with pre-norm
        residual = x
        normed   = self.norm_ff(x)
        ff_out   = self.feed_fwd(normed)
        x        = residual + self.drop_ff(ff_out)

        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER  (Pre-LayerNorm variant)
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer using PRE-LayerNorm:

        x → LayerNorm → Masked Self-Attn  → Residual
          → LayerNorm → Cross-Attn(memory) → Residual
          → LayerNorm → FFN               → Residual

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.masked_sa        = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn       = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_fwd         = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.drop1 = nn.Dropout(p=dropout)
        self.drop2 = nn.Dropout(p=dropout)
        self.drop3 = nn.Dropout(p=dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # Sub-layer 1: masked self-attention
        residual = x
        x        = residual + self.drop1(
            self.masked_sa(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask)
        )

        # Sub-layer 2: cross-attention (queries from decoder, keys/values from encoder)
        residual = x
        x        = residual + self.drop2(
            self.cross_attn(self.norm2(x), memory, memory, src_mask)
        )

        # Sub-layer 3: position-wise FFN
        residual = x
        x        = residual + self.drop3(self.feed_fwd(self.norm3(x)))

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with a final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        # Deep-copy so each layer has independent parameters
        self.layers   = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.final_norm = nn.LayerNorm(layer.norm_attn.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for enc_layer in self.layers:
            x = enc_layer(x, mask)
        return self.final_norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with a final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers     = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.final_norm = nn.LayerNorm(layer.norm1.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for dec_layer in self.layers:
            x = dec_layer(x, memory, src_mask, tgt_mask)
        return self.final_norm(x)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════
#  LIGHTWEIGHT VOCABULARY  (embedded so model.py is self-contained)
# ══════════════════════════════════════════════════════════════════════

class _InferenceVocab:
    """
    Minimal vocabulary object used at inference time.

    Built from a serialisable state-dict so model.py has no dependency
    on dataset.py at inference time.  The four special-token indices
    are fixed at the same positions used during training.
    """

    def __init__(
        self,
        word2idx: Dict[str, int],
        idx2word: Dict[int, str],
    ) -> None:
        self._w2i = word2idx
        self._i2w = idx2word

    # ── Special-token indices (fixed by convention in dataset.py) ──────
    @property
    def unk_idx(self) -> int: return 0
    @property
    def pad_idx(self) -> int: return 1
    @property
    def sos_idx(self) -> int: return 2
    @property
    def eos_idx(self) -> int: return 3

    def __len__(self) -> int:
        return len(self._w2i)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self._w2i.get(t, self.unk_idx) for t in tokens]

    def lookup_token(self, idx: int) -> str:
        return self._i2w.get(idx, "<unk>")

    # torchtext-style accessor
    @property
    def itos(self) -> Dict[int, str]:
        return self._i2w

    # ── Serialisation ─────────────────────────────────────────────────

    def state_dict(self) -> dict:
        """Return a plain-dict representation safe for torch.save."""
        return {"word2idx": self._w2i, "idx2word": self._i2w}

    @classmethod
    def from_state(cls, state: dict) -> "_InferenceVocab":
        """Reconstruct from a state-dict produced by state_dict()."""
        # Keys in idx2word are stored as ints by torch.save;
        # ensure they are ints regardless of JSON/pickle source.
        idx2word = {int(k): v for k, v in state["idx2word"].items()}
        return cls(state["word2idx"], idx2word)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for German→English machine translation.

    Design choice: Pre-LayerNorm (Pre-LN) throughout encoder and decoder.
    Rationale: Pre-LN keeps gradient magnitudes stable from the very
    first training step, preventing the early-divergence problem that
    Post-LN transformers exhibit when combined with the Noam warm-up
    schedule on small datasets like Multi30k.

    ── Self-contained inference ───────────────────────────────────────
    When called with NO arguments, __init__ will:
      1. Download the trained checkpoint from Google Drive via gdown.
      2. Read the model config (vocab sizes, depth, heads, …) from it.
      3. Build the full architecture with those dimensions.
      4. Load the saved weight tensors.
      5. Reconstruct source/target vocabularies from the checkpoint.
      6. Load the spaCy German tokeniser.

    After that, `model.infer(german_sentence)` works without any
    additional setup.

    ── Training ──────────────────────────────────────────────────────
    Pass explicit vocab sizes and set gdrive_id to the sentinel string
    (or any non-ID string) to skip the checkpoint download:

        model = Transformer(
            src_vocab_size = len(src_vocab),
            tgt_vocab_size = len(tgt_vocab),
            gdrive_id      = "SKIP",      # no download
        )

    Args:
        src_vocab_size (int | None): Source vocab size.  None → read from ckpt.
        tgt_vocab_size (int | None): Target vocab size.  None → read from ckpt.
        d_model        (int)       : Model dimensionality            (default 256).
        N              (int)       : Encoder/decoder depth           (default 3).
        num_heads      (int)       : Parallel attention heads        (default 8).
        d_ff           (int)       : FFN inner width                 (default 512).
        dropout        (float)     : Dropout probability             (default 0.1).
        gdrive_id      (str)       : Google Drive file-ID for weights download.
                                     Set to "SKIP" to skip download entirely.
        checkpoint_path(str)       : Local path for the downloaded checkpoint.
    """

    # ── Sentinel that means "do not download" ─────────────────────────
    _SKIP_DOWNLOAD = "SKIP"

    def __init__(
        self,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        d_model:        int            = 256,
        N:              int            = 3,
        num_heads:      int            = 8,
        d_ff:           int            = 512,
        dropout:        float          = 0.1,
        gdrive_id:      str            = "1jD9mICFAzSmWYe_u70gIDog27-Vsn8Ta",
        checkpoint_path:str            = "best_checkpoint.pt",
    ) -> None:
        super().__init__()

        # ── Step 1: Download checkpoint (if gdrive_id looks real) ──────
        ckpt = self._maybe_load_checkpoint(gdrive_id, checkpoint_path)

        # ── Step 2: Resolve architecture dimensions ────────────────────
        # Preference order: explicit arg > checkpoint config > default
        if ckpt is not None and "model_config" in ckpt:
            cfg = ckpt["model_config"]
            src_vocab_size = src_vocab_size or cfg["src_vocab_size"]
            tgt_vocab_size = tgt_vocab_size or cfg["tgt_vocab_size"]
            d_model   = cfg.get("d_model",   d_model)
            N         = cfg.get("N",         N)
            num_heads = cfg.get("num_heads", num_heads)
            d_ff      = cfg.get("d_ff",      d_ff)
            dropout   = cfg.get("dropout",   dropout)

        if src_vocab_size is None or tgt_vocab_size is None:
            raise ValueError(
                "src_vocab_size and tgt_vocab_size must be provided "
                "either explicitly or via a checkpoint."
            )

        # ── Step 3: Build all sub-modules ─────────────────────────────
        self._assemble(src_vocab_size, tgt_vocab_size,
                       d_model, N, num_heads, d_ff, dropout)
        self._init_weights()

        # ── Step 4: Restore trained weights ───────────────────────────
        if ckpt is not None and "model_state_dict" in ckpt:
            self.load_state_dict(ckpt["model_state_dict"])
            print("[Transformer] Model weights restored from checkpoint.")

        # ── Step 5: Reconstruct vocabularies ──────────────────────────
        if ckpt is not None and "src_vocab_state" in ckpt:
            self.src_vocab = _InferenceVocab.from_state(ckpt["src_vocab_state"])
            self.tgt_vocab = _InferenceVocab.from_state(ckpt["tgt_vocab_state"])
            print(f"[Transformer] Vocabularies restored  "
                  f"(src={len(self.src_vocab)}, tgt={len(self.tgt_vocab)}).")
        else:
            # Training path — caller will attach vocabs via save_checkpoint
            self.src_vocab = None
            self.tgt_vocab = None

        # ── Step 6: Load German spaCy tokeniser OR fallback ───────────
        self._de_tokenize = None
        try:
            import spacy
            _de_nlp = spacy.load("de_core_news_sm")
            # Capture in closure so the lambda is self-contained
            self._de_tokenize = (
                lambda text, _nlp=_de_nlp:
                    [tok.text.lower() for tok in _nlp.tokenizer(text)]
            )
            print("[Transformer] spaCy German tokeniser ready.")
        except (OSError, ImportError) as e:
            print(
                f"[Transformer] WARNING: spaCy/de_core_news_sm not found ({e}). "
                "Falling back to basic regex tokeniser."
            )
            # Basic fallback tokeniser that splits words and punctuation
            self._de_tokenize = lambda text: re.findall(r'\w+|[^\w\s]', text.lower())

    # ── Internal helpers ──────────────────────────────────────────────

    @staticmethod
    def _maybe_load_checkpoint(gdrive_id: str, path: str) -> Optional[dict]:
        """
        Download checkpoint from Google Drive (if not already present) and
        return its contents as a dict, or None if unavailable / skipped.
        """
        if gdrive_id == Transformer._SKIP_DOWNLOAD:
            return None
        if "YOUR_GDRIVE" in gdrive_id or len(gdrive_id) < 10:
            # Placeholder ID — silently skip
            return None

        if not os.path.exists(path):
            try:
                import gdown
                print(f"[Transformer] Downloading checkpoint from Drive …")
                gdown.download(id=gdrive_id, output=path, quiet=False)
            except Exception as exc:
                print(f"[Transformer] gdown failed: {exc}")
                return None

        if not os.path.exists(path):
            return None

        try:
            ckpt = torch.load(path, map_location="cpu")
            print(f"[Transformer] Checkpoint read from '{path}'.")
            return ckpt
        except Exception as exc:
            print(f"[Transformer] Could not parse checkpoint: {exc}")
            return None

    def _assemble(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:        int,
        N:              int,
        num_heads:      int,
        d_ff:           int,
        dropout:        float,
    ) -> None:
        """Instantiate all learnable sub-modules."""
        self.d_model = d_model

        self.src_embed   = nn.Embedding(src_vocab_size, d_model, padding_idx=1)
        self.tgt_embed   = nn.Embedding(tgt_vocab_size, d_model, padding_idx=1)
        self.pos_enc     = PositionalEncoding(d_model, dropout)

        enc_layer        = EncoderLayer(d_model, num_heads, d_ff, dropout)
        dec_layer        = DecoderLayer(d_model, num_heads, d_ff, dropout)
        self.encoder     = Encoder(enc_layer, N)
        self.decoder     = Decoder(dec_layer, N)
        self.output_proj = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def _init_weights(self) -> None:
        """Xavier-uniform for linear/embedding weights; zero bias."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    # ── AUTOGRADER HOOKS (signatures must not change) ─────────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # §3.4: scale embeddings by sqrt(d_model) before adding positional enc
        embedded   = self.src_embed(src) * math.sqrt(self.d_model)
        positioned = self.pos_enc(embedded)
        return self.encoder(positioned, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        embedded   = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        positioned = self.pos_enc(embedded)
        dec_out    = self.decoder(positioned, memory, src_mask, tgt_mask)
        return self.output_proj(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def infer(self, src_sentence: str) -> str:
        """
        End-to-end German → English translation for a single sentence.

        Full pipeline:
            raw DE string
            → spaCy tokenisation (lowercase) OR fallback tokenisation
            → integer encoding with src_vocab  (+<sos>/<eos>)
            → Transformer encoder
            → autoregressive greedy decoder
            → integer decoding with tgt_vocab
            → detokenised English string

        All resources (weights, vocabs, tokeniser) are loaded inside
        __init__; no external setup is required.

        Args:
            src_sentence : Raw German input string.

        Returns:
            Translated English string, special tokens stripped.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "Vocabularies not loaded. "
                "Make sure the checkpoint was saved with vocab state dicts."
            )

        self.eval()
        device = next(self.parameters()).device

        # ── Tokenise and encode source ─────────────────────────────────
        de_tokens = self._de_tokenize(src_sentence)
        src_ids   = (
            [self.src_vocab.sos_idx]
            + self.src_vocab.encode(de_tokens)
            + [self.src_vocab.eos_idx]
        )
        src_tensor = torch.tensor(src_ids, dtype=torch.long,
                                  device=device).unsqueeze(0)   # [1, src_len]
        src_mask   = make_src_mask(src_tensor, pad_idx=self.src_vocab.pad_idx)

        sos_id = self.tgt_vocab.sos_idx
        eos_id = self.tgt_vocab.eos_idx
        pad_id = self.tgt_vocab.pad_idx

        # ── Autoregressive greedy decoding ────────────────────────────
        with torch.no_grad():
            memory    = self.encode(src_tensor, src_mask)
            generated = torch.tensor([[sos_id]], dtype=torch.long, device=device)
            max_steps = src_tensor.size(1) + 50

            for _ in range(max_steps):
                tgt_mask  = make_tgt_mask(generated, pad_idx=pad_id)
                logits    = self.decode(memory, src_mask, generated, tgt_mask)
                next_tok  = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=1)
                if next_tok.item() == eos_id:
                    break

        # ── Decode indices → English words ────────────────────────────
        skip = {sos_id, eos_id, pad_id}
        words = [
            self.tgt_vocab.lookup_token(idx)
            for idx in generated[0].tolist()
            if idx not in skip
        ]
        return " ".join(words)