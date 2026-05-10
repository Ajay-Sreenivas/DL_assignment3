"""
dataset.py — Multi30k Dataset, Vocabulary, and DataLoader Utilities
DA6401 Assignment 3: "Attention Is All You Need"

Pipeline:
    1. Load bentrevett/multi30k from HuggingFace datasets hub.
    2. Tokenise German (de_core_news_sm) and English (en_core_web_sm)
       using spaCy — lower-cased, keeping only the surface form.
    3. Build vocabulary from training split only (min_freq=2).
       Special tokens: <unk>=0, <pad>=1, <sos>=2, <eos>=3.
    4. Encode every sentence as a list of integer indices.
    5. Expose a PyTorch Dataset + collate_fn for use with DataLoader.
"""

from __future__ import annotations

import spacy
import torch
from collections import Counter
from typing import List, Tuple, Optional, Dict
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocabulary:
    """
    Simple word-level vocabulary with four reserved special tokens:

        Index 0 → <unk>   (unknown word)
        Index 1 → <pad>   (padding)
        Index 2 → <sos>   (start-of-sequence)
        Index 3 → <eos>   (end-of-sequence)

    Args:
        min_freq (int): Minimum token frequency required to enter vocab.
    """

    _SPECIALS = ["<unk>", "<pad>", "<sos>", "<eos>"]

    def __init__(self, min_freq: int = 2) -> None:
        self.min_freq = min_freq

        # Initialise with specials at fixed indices
        self._word_to_idx: Dict[str, int] = {
            tok: idx for idx, tok in enumerate(self._SPECIALS)
        }
        self._idx_to_word: Dict[int, str] = {
            idx: tok for idx, tok in enumerate(self._SPECIALS)
        }

    # ── Index constants ────────────────────────────────────────────────
    @property
    def unk_idx(self) -> int:  return 0
    @property
    def pad_idx(self) -> int:  return 1
    @property
    def sos_idx(self) -> int:  return 2
    @property
    def eos_idx(self) -> int:  return 3

    # ── torchtext-compatible accessors ────────────────────────────────
    @property
    def itos(self) -> Dict[int, str]:
        return self._idx_to_word

    def lookup_token(self, idx: int) -> str:
        return self._idx_to_word.get(idx, "<unk>")

    def lookup_index(self, token: str) -> int:
        return self._word_to_idx.get(token, self.unk_idx)

    def __len__(self) -> int:
        return len(self._word_to_idx)

    # ── Building ────────────────────────────────────────────────────────

    def build_from_token_lists(self, all_token_lists: List[List[str]]) -> None:
        """
        Add tokens that appear at least min_freq times across all_token_lists.

        Args:
            all_token_lists : Iterable of tokenised sentences.
        """
        freq: Counter = Counter()
        for tokens in all_token_lists:
            freq.update(tokens)

        for word, count in sorted(freq.items()):  # sorted for reproducibility
            if count >= self.min_freq and word not in self._word_to_idx:
                new_idx = len(self._word_to_idx)
                self._word_to_idx[word] = new_idx
                self._idx_to_word[new_idx] = word

    # ── Encoding / decoding ─────────────────────────────────────────────

    def encode(self, tokens: List[str]) -> List[int]:
        """Map a list of token strings to their integer indices."""
        return [self._word_to_idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, indices: List[int], skip_special: bool = True) -> List[str]:
        """Map integer indices back to token strings."""
        skip_set = {self.unk_idx, self.pad_idx, self.sos_idx, self.eos_idx} \
                   if skip_special else set()
        return [self._idx_to_word.get(i, "<unk>")
                for i in indices if i not in skip_set]

    # ── Serialisation for checkpoint embedding ─────────────────────────

    def state_dict(self) -> dict:
        """
        Return a plain-dict snapshot suitable for torch.save.

        The returned dict is compatible with model._InferenceVocab.from_state(),
        allowing the Transformer to reconstruct the vocabulary from a checkpoint
        without importing dataset.py at inference time.
        """
        return {
            "word2idx": dict(self._word_to_idx),
            "idx2word": {int(k): v for k, v in self._idx_to_word.items()},
        }


# ══════════════════════════════════════════════════════════════════════
#  TORCH DATASET WRAPPER
# ══════════════════════════════════════════════════════════════════════

class TranslationDataset(Dataset):
    """
    Wraps a list of (src_indices, tgt_indices) pairs as a PyTorch Dataset.

    Args:
        pairs : List of (src_tensor, tgt_tensor) token-index tensors.
    """

    def __init__(self, pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> None:
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pairs[idx]


def collate_fn(
    batch: List[Tuple[torch.Tensor, torch.Tensor]],
    src_pad_idx: int = 1,
    tgt_pad_idx: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a batch of (src, tgt) pairs to equal length within each side.

    Returns:
        src_batch : [batch, max_src_len]
        tgt_batch : [batch, max_tgt_len]
    """
    src_seqs, tgt_seqs = zip(*batch)
    src_padded = pad_sequence(src_seqs, batch_first=True, padding_value=src_pad_idx)
    tgt_padded = pad_sequence(tgt_seqs, batch_first=True, padding_value=tgt_pad_idx)
    return src_padded, tgt_padded


# ══════════════════════════════════════════════════════════════════════
#  MAIN DATASET CLASS
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset:
    """
    Manages the full lifecycle of the Multi30k German→English dataset.

    Usage:
        # Build vocabs on training data
        train_mgr = Multi30kDataset('train')
        train_mgr.build_vocab()
        train_mgr.process_data()

        # Reuse vocabs for val/test (vocabulary must NOT be rebuilt)
        val_mgr = Multi30kDataset('validation',
                                  src_vocab=train_mgr.src_vocab,
                                  tgt_vocab=train_mgr.tgt_vocab)
        val_mgr.process_data()

    Args:
        split    (str)       : One of 'train', 'validation', 'test'.
        src_vocab (Vocabulary): Optional pre-built source vocabulary.
        tgt_vocab (Vocabulary): Optional pre-built target vocabulary.
        min_freq  (int)      : Minimum token frequency for vocab building.
    """

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocabulary] = None,
        tgt_vocab: Optional[Vocabulary] = None,
        min_freq: int = 2,
    ) -> None:
        self.split     = split
        self.min_freq  = min_freq
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.processed_pairs: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

        # ── Load raw data ───────────────────────────────────────────────
        from datasets import load_dataset as hf_load
        print(f"[Multi30kDataset] Loading split='{split}' …")
        raw = hf_load("bentrevett/multi30k", split=split)
        # Each example has keys 'en' and 'de'
        self.raw_de: List[str] = [ex["de"] for ex in raw]
        self.raw_en: List[str] = [ex["en"] for ex in raw]
        print(f"[Multi30kDataset] {len(self.raw_de)} examples loaded.")

        # ── Load spaCy tokenisers ───────────────────────────────────────
        try:
            self._de_nlp = spacy.load("de_core_news_sm")
        except OSError:
            raise OSError(
                "German spaCy model not found. "
                "Run: python -m spacy download de_core_news_sm"
            )
        try:
            self._en_nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise OSError(
                "English spaCy model not found. "
                "Run: python -m spacy download en_core_web_sm"
            )

    # ── Tokenisation helpers ──────────────────────────────────────────

    def _tokenize_de(self, text: str) -> List[str]:
        """Tokenise a German sentence into lowercase surface forms."""
        return [tok.text.lower() for tok in self._de_nlp.tokenizer(text)]

    def _tokenize_en(self, text: str) -> List[str]:
        """Tokenise an English sentence into lowercase surface forms."""
        return [tok.text.lower() for tok in self._en_nlp.tokenizer(text)]

    # Public accessor so Transformer.infer can reuse the same tokeniser
    @property
    def de_tokenizer(self):
        return self._tokenize_de

    # ── API ───────────────────────────────────────────────────────────

    def build_vocab(self) -> None:
        """
        Build source (German) and target (English) vocabularies from the
        current split.  Should only be called on the TRAINING split to
        avoid data leakage.
        """
        print("[Multi30kDataset] Building vocabulary …")
        de_token_lists = [self._tokenize_de(s) for s in self.raw_de]
        en_token_lists = [self._tokenize_en(s) for s in self.raw_en]

        self.src_vocab = Vocabulary(min_freq=self.min_freq)
        self.src_vocab.build_from_token_lists(de_token_lists)

        self.tgt_vocab = Vocabulary(min_freq=self.min_freq)
        self.tgt_vocab.build_from_token_lists(en_token_lists)

        print(f"[Multi30kDataset] src_vocab size: {len(self.src_vocab)}")
        print(f"[Multi30kDataset] tgt_vocab size: {len(self.tgt_vocab)}")

    def process_data(self) -> None:
        """
        Tokenise all sentences and convert to integer index tensors.
        Wraps each sentence with <sos> and <eos> tokens.

        Result stored in self.processed_pairs as a list of
        (src_tensor, tgt_tensor) tuples.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError("Call build_vocab() or supply vocabularies first.")

        print(f"[Multi30kDataset] Processing '{self.split}' split …")
        pairs = []
        for de_sent, en_sent in zip(self.raw_de, self.raw_en):
            de_tok = self._tokenize_de(de_sent)
            en_tok = self._tokenize_en(en_sent)

            src_indices = (
                [self.src_vocab.sos_idx]
                + self.src_vocab.encode(de_tok)
                + [self.src_vocab.eos_idx]
            )
            tgt_indices = (
                [self.tgt_vocab.sos_idx]
                + self.tgt_vocab.encode(en_tok)
                + [self.tgt_vocab.eos_idx]
            )

            pairs.append((
                torch.tensor(src_indices, dtype=torch.long),
                torch.tensor(tgt_indices, dtype=torch.long),
            ))

        self.processed_pairs = pairs
        print(f"[Multi30kDataset] {len(pairs)} pairs processed.")

    def get_dataset(self) -> TranslationDataset:
        """Return a PyTorch-compatible Dataset."""
        if self.processed_pairs is None:
            raise RuntimeError("Call process_data() first.")
        return TranslationDataset(self.processed_pairs)

    def get_dataloader(
        self,
        batch_size: int = 128,
        shuffle: bool = True,
        num_workers: int = 0,
    ) -> DataLoader:
        """
        Return a DataLoader with padding-aware collation.

        Args:
            batch_size  : Number of sentence pairs per batch.
            shuffle     : Shuffle samples each epoch (True for training).
            num_workers : Parallel data workers.
        """
        dataset = self.get_dataset()
        pad_src = self.src_vocab.pad_idx
        pad_tgt = self.tgt_vocab.pad_idx

        def _collate(batch):
            return collate_fn(batch, src_pad_idx=pad_src, tgt_pad_idx=pad_tgt)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=_collate,
            num_workers=num_workers,
        )