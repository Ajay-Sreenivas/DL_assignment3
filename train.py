"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need", §5.4.

    Instead of a hard one-hot target, the model is trained against a
    smoothed distribution:
        y_smooth[correct] = 1 - eps + eps / vocab_size
        y_smooth[other]   = eps / vocab_size
        y_smooth[<pad>]   = 0  (always zero — we never want to predict <pad>)

    This acts as a regulariser: it prevents the model from becoming
    over-confident on the training set, which would cause the softmax
    outputs to become extremely peaked and gradients to vanish.

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — given zero probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing   # mass on the correct class

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [N, vocab_size]  (raw model output, pre-softmax)
            target : shape [N]              (gold token indices)

        Returns:
            Scalar mean loss (KL-divergence between smoothed target and model).
        """
        # Log-probabilities from the model
        log_probs = F.log_softmax(logits, dim=-1)  # [N, vocab_size]

        # Build smoothed target distribution
        # Every class starts with eps / vocab_size
        smooth_dist = torch.full_like(log_probs, self.smoothing / self.vocab_size)

        # Add extra mass to the correct class
        smooth_dist.scatter_(
            1,
            target.unsqueeze(1),
            self.confidence + self.smoothing / self.vocab_size
        )

        # Zero out the <pad> column — we never want to predict <pad>
        smooth_dist[:, self.pad_idx] = 0.0

        # Mask out positions where the TARGET itself is <pad>
        pad_positions = (target == self.pad_idx)
        smooth_dist[pad_positions] = 0.0

        # KL(smooth_dist || model) == -sum(smooth_dist * log_probs) - H(smooth_dist)
        # Since H is constant w.r.t. model, minimise negative sum
        loss = -(smooth_dist * log_probs).sum(dim=-1)    # [N]

        # Only average over non-pad positions
        non_pad_count = (~pad_positions).sum().clamp(min=1)
        return loss.sum() / non_pad_count


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    During training:
        - Computes forward pass with teacher forcing
        - Backpropagates gradients
        - Clips gradient norms to 1.0 (standard Transformer practice)
        - Steps optimizer and Noam scheduler

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during evaluation).
        scheduler  : NoamScheduler instance (None during evaluation).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, backward pass + scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over all batches (float).
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0
    batch_count  = 0
    epoch_start  = time.time()

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, (src_batch, tgt_batch) in enumerate(data_iter):
            src_batch = src_batch.to(device)   # [batch, src_len]
            tgt_batch = tgt_batch.to(device)   # [batch, tgt_len]

            # Teacher-forcing: decoder input excludes final <eos>,
            # target output excludes leading <sos>
            dec_input  = tgt_batch[:, :-1]     # [batch, tgt_len - 1]
            dec_target = tgt_batch[:, 1:]      # [batch, tgt_len - 1]

            # Build masks
            src_mask = make_src_mask(src_batch, pad_idx=1)
            tgt_mask = make_tgt_mask(dec_input, pad_idx=1)

            # Forward pass
            logits = model(src_batch, dec_input, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            # Flatten for loss: [batch * (tgt_len-1), vocab_size]
            batch_size, seq_len, vocab_sz = logits.shape
            flat_logits = logits.contiguous().view(batch_size * seq_len, vocab_sz)
            flat_target = dec_target.contiguous().view(-1)

            loss = loss_fn(flat_logits, flat_target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping: prevents exploding gradients
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Accumulate stats (weight by non-pad token count)
            non_pad = (dec_target != 1).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad
            batch_count  += 1

    avg_loss  = total_loss / max(total_tokens, 1)
    elapsed   = time.time() - epoch_start
    mode_str  = "TRAIN" if is_train else "EVAL"
    print(
        f"[Epoch {epoch_num:03d} | {mode_str}] "
        f"loss={avg_loss:.4f}  "
        f"ppl={torch.exp(torch.tensor(avg_loss)).item():.2f}  "
        f"time={elapsed:.1f}s  "
        f"batches={batch_count}"
    )
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    At each step the decoder receives all previously generated tokens
    and picks the argmax of the next-token distribution.

    Args:
        model        : Trained Transformer (should be in eval mode).
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol at position 0; stops when end_symbol
             is produced or max_len is reached.
    """
    src      = src.to(device)
    src_mask = src_mask.to(device)

    # Encode the source once; reuse memory at every decoding step
    with torch.no_grad():
        memory = model.encode(src, src_mask)     # [1, src_len, d_model]

    # Initialise decoder sequence with <sos>
    generated = torch.tensor([[start_symbol]], dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(generated, pad_idx=1).to(device)

        with torch.no_grad():
            logits = model.decode(memory, src_mask, generated, tgt_mask)
            # logits: [1, current_len, vocab_size]

        # Greedy: take the token with highest probability at the last position
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        generated  = torch.cat([generated, next_token], dim=1)       # [1, len+1]

        if next_token.item() == end_symbol:
            break

    return generated   # [1, out_len]


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Uses the HuggingFace `evaluate` library (sacrebleu backend) to
    compute corpus BLEU over all test sentences.

    Args:
        model           : Trained Transformer.
        test_dataloader : DataLoader for the test split.
        tgt_vocab       : Vocabulary with .sos_idx, .eos_idx, .pad_idx,
                          and .lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode steps per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, 0–100).
    """
    try:
        import evaluate as hf_eval
        bleu_metric = hf_eval.load("bleu")
        use_hf = True
    except Exception:
        use_hf = False

    model.eval()

    predictions: list = []
    references:  list = []

    sos = tgt_vocab.sos_idx
    eos = tgt_vocab.eos_idx
    pad = tgt_vocab.pad_idx

    with torch.no_grad():
        for src_batch, tgt_batch in test_dataloader:
            src_batch = src_batch.to(device)

            for i in range(src_batch.size(0)):
                # Single-sentence inference
                src_single   = src_batch[i].unsqueeze(0)               # [1, src_len]
                src_mask     = make_src_mask(src_single, pad_idx=1).to(device)

                output_ids   = greedy_decode(
                    model, src_single, src_mask,
                    max_len=max_len,
                    start_symbol=sos,
                    end_symbol=eos,
                    device=device,
                )

                # Decode generated sequence — strip <sos>/<eos>/<pad>
                pred_tokens = []
                for idx in output_ids[0].tolist():
                    if idx in (sos, pad):
                        continue
                    if idx == eos:
                        break
                    pred_tokens.append(tgt_vocab.lookup_token(idx))

                # Decode reference — strip specials
                ref_tokens = []
                for idx in tgt_batch[i].tolist():
                    if idx in (sos, pad):
                        continue
                    if idx == eos:
                        break
                    ref_tokens.append(tgt_vocab.lookup_token(idx))

                predictions.append(" ".join(pred_tokens))
                references.append(" ".join(ref_tokens))

    # Compute corpus BLEU
    if use_hf and predictions:
        result = bleu_metric.compute(
            predictions=predictions,
            references=[[r] for r in references],
        )
        bleu_score = result["bleu"] * 100.0
    else:
        # Fallback: sentence-level sacrebleu via sacrebleu package
        try:
            from sacrebleu.metrics import BLEU as SacreBLEU
            bleu_fn    = SacreBLEU(effective_order=True)
            bleu_score = bleu_fn.corpus_score(
                predictions,
                [references],
            ).score
        except ImportError:
            # Last resort: simple NLTK BLEU
            from nltk.translate.bleu_score import corpus_bleu
            tokenised_refs  = [[r.split()] for r in references]
            tokenised_hyps  = [p.split() for p in predictions]
            bleu_score      = corpus_bleu(tokenised_refs, tokenised_hyps) * 100.0

    print(f"[evaluate_bleu] Corpus BLEU = {bleu_score:.2f}")
    return bleu_score


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Persist model + optimiser + scheduler + vocabulary state to disk.

    Saved dict keys:
        'epoch'                : Current epoch number
        'model_state_dict'     : All learnable weight tensors
        'optimizer_state_dict' : Adam state (moments, step counts)
        'scheduler_state_dict' : Noam scheduler state
        'model_config'         : kwargs to reconstruct Transformer(...)
        'src_vocab_state'      : Source vocabulary (word2idx + idx2word)
        'tgt_vocab_state'      : Target vocabulary (word2idx + idx2word)

    The two vocab state dicts are consumed by Transformer.__init__ when
    the model is reconstructed at inference time via Transformer() with
    no arguments — they make the .pt file fully self-contained.

    Args:
        model     : Transformer with .src_vocab and .tgt_vocab set.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : Destination file path (default 'checkpoint.pt').
    """
    arch_config = {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model":        model.d_model,
        "N":              len(model.encoder.layers),
        "num_heads":      model.encoder.layers[0].self_attn.num_heads,
        "d_ff":           model.encoder.layers[0].feed_fwd.linear1.out_features,
        "dropout":        model.encoder.layers[0].drop_attn.p,
    }

    payload = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "model_config":         arch_config,
    }

    # Embed vocabulary state so Transformer() can reconstruct itself
    # from the single .pt file without importing dataset.py.
    if getattr(model, "src_vocab", None) is not None:
        payload["src_vocab_state"] = model.src_vocab.state_dict()
    if getattr(model, "tgt_vocab", None) is not None:
        payload["tgt_vocab_state"] = model.tgt_vocab.state_dict()

    torch.save(payload, path)
    vocab_saved = "src_vocab_state" in payload
    print(f"[Checkpoint] Epoch {epoch} saved → '{path}'  (vocab embedded: {vocab_saved})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimiser / scheduler) from disk.

    Args:
        path      : Path to the checkpoint file.
        model     : Transformer with matching architecture.
        optimizer : Optimiser to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch number stored in the checkpoint.
    """
    ckpt = torch.load(path, map_location="cpu")

    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[Checkpoint] Model weights loaded from '{path}'.")

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt.get("epoch", 0)


# ══════════════════════════════════════════════════════════════════════
#  FULL EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    End-to-end training run with W&B logging.

    Hyperparameter defaults follow the "base" Transformer from the paper
    but scaled for Multi30k (smaller d_model / N to fit the dataset size
    and available compute).
    """
    import wandb
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler

    # ── Hyperparameters ────────────────────────────────────────────────
    cfg = dict(
        d_model      = 256,       # smaller than paper's 512 for Multi30k
        N            = 3,         # 3 encoder + 3 decoder layers
        num_heads    = 8,
        d_ff         = 512,
        dropout      = 0.1,
        label_smooth = 0.1,
        warmup_steps = 4000,
        batch_size   = 128,
        num_epochs   = 15,
        lr_base      = 1.0,       # Noam controls the actual LR magnitude
        seed         = 42,
    )

    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Experiment] Using device: {device}")

    # ── W&B init ───────────────────────────────────────────────────────
    wandb.init(project="da6401-a3", config=cfg, name="transformer-baseline")

    # ── Dataset ────────────────────────────────────────────────────────
    train_mgr = Multi30kDataset("train")
    train_mgr.build_vocab()
    train_mgr.process_data()

    val_mgr = Multi30kDataset(
        "validation",
        src_vocab=train_mgr.src_vocab,
        tgt_vocab=train_mgr.tgt_vocab,
    )
    val_mgr.process_data()

    test_mgr = Multi30kDataset(
        "test",
        src_vocab=train_mgr.src_vocab,
        tgt_vocab=train_mgr.tgt_vocab,
    )
    test_mgr.process_data()

    train_loader = train_mgr.get_dataloader(batch_size=cfg["batch_size"], shuffle=True)
    val_loader   = val_mgr.get_dataloader(batch_size=cfg["batch_size"], shuffle=False)
    test_loader  = test_mgr.get_dataloader(batch_size=cfg["batch_size"], shuffle=False)

    src_vocab = train_mgr.src_vocab
    tgt_vocab = train_mgr.tgt_vocab

    # ── Model ─────────────────────────────────────────────────────────
    # Pass gdrive_id="SKIP" so __init__ does NOT try to download anything
    # during training — we supply explicit vocab sizes instead.
    model = Transformer(
        src_vocab_size = len(src_vocab),
        tgt_vocab_size = len(tgt_vocab),
        d_model        = cfg["d_model"],
        N              = cfg["N"],
        num_heads      = cfg["num_heads"],
        d_ff           = cfg["d_ff"],
        dropout        = cfg["dropout"],
        gdrive_id      = "SKIP",
    ).to(device)

    # Attach vocabulary objects NOW so save_checkpoint can embed them.
    # save_checkpoint reads model.src_vocab / model.tgt_vocab to produce
    # a self-contained .pt file that Transformer() (no-arg) can load.
    model.src_vocab = src_vocab
    model.tgt_vocab = tgt_vocab

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Experiment] Trainable parameters: {num_params:,}")
    wandb.config.update({"num_params": num_params})

    # ── Optimiser + Scheduler + Loss ───────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr_base"],
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = NoamScheduler(
        optimizer,
        d_model=cfg["d_model"],
        warmup_steps=cfg["warmup_steps"],
    )
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(tgt_vocab),
        pad_idx=tgt_vocab.pad_idx,
        smoothing=cfg["label_smooth"],
    )

    best_val_loss = float("inf")
    best_ckpt     = "best_checkpoint.pt"

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(cfg["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn,
            optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn,
            None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        current_lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "lr":         current_lr,
        })

        # Save best model by validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path=best_ckpt)

        # Also save periodic checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            path=f"checkpoint_epoch_{epoch:03d}.pt",
        )

    # ── Test BLEU ──────────────────────────────────────────────────────
    # Reload best weights; vocabs are already on model from training loop
    print("[Experiment] Loading best checkpoint for final evaluation …")
    load_checkpoint(best_ckpt, model)
    bleu = evaluate_bleu(model, test_loader, tgt_vocab, device=device)
    wandb.log({"test_bleu": bleu})
    print(f"[Experiment] Final Test BLEU: {bleu:.2f}")

    # ── Demo via self-contained infer() ───────────────────────────────
    # model.src_vocab / model.tgt_vocab are already set from training;
    # model._de_tokenize was loaded in __init__. Calling infer() here
    # mirrors exactly what the autograder does: model.infer(sentence).
    demo_sentence = "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche."
    translation   = model.infer(demo_sentence)
    print(f"\n[Demo] DE: {demo_sentence}")
    print(f"[Demo] EN: {translation}")
    wandb.log({"demo_translation": translation})

    wandb.finish()


# ══════════════════════════════════════════════════════════════════════
#  ABLATION HELPERS  (for W&B Report experiments)
# ══════════════════════════════════════════════════════════════════════

def run_ablation_no_noam(cfg: dict, train_loader, val_loader, device: str) -> None:
    """
    Section 2.1 — Fixed LR baseline (no warmup, constant lr=1e-4).
    Logs curves to the currently active W&B run.
    """
    import wandb
    from dataset import Multi30kDataset
    from lr_scheduler import NoamScheduler

    src_vocab_size = cfg["src_vocab_size"]
    tgt_vocab_size = cfg["tgt_vocab_size"]

    model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model=cfg["d_model"], N=cfg["N"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn   = LabelSmoothingLoss(tgt_vocab_size, pad_idx=1, smoothing=0.1)

    for epoch in range(cfg["num_epochs"]):
        t_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, None,
            epoch_num=epoch, is_train=True, device=device,
        )
        v_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )
        wandb.log({"fixed_lr/train_loss": t_loss, "fixed_lr/val_loss": v_loss,
                   "epoch": epoch})


def run_ablation_no_scaling(cfg: dict, train_loader, val_loader, device: str) -> None:
    """
    Section 2.2 — Transformer without the 1/sqrt(dk) scaling factor.

    We monkey-patch scaled_dot_product_attention to omit the scale.
    This is used only for the ablation experiment.
    """
    import wandb
    import model as model_module

    # Save original and replace with unscaled version
    _original_sdpa = model_module.scaled_dot_product_attention

    def unscaled_attention(Q, K, V, mask=None):
        """Dot-product attention WITHOUT the sqrt(dk) scale."""
        raw_scores = torch.matmul(Q, K.transpose(-2, -1))   # no division
        if mask is not None:
            raw_scores = raw_scores.masked_fill(mask, float("-inf"))
        weights = torch.nn.functional.softmax(raw_scores, dim=-1)
        return torch.matmul(weights, V), weights

    model_module.scaled_dot_product_attention = unscaled_attention

    src_vocab_size = cfg["src_vocab_size"]
    tgt_vocab_size = cfg["tgt_vocab_size"]

    ablation_model = Transformer(
        src_vocab_size, tgt_vocab_size,
        d_model=cfg["d_model"], N=cfg["N"],
        num_heads=cfg["num_heads"], d_ff=cfg["d_ff"],
        dropout=cfg["dropout"],
    ).to(device)

    from lr_scheduler import NoamScheduler
    optimizer  = torch.optim.Adam(ablation_model.parameters(),
                                   lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    scheduler  = NoamScheduler(optimizer, cfg["d_model"], cfg["warmup_steps"])
    loss_fn    = LabelSmoothingLoss(tgt_vocab_size, pad_idx=1, smoothing=0.1)

    grad_norms = []

    for epoch in range(cfg["num_epochs"]):
        ablation_model.train()
        for step, (src_b, tgt_b) in enumerate(train_loader):
            src_b = src_b.to(device)
            tgt_b = tgt_b.to(device)
            dec_in  = tgt_b[:, :-1]
            dec_tgt = tgt_b[:, 1:]
            sm      = make_src_mask(src_b)
            tm      = make_tgt_mask(dec_in)

            logits = ablation_model(src_b, dec_in, sm, tm)
            loss   = loss_fn(
                logits.view(-1, tgt_vocab_size),
                dec_tgt.contiguous().view(-1),
            )
            optimizer.zero_grad()
            loss.backward()

            # Log gradient norms of Q and K projections in first 1000 steps
            if epoch == 0 and step < 1000:
                total_norm = 0.0
                for name, p in ablation_model.named_parameters():
                    if p.grad is not None and ("proj_q" in name or "proj_k" in name):
                        total_norm += p.grad.data.norm(2).item() ** 2
                grad_norms.append(total_norm ** 0.5)
                wandb.log({"ablation_no_scale/qk_grad_norm": total_norm ** 0.5,
                           "global_step": epoch * len(train_loader) + step})

            torch.nn.utils.clip_grad_norm_(ablation_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    # Restore original function
    model_module.scaled_dot_product_attention = _original_sdpa


if __name__ == "__main__":
    run_training_experiment()