"""
lr_scheduler.py — Noam Learning Rate Scheduler
DA6401 Assignment 3: "Attention Is All You Need"

Reference: Vaswani et al. 2017, §5.3
Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))

Intuition:
    - During warm-up (step < warmup_steps), LR rises linearly.
      This prevents the self-attention Q/K projections from receiving
      large gradient updates before they have been initialised to
      meaningful directions — reducing early-step divergence.
    - After warm-up, LR decays as step^(-0.5), giving a smooth
      annealing schedule that mirrors inverse square root decay.
"""

import math
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):
    """
    Noam learning-rate scheduler as described in "Attention Is All You Need".

    Applies linear warm-up followed by inverse-square-root decay.

    The base learning rate stored in the optimiser param groups acts as
    a global scaling constant (set it to 1.0 for pure Noam control, or
    to another value if you want an extra scaling factor).

    Args:
        optimizer    (Optimizer): Wrapped optimiser.
        d_model      (int)      : Model dimensionality.
        warmup_steps (int)      : Steps over which LR increases linearly.
        last_epoch   (int)      : Last epoch index (default -1 → start fresh).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        # Store hyperparameters BEFORE calling super().__init__ because
        # the parent immediately calls get_lr() which needs them.
        self.d_model      = d_model
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch=last_epoch)

    # ------------------------------------------------------------------

    def _get_lr_scale(self) -> float:
        """
        Compute the Noam scaling factor for the current step.

        step = self.last_epoch + 1  (avoids division-by-zero at step 0)

        scale = d_model^{-0.5} * min(step^{-0.5},
                                      step * warmup_steps^{-1.5})
        """
        # last_epoch is 0-indexed; convert to 1-indexed step
        step = self.last_epoch + 1

        # Two regimes unified in a single min expression
        warmup_term  = step * (self.warmup_steps ** -1.5)   # linear ramp
        decay_term   = step ** -0.5                          # inverse sqrt decay
        noam_scale   = (self.d_model ** -0.5) * min(decay_term, warmup_term)
        return noam_scale

    # ------------------------------------------------------------------

    def get_lr(self) -> list:
        """
        Compute the learning rate for every param group.

        Multiplies each group's *base* LR (set at optimiser construction)
        by the current Noam scale factor.

        Returns:
            list[float]: One learning rate per param group.
        """
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


# ──────────────────────────────────────────────────────────────────────
#  Helper — do NOT modify
# ──────────────────────────────────────────────────────────────────────

def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list:
    """
    Simulate the LR trajectory of NoamScheduler for `total_steps` steps.

    Returns:
        list[float]: LR value at each step (length == total_steps).
    """
    dummy_model = torch.nn.Linear(1, 1)
    optimizer   = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler   = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history


# ──────────────────────────────────────────────────────────────────────
#  Quick visual check — run:  python lr_scheduler.py
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    peak_step = lrs.index(max(lrs))
    print(f"Peak LR = {max(lrs):.6f} at step {peak_step}")
    print(f"Theoretical peak step ≈ {WARMUP_STEPS}")

    plt.figure(figsize=(9, 4))
    plt.plot(lrs, linewidth=1.5, label="Noam LR")
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.show()
