"""Scoring functions for probabilistic 3-way (H/D/A) forecasts."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss


def one_hot(y: np.ndarray, n_classes: int = 3) -> np.ndarray:
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def ranked_probability_score(probs: np.ndarray, y: np.ndarray) -> float:
    """RPS for ordered outcomes (H < D < A). Lower is better; 0 is perfect.

    The standard scoring rule in the football-forecasting literature because
    it rewards putting probability *near* the right outcome (a draw forecast
    is 'closer' to a home win than an away-win forecast is).
    """
    obs = one_hot(y, probs.shape[1])
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(obs, axis=1)
    return float(np.mean(np.sum((cum_p - cum_o) ** 2, axis=1) / (probs.shape[1] - 1)))


def multiclass_brier(probs: np.ndarray, y: np.ndarray) -> float:
    obs = one_hot(y, probs.shape[1])
    return float(np.mean(np.sum((probs - obs) ** 2, axis=1)))


def evaluate_probs(probs: np.ndarray, y: np.ndarray) -> dict[str, float]:
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1, 2])),
        "brier": multiclass_brier(probs, y),
        "rps": ranked_probability_score(probs, y),
    }
