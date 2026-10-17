"""CPU models and NumPy feature builders for the observation-shift study."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

try:  # Feature builders remain usable in the NumPy-only runtime.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in the supplied NumPy-only runtime
    torch = None
    nn = None


def _validate_observations(x: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Check shapes and return a float32 value array and binary mask."""
    x = np.asarray(x, dtype=np.float32)
    m = np.asarray(m)
    if x.ndim != 2 or m.shape != x.shape:
        raise ValueError("x and m must both have shape [n, T]")
    if x.shape[1] == 0:
        raise ValueError("time dimension T must be positive")
    if not np.all((m == 0) | (m == 1)):
        raise ValueError("m must contain only 0/1 values")
    m = m.astype(np.float32, copy=False)
    if np.any((m == 1) & ~np.isfinite(x)):
        raise ValueError("observed x values must be finite")
    return x, m


def build_locf_delta(x: np.ndarray, m: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build forward-filled values and GRU-D elapsed-time features.

    Before the first observation, LOCF is zero. Delta at t=0 is initialized to
    zero; afterwards it is the number of steps since the most recent
    observation strictly before t (or since t=0 if none has occurred), divided
    by T. This is the standard GRU-D recurrence convention.
    """
    x, m = _validate_observations(x, m)
    n, t_len = x.shape
    times = np.broadcast_to(np.arange(t_len, dtype=np.int32), (n, t_len))
    x_obs = np.where(m > 0, x, 0.0)

    last_at_or_before = np.maximum.accumulate(
        np.where(m > 0, times, -1), axis=1
    )
    valid_last = last_at_or_before >= 0
    gathered = np.take_along_axis(x_obs, np.maximum(last_at_or_before, 0), axis=1)
    x_locf = np.where(valid_last, gathered, 0.0).astype(np.float32, copy=False)

    # Shift the inclusive last-observation index right to obtain the last
    # observation strictly before each time point.
    previous = np.full((n, t_len), -1, dtype=np.int32)
    if t_len > 1:
        previous[:, 1:] = last_at_or_before[:, :-1]
    elapsed = np.where(previous >= 0, times - previous, times)
    delta = (elapsed.astype(np.float32) / float(t_len)).astype(np.float32, copy=False)
    return x_locf, delta


def build_recurrent_features(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Return [x_locf, m, delta] in a float32 [n, T, 3] array for the GRU."""
    x, m = _validate_observations(x, m)
    x_locf, delta = build_locf_delta(x, m)
    return np.stack((x_locf, m, delta), axis=-1).astype(np.float32, copy=False)


def build_cde_path(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Build the CDE control path [t/T, interpolated x, cumulative count/T].

    Interpolation and endpoint extension are performed across the batch with
    array operations; no Python loop over patients is needed. A never-observed
    row has an all-zero value channel.
    """
    x, m = _validate_observations(x, m)
    n, t_len = x.shape
    times = np.broadcast_to(np.arange(t_len, dtype=np.int32), (n, t_len))
    x_obs = np.where(m > 0, x, 0.0)

    left = np.maximum.accumulate(np.where(m > 0, times, -1), axis=1)
    right = np.minimum.accumulate(
        np.where(m > 0, times, t_len)[:, ::-1], axis=1
    )[:, ::-1]
    left_values = np.take_along_axis(x_obs, np.maximum(left, 0), axis=1)
    right_values = np.take_along_axis(x_obs, np.minimum(right, t_len - 1), axis=1)

    has_left = left >= 0
    has_right = right < t_len
    both = has_left & has_right
    denominator = np.maximum(right - left, 1)
    fraction = (times - left).astype(np.float32) / denominator.astype(np.float32)
    between = left_values + fraction * (right_values - left_values)
    interpolated = np.where(
        both,
        between,
        np.where(has_left, left_values, np.where(has_right, right_values, 0.0)),
    ).astype(np.float32, copy=False)

    time_channel = np.broadcast_to(
        np.arange(t_len, dtype=np.float32) / float(t_len), (n, t_len)
    )
    count_channel = np.cumsum(m, axis=1, dtype=np.float32) / float(t_len)
    return np.stack((time_channel, interpolated, count_channel), axis=-1).astype(
        np.float32, copy=False
    )


if torch is not None:

    class GRUModel(nn.Module):
        """GRU over LOCF values, observation masks, and elapsed times."""

        def __init__(self, hidden_size: int = 64) -> None:
            super().__init__()
            self.rnn = nn.GRU(input_size=3, hidden_size=hidden_size, batch_first=True)
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, features: torch.Tensor) -> torch.Tensor:
            _, h_last = self.rnn(features)
            return self.head(h_last[-1]).squeeze(-1)


    class GRUDModel(nn.Module):
        """GRU-D with learned input/hidden decays and a GRUCell time loop."""

        def __init__(self, empirical_mean: float = 0.0, hidden_size: int = 64) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.register_buffer("empirical_mean", torch.tensor(float(empirical_mean)))
            self.gamma_x = nn.Linear(1, 1)
            self.gamma_h = nn.Linear(1, hidden_size)
            self.cell = nn.GRUCell(input_size=2, hidden_size=hidden_size)
            self.head = nn.Linear(hidden_size, 1)

        def forward(
            self, x: torch.Tensor, m: torch.Tensor, delta: torch.Tensor
        ) -> torch.Tensor:
            batch, t_len = x.shape
            h = x.new_zeros((batch, self.hidden_size))
            x_last = self.empirical_mean.expand(batch)
            mean = self.empirical_mean
            for t in range(t_len):
                mask_t = m[:, t]
                delta_t = delta[:, t:t + 1]
                gamma_x = torch.exp(-torch.relu(self.gamma_x(delta_t))).squeeze(-1)
                gamma_h = torch.exp(-torch.relu(self.gamma_h(delta_t)))
                x_decayed = gamma_x * x_last + (1.0 - gamma_x) * mean
                x_t = torch.where(mask_t > 0.5, x[:, t], x_decayed)
                h = self.cell(torch.stack((x_t, mask_t), dim=-1), gamma_h * h)
                x_last = torch.where(mask_t > 0.5, x[:, t], x_last)
            return self.head(h).squeeze(-1)


    class CDEModel(nn.Module):
        """Neural CDE integrated by two explicit Euler substeps per interval."""

        def __init__(self, hidden_size: int = 64, channels: int = 3) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.channels = channels
            self.initial = nn.Linear(channels, hidden_size)
            self.vector_field = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.GELU(),
                nn.Linear(128, 128),
                nn.GELU(),
                nn.Linear(128, hidden_size * channels),
                nn.Tanh(),
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, path: torch.Tensor) -> torch.Tensor:
            h = self.initial(path[:, 0, :])
            increments = path[:, 1:, :] - path[:, :-1, :]
            for t in range(increments.shape[1]):
                d_x = increments[:, t, :] * 0.5
                for _ in range(2):
                    vector = self.vector_field(h).reshape(
                        h.shape[0], self.hidden_size, self.channels
                    )
                    h = h + torch.bmm(vector, d_x.unsqueeze(-1)).squeeze(-1)
            return self.head(h).squeeze(-1)


    class TransformerModel(nn.Module):
        """Transformer with observed-time tokens and an always-valid CLS token."""

        def __init__(self, hidden_size: int = 64, max_length: int = 48) -> None:
            super().__init__()
            self.hidden_size = hidden_size
            self.input_projection = nn.Linear(1, hidden_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            self.register_buffer("time_embedding", self._sinusoidal(max_length, hidden_size))
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=4,
                dim_feedforward=hidden_size * 4,
                batch_first=True,
                dropout=0.0,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)
            self.head = nn.Linear(hidden_size, 1)

        @staticmethod
        def _sinusoidal(length: int, dim: int) -> torch.Tensor:
            positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
            scale = torch.exp(
                torch.arange(0, dim, 2, dtype=torch.float32)
                * (-np.log(10000.0) / dim)
            )
            embedding = torch.zeros(length, dim, dtype=torch.float32)
            embedding[:, 0::2] = torch.sin(positions * scale)
            embedding[:, 1::2] = torch.cos(positions * scale[: embedding[:, 1::2].shape[1]])
            return embedding

        def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
            t_len = x.shape[1]
            if t_len > self.time_embedding.shape[0]:
                raise ValueError("input sequence exceeds configured max_length")
            tokens = self.input_projection(x.unsqueeze(-1))
            tokens = tokens + self.time_embedding[:t_len].unsqueeze(0)
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            sequence = torch.cat((cls, tokens), dim=1)
            padding = torch.cat(
                (torch.zeros_like(m[:, :1], dtype=torch.bool), m < 0.5), dim=1
            )
            encoded = self.encoder(sequence, src_key_padding_mask=padding)
            return self.head(encoded[:, 0, :]).squeeze(-1)


    def make_model(
        arch: str, empirical_mean: float = 0.0, hidden_size: int = 64, max_length: int = 48
    ) -> nn.Module:
        """Construct one of the four specified architectures."""
        if arch == "gru":
            return GRUModel(hidden_size=hidden_size)
        if arch == "grud":
            return GRUDModel(empirical_mean=empirical_mean, hidden_size=hidden_size)
        if arch == "cde":
            return CDEModel(hidden_size=hidden_size)
        if arch == "transformer":
            return TransformerModel(hidden_size=hidden_size, max_length=max_length)
        raise ValueError(f"unknown architecture: {arch}")

else:

    class _TorchUnavailable:
        def __init__(self, *args, **kwargs) -> None:
            raise ImportError("PyTorch is required to construct model classes")

    GRUModel = GRUDModel = CDEModel = TransformerModel = _TorchUnavailable

    def make_model(*args, **kwargs):
        raise ImportError("PyTorch is required to construct model classes")

