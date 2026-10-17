"""Plain-assert checks for NumPy features and the PyTorch model/runner code."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

import models


def test_feature_builders() -> None:
    x = np.array(
        [[np.nan, 2.0, np.nan, np.nan, 6.0, np.nan],
         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]],
        dtype=np.float32,
    )
    m = np.array([[0, 1, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0]], dtype=np.int8)

    locf, delta = models.build_locf_delta(x, m)
    np.testing.assert_allclose(locf[0], [0, 2, 2, 2, 6, 6], atol=0, rtol=0)
    np.testing.assert_allclose(delta[0], np.array([0, 1, 1, 2, 3, 1]) / 6, atol=1e-7)
    np.testing.assert_allclose(locf[1], np.zeros(6), atol=0, rtol=0)
    np.testing.assert_allclose(delta[1], np.arange(6) / 6, atol=1e-7)

    recurrent = models.build_recurrent_features(x, m)
    assert recurrent.shape == (2, 6, 3)
    np.testing.assert_allclose(recurrent[0, :, 0], locf[0], atol=0, rtol=0)
    np.testing.assert_allclose(recurrent[0, :, 1], m[0], atol=0, rtol=0)
    np.testing.assert_allclose(recurrent[0, :, 2], delta[0], atol=0, rtol=0)

    path = models.build_cde_path(x, m)
    assert path.shape == (2, 6, 3)
    np.testing.assert_allclose(path[0, :, 0], np.arange(6) / 6, atol=1e-7)
    np.testing.assert_allclose(path[0, :, 1], [2, 2, 10 / 3, 14 / 3, 6, 6], atol=1e-6)
    np.testing.assert_allclose(path[0, :, 2], np.array([0, 1, 1, 1, 2, 2]) / 6, atol=1e-7)
    np.testing.assert_allclose(path[1, :, 1], np.zeros(6), atol=0, rtol=0)
    np.testing.assert_allclose(path[1, :, 2], np.zeros(6), atol=0, rtol=0)
    assert np.isfinite(recurrent).all() and np.isfinite(path).all()


def _torch_inputs(arch: str, x: np.ndarray, m: np.ndarray):
    import torch

    if arch == "gru":
        return (torch.from_numpy(models.build_recurrent_features(x, m)),)
    if arch == "grud":
        locf, delta = models.build_locf_delta(x, m)
        x_clean = np.where(m > 0, x, 0.0).astype(np.float32)
        return (torch.from_numpy(x_clean), torch.from_numpy(m.astype(np.float32)),
                torch.from_numpy(delta))
    if arch == "cde":
        return (torch.from_numpy(models.build_cde_path(x, m)),)
    x_clean = np.where(m > 0, x, 0.0).astype(np.float32)
    return (torch.from_numpy(x_clean), torch.from_numpy(m.astype(np.float32)))


def test_model_forward_and_one_step() -> None:
    import torch
    import torch.nn.functional as F

    torch.set_num_threads(1)
    rng = np.random.default_rng(23)
    n, t_len = 8, 8
    m = rng.binomial(1, 0.55, size=(n, t_len)).astype(np.float32)
    m[0, :] = 0.0  # explicitly exercise an empty history
    x = rng.normal(size=(n, t_len)).astype(np.float32)
    x[m == 0] = np.nan
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.float32)

    for index, arch in enumerate(("gru", "grud", "cde", "transformer")):
        torch.manual_seed(900 + index)
        features = _torch_inputs(arch, x, m)
        model = models.make_model(
            arch, empirical_mean=0.2, hidden_size=16, max_length=t_len
        )
        model.eval()
        with torch.no_grad():
            logits = model(*features)
        assert logits.shape == (n,), f"{arch} returned {tuple(logits.shape)}"
        assert torch.isfinite(logits).all(), f"{arch} produced non-finite logits"

        # One full-batch gradient step must lower the tiny overfit batch loss.
        initial_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        model.train()
        model.zero_grad(set_to_none=True)
        loss = F.binary_cross_entropy_with_logits(model(*features), labels)
        loss.backward()
        before = float(loss.detach())
        reduced = False
        for learning_rate in (1e-2, 1e-3, 1e-4, 1e-5):
            model.load_state_dict(initial_state)
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
            optimizer.zero_grad(set_to_none=True)
            trial_loss = F.binary_cross_entropy_with_logits(model(*features), labels)
            trial_loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                after = float(F.binary_cross_entropy_with_logits(model(*features), labels))
            if after < before:
                reduced = True
                break
            model.train()
        assert reduced, f"{arch} did not reduce tiny-batch loss after one update"


def test_run_grid_smoke() -> None:
    """Exercise the actual spawn pool, Bayes files, and all four model shards."""
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="models_smoke_", dir=root) as temporary:
        output = Path(temporary) / "results_smoke"
        command = [
            sys.executable,
            str(root / "run_grid.py"),
            "--exp", "all",
            "--workers", "1",
            "--out", str(output),
            "--smoke",
        ]
        subprocess.run(command, cwd=root, check=True, timeout=1800)
        expected = {
            "E1": ("sigma0.5_gamma1", ("A", "B_match", "B_dense")),
            "E2": ("sigma0.5", ("A", "B_sparse", "B_bursty")),
        }
        for exp, (config, policy_names) in expected.items():
            bayes_path = output / exp / config / "seed0" / "bayes.npz"
            assert bayes_path.exists(), f"missing {bayes_path}"
            with np.load(bayes_path) as data:
                assert data["y"].shape == (1000,)
                for rule in ("A_rule", "V_rule"):
                    for name in policy_names:
                        values = data[f"{rule}_{name}"]
                        assert values.shape == (1000,) and np.isfinite(values).all()
            for arch in ("gru", "grud", "cde", "transformer"):
                shard = output / exp / config / arch / "seed0"
                assert (shard / "preds.npz").exists()
                with np.load(shard / "preds.npz") as data:
                    assert data["y"].shape == (1000,)
                    for name in policy_names:
                        values = data[name]
                        assert values.shape == (1000,)
                        assert np.isfinite(values).all()
                        assert ((values >= 0) & (values <= 1)).all()
                metadata = json.loads((shard / "meta.json").read_text(encoding="utf-8"))
                assert metadata["best_epoch"] in range(1, 6)
                assert metadata["epochs_run"] in range(1, 6)


def main() -> None:
    test_feature_builders()
    print("PASS NumPy feature builders")
    if models.torch is None:
        print("SKIP PyTorch forward/training and run_grid smoke checks: torch is unavailable")
        return
    test_model_forward_and_one_step()
    print("PASS model forwards, empty histories, and one-step loss reductions")
    test_run_grid_smoke()
    print("PASS run_grid --smoke end to end")


if __name__ == "__main__":
    main()

