from __future__ import annotations

import hashlib
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .configs import Exact2DConfig
from .data import dataset_info, load_region_dataset
from .exact2d import (
    enumerate_relu,
    enumerate_ttfs,
    exact_decision_boundary_pieces,
    make_slice_from_three,
    polygon_from_constraints,
    validate_against_grid,
)
from .initializations import initialize_exact_relu, initialize_exact_ttfs
from .models import PositiveTTFSClassifier, ReLUClassifier
from .utils import cpu_generator, ensure_dir, seed_all


def choose_reference_samples(dataset, seed: int = 542, classes=None):
    rng = np.random.default_rng(seed)
    if classes is None:
        labels = sorted(set(int(dataset[i][1]) for i in range(min(len(dataset), 20_000))))
        classes = tuple(int(c) for c in rng.choice(labels, size=3, replace=False))
    else:
        classes = tuple(int(c) for c in classes)
        if len(classes) != 3 or len(set(classes)) != 3:
            raise ValueError("classes must contain three distinct labels")
    by_class = {c: [] for c in classes}
    for i in range(len(dataset)):
        y = int(dataset[i][1])
        if y in by_class:
            by_class[y].append(i)
    refs = []
    for c in classes:
        if not by_class[c]:
            raise RuntimeError(f"No sample found for class {c}")
        idx = int(rng.choice(by_class[c]))
        x, y = dataset[idx]
        refs.append((x.view(-1).float(), int(y), idx))
    return refs


def _make_loader(ds, batch_size: int, seed: int, shuffle: bool):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=cpu_generator(seed) if shuffle else None,
        num_workers=0,
    )


def _make_optimizer(model, is_ttfs: bool, lr: float, weight_decay: float, delay_lr: float):
    if is_ttfs:
        ordinary, delays = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (delays if "input_delay" in name else ordinary).append(p)
        groups = [{"params": ordinary, "lr": lr}]
        if delays:
            groups.append({"params": delays, "lr": delay_lr})
    else:
        groups = [{"params": [p for p in model.parameters() if p.requires_grad], "lr": lr}]
    return torch.optim.Adam(groups, weight_decay=weight_decay)


def _train_epoch(model, loader, optimizer, is_ttfs: bool, device):
    model.train()
    total_loss = total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        optimizer.step()
        if is_ttfs:
            model.project_constraints_()
        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)
    return total_loss / max(total_n, 1)


@torch.no_grad()
def _accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += int((model(xb).argmax(1) == yb).sum().item())
        total += int(yb.numel())
    return correct / max(total, 1)


def _signature_color(signature):
    payload = repr(signature).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=3).digest()
    rgb = np.frombuffer(digest, dtype=np.uint8).astype(float) / 255.0
    return 0.25 + 0.75 * rgb


def plot_exact_geometry(cells, pieces, sl, output=None, title=None):
    fig, axes = plt.subplots(2, 1, figsize=(5.0, 8.0))
    for cell in cells:
        poly = polygon_from_constraints(cell["A"], cell["b"], sl.bounds)
        if len(poly) < 3:
            continue
        axes[0].fill(poly[:, 0], poly[:, 1], color=_signature_color(cell["signature"]), linewidth=0)
    for piece in pieces:
        p0, p1 = piece["p0"], piece["p1"]
        axes[1].plot([p0[0], p1[0]], [p0[1], p1[1]], color="black", linewidth=1.0)
    for ax in axes:
        ax.set_xlim(sl.bounds[0], sl.bounds[1])
        ax.set_ylim(sl.bounds[2], sl.bounds[3])
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title("Linear regions")
    axes[1].set_title("Decision boundary")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if output:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight")
    return fig


def run_exact2d(
    dataset_name: str,
    data_root: str | Path,
    results_root: str | Path,
    width: int = 10,
    depth: int = 3,
    seed: int = 42,
    epochs: int = 0,
    checkpoints=(0,),
    reference_classes=None,
    config: Exact2DConfig | None = None,
    device: str | torch.device = "cpu",
    download: bool = True,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    delay_lr: float = 1e-3,
    validate_grid: bool = True,
    make_plots: bool = True,
    verbose: bool = True,
):
    config = config or Exact2DConfig()
    device = torch.device(device)
    info = dataset_info(dataset_name)
    train_ds = load_region_dataset(dataset_name, data_root, train=True, download=download)
    test_ds = load_region_dataset(dataset_name, data_root, train=False, download=download)
    refs = choose_reference_samples(train_ds, seed + 500, reference_classes)
    sl = make_slice_from_three(refs[0][0], refs[1][0], refs[2][0], margin=config.margin)
    results_root = ensure_dir(results_root)

    ttfs = PositiveTTFSClassifier(
        info.raw_dim,
        width,
        info.num_classes,
        depth=depth,
        delays_on=config.delays_on,
        train_delays=config.train_delays,
        weight_train_bounds=(config.weight_train_min, config.weight_train_max),
        theta_train_bounds=(config.theta_train_min, config.theta_train_max),
        delay_train_bounds=(config.delay_train_min, config.delay_train_max),
    ).to(device)
    relu = ReLUClassifier(info.raw_dim, width, info.num_classes, depth=depth).to(device)
    seed_all(seed)
    initialize_exact_ttfs(ttfs, seed, config)
    initialize_exact_relu(relu, seed)

    train_loader_ttfs = _make_loader(train_ds, batch_size, seed + 300, True)
    train_loader_relu = _make_loader(train_ds, batch_size, seed + 300, True)
    test_loader = _make_loader(test_ds, 256, seed, False)
    opt_ttfs = _make_optimizer(ttfs, True, lr, weight_decay, delay_lr)
    opt_relu = _make_optimizer(relu, False, lr, weight_decay, delay_lr)

    checkpoints = tuple(sorted(set(int(x) for x in checkpoints) | {0, int(epochs)}))
    rows = []
    for epoch in range(epochs + 1):
        if epoch in checkpoints:
            for model_kind, model in [("ttfs", ttfs), ("relu", relu)]:
                start = time.time()
                try:
                    cells = (
                        enumerate_ttfs(model, sl, config, verbose=verbose)
                        if model_kind == "ttfs"
                        else enumerate_relu(model, sl, config, verbose=verbose)
                    )
                    pieces = exact_decision_boundary_pieces(
                        model, cells, model_kind, sl, info.num_classes, config
                    )
                    validation = None
                    if validate_grid:
                        validation = validate_against_grid(
                            model, cells, sl, model_kind, config.validation_grid_res, device
                        )
                    status = "complete"
                    exact_count = len(cells)
                    boundary_count = len(pieces)
                    if make_plots:
                        plot_exact_geometry(
                            cells,
                            pieces,
                            sl,
                            output=results_root / f"{info.name}_{model_kind}_w{width}_d{depth}_epoch{epoch}_regions_db.png",
                            title=f"{model_kind.upper()} epoch {epoch}",
                        )
                        plt.close("all")
                except Exception as exc:
                    status = f"failed:{type(exc).__name__}:{exc}"
                    exact_count = np.nan
                    boundary_count = np.nan
                    validation = None
                rows.append({
                    "dataset": info.name,
                    "model": model_kind,
                    "width": width,
                    "depth": depth,
                    "epoch": epoch,
                    "status": status,
                    "exact_region_count": exact_count,
                    "exact_decision_boundary_pieces": boundary_count,
                    "test_accuracy": _accuracy(model, test_loader, device),
                    "seconds": time.time() - start,
                    "delays_on": config.delays_on if model_kind == "ttfs" else np.nan,
                    "train_delays": config.train_delays if model_kind == "ttfs" else np.nan,
                    "grid_sampled_patterns": None if validation is None else validation["sampled_unique_patterns"],
                })
                if verbose:
                    print(rows[-1])
        if epoch == epochs:
            break
        _train_epoch(ttfs, train_loader_ttfs, opt_ttfs, True, device)
        _train_epoch(relu, train_loader_relu, opt_relu, False, device)

    df = pd.DataFrame(rows)
    out_csv = results_root / f"{info.name}_exact2d_w{width}_d{depth}.csv"
    df.to_csv(out_csv, index=False)
    return {"results": df, "slice": sl, "references": refs, "path": out_csv}
