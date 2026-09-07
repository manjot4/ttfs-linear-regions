from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .configs import RegionInitConfig, TrainingConfig, WidthDepthConfig
from .data import calibration_batch, dataset_info, load_region_dataset, load_training_oriented_datasets, sample_pairs
from .initializations import (
    initialize_positive_region_model,
    initialize_relu_classifier,
    initialize_relu_region_model,
    initialize_shared_region_model,
    initialize_signed_region_model,
    initialize_ttfs_classifier_init1,
    initialize_ttfs_classifier_init3,
)
from .models import PositiveTTFSClassifier, PositiveTTFSNet, ReLUClassifier, ReLUNet, SharedTTFSNet, SignedTTFSNet
from .trajectories import count_regions_classifier, evaluate_pairs_layerwise
from .utils import cpu_generator, ensure_dir, seed_all


def _aggregate_width_depth(width_df: pd.DataFrame, depth_df: pd.DataFrame):
    group_width = [c for c in ["model_family", "init_name", "seed", "width"] if c in width_df.columns]
    width_seed = width_df.groupby(group_width, as_index=False).agg(
        regions=("regions", "mean"), ceiling=("ceiling_fraction", "mean")
    )
    summary_groups = [c for c in ["model_family", "init_name", "width"] if c in width_seed.columns]
    width_summary = width_seed.groupby(summary_groups, as_index=False).agg(
        mean_regions=("regions", "mean"),
        std_regions=("regions", "std"),
        mean_ceiling=("ceiling", "mean"),
        n_seeds=("seed", "nunique"),
    )
    width_summary["std_regions"] = width_summary["std_regions"].fillna(0.0)

    group_depth = [c for c in ["model_family", "init_name", "seed", "schedule", "depth"] if c in depth_df.columns]
    depth_seed = depth_df.groupby(group_depth, as_index=False).agg(
        regions=("regions", "mean"),
        new_regions=("new_regions_this_layer", "mean"),
        ceiling=("ceiling_fraction", "mean"),
    )
    summary_depth_groups = [c for c in ["model_family", "init_name", "schedule", "depth"] if c in depth_seed.columns]
    depth_summary = depth_seed.groupby(summary_depth_groups, as_index=False).agg(
        mean_regions=("regions", "mean"),
        std_regions=("regions", "std"),
        mean_new_regions=("new_regions", "mean"),
        mean_ceiling=("ceiling", "mean"),
        n_seeds=("seed", "nunique"),
    )
    depth_summary["std_regions"] = depth_summary["std_regions"].fillna(0.0)
    return width_summary, depth_summary


def run_width_depth(
    dataset_name: str,
    model_family: str,
    data_root: str | Path,
    results_root: str | Path,
    init_name: str = "init1",
    config: WidthDepthConfig | None = None,
    init_config: RegionInitConfig | None = None,
    device: str | torch.device = "cpu",
    download: bool = True,
) -> dict[str, pd.DataFrame | Path]:
    """Run the initialization-time width/depth region experiment.

    ``model_family`` is one of ``ttfs``, ``relu``, ``shared``, ``signed``.
    ``init_name`` is relevant for ``ttfs`` and can be ``init1`` or ``init2``.
    """
    config = config or WidthDepthConfig()
    init_config = init_config or RegionInitConfig()
    device = torch.device(device)
    ds = load_region_dataset(dataset_name, data_root, train=True, download=download)
    info = dataset_info(dataset_name)
    pairs = sample_pairs(ds, config.num_pairs, config.probe_seed)
    cal_x = calibration_batch(
        ds, config.calibration_size, config.probe_seed + 1, device=device, flatten=False
    )
    results_root = ensure_dir(results_root)

    aliases = {"positive": "ttfs", "snn": "ttfs", "shared_ttfs": "shared", "signed_ttfs": "signed"}
    model_family = aliases.get(model_family.lower(), model_family.lower())
    if model_family not in {"ttfs", "relu", "shared", "signed"}:
        raise ValueError(model_family)

    if model_family == "ttfs":
        init_alias = {"init1": "generic_positive", "init2": "max_regions"}
        canonical_init = init_alias.get(init_name.lower(), init_name)
        batch_size = config.ttfs_batch_size
    elif model_family == "relu":
        canonical_init = "he_nonzero_bias"
        batch_size = config.relu_batch_size
    elif model_family == "shared":
        canonical_init = "shared_generic_positive"
        batch_size = config.shared_batch_size
    else:
        canonical_init = "generic_signed_gaussian"
        batch_size = config.signed_batch_size

    width_runs, depth_runs, init_rows = [], [], []
    for seed in config.seeds:
        for width in config.widths:
            seed_all(seed)
            if model_family == "ttfs":
                model = PositiveTTFSNet(
                    info.raw_dim, width, [width], use_norm=(canonical_init == "max_regions")
                ).to(device)
                rr = initialize_positive_region_model(model, canonical_init, cal_x, seed, init_config, config.delays_on)
            elif model_family == "relu":
                model = ReLUNet(info.raw_dim, width, [width]).to(device)
                rr = initialize_relu_region_model(model, seed)
            elif model_family == "shared":
                model = SharedTTFSNet(info.raw_dim, width, [width]).to(device)
                rr = initialize_shared_region_model(model, seed, init_config, config.delays_on)
            else:
                model = SignedTTFSNet(
                    info.raw_dim, width, [width], denom_eps=init_config.denom_eps, last_spike=init_config.signed_last_spike
                ).to(device)
                rr = initialize_signed_region_model(model, seed, init_config, config.delays_on)
            for z in rr:
                z.update(experiment="width", seed=seed, width=width, schedule="")
            init_rows.extend(rr)
            df = evaluate_pairs_layerwise(model, ds, pairs, config.steps, batch_size, device)
            df["model_family"] = model_family
            df["init_name"] = canonical_init
            df["seed"] = seed
            df["width"] = width
            df["schedule"] = ""
            width_runs.append(df)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for schedule, widths in config.schedules.items():
            seed_all(seed)
            widths = list(widths)
            if model_family == "ttfs":
                model = PositiveTTFSNet(
                    info.raw_dim, widths[0], widths, use_norm=(canonical_init == "max_regions")
                ).to(device)
                rr = initialize_positive_region_model(model, canonical_init, cal_x, seed, init_config, config.delays_on)
            elif model_family == "relu":
                model = ReLUNet(info.raw_dim, widths[0], widths).to(device)
                rr = initialize_relu_region_model(model, seed)
            elif model_family == "shared":
                model = SharedTTFSNet(info.raw_dim, widths[0], widths).to(device)
                rr = initialize_shared_region_model(model, seed, init_config, config.delays_on)
            else:
                model = SignedTTFSNet(
                    info.raw_dim, widths[0], widths, denom_eps=init_config.denom_eps, last_spike=init_config.signed_last_spike
                ).to(device)
                rr = initialize_signed_region_model(model, seed, init_config, config.delays_on)
            for z in rr:
                z.update(experiment="depth_schedule", seed=seed, width=np.nan, schedule=schedule)
            init_rows.extend(rr)
            df = evaluate_pairs_layerwise(model, ds, pairs, config.steps, batch_size, device)
            df["model_family"] = model_family
            df["init_name"] = canonical_init
            df["seed"] = seed
            df["width"] = np.nan
            df["schedule"] = schedule
            depth_runs.append(df)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    width_df = pd.concat(width_runs, ignore_index=True)
    depth_df = pd.concat(depth_runs, ignore_index=True)
    initialization_df = pd.DataFrame(init_rows)
    width_summary, depth_summary = _aggregate_width_depth(width_df, depth_df)

    prefix = f"{dataset_info(dataset_name).name}_{model_family}_{canonical_init}"
    paths = {
        "width_raw": results_root / f"{prefix}_width_raw.csv",
        "depth_raw": results_root / f"{prefix}_depth_raw.csv",
        "width_summary": results_root / f"{prefix}_width_summary.csv",
        "depth_summary": results_root / f"{prefix}_depth_summary.csv",
        "initialization": results_root / f"{prefix}_initialization.csv",
    }
    width_df.to_csv(paths["width_raw"], index=False)
    depth_df.to_csv(paths["depth_raw"], index=False)
    width_summary.to_csv(paths["width_summary"], index=False)
    depth_summary.to_csv(paths["depth_summary"], index=False)
    initialization_df.to_csv(paths["initialization"], index=False)
    return {
        "width_raw": width_df,
        "depth_raw": depth_df,
        "width_summary": width_summary,
        "depth_summary": depth_summary,
        "initialization": initialization_df,
        "paths": paths,
    }


def _make_loader(dataset, batch_size: int, seed: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=cpu_generator(seed) if shuffle else None,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _train_epoch(model, loader, optimizer, device, grad_clip: float, label_smoothing: float = 0.0, project_snn: bool = False) -> float:
    model.train()
    total_loss = 0.0
    total_n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, label_smoothing=label_smoothing)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if project_snn:
            model.project_constraints_()
        total_loss += float(loss.item()) * xb.size(0)
        total_n += xb.size(0)
    return total_loss / max(1, total_n)


@torch.no_grad()
def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += yb.numel()
    return correct / max(1, total)


def run_depth1_training(
    dataset_name: str,
    data_root: str | Path,
    results_root: str | Path,
    config: TrainingConfig | None = None,
    init_config: RegionInitConfig | None = None,
    device: str | torch.device = "cpu",
    download: bool = True,
    widths: tuple[int, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
) -> dict[str, pd.DataFrame | Path]:
    """Depth-one training experiment used in the appendix.

    Region complexity is measured for Init 1 TTFS and matched ReLU. The Init 3
    arm is accuracy-only, matching the attached training notebooks.
    """
    config = config or TrainingConfig()
    init_config = init_config or RegionInitConfig()
    device = torch.device(device)
    info = dataset_info(dataset_name)
    widths = tuple(widths or config.widths)
    seeds = tuple(seeds or config.seeds)
    results_root = ensure_dir(results_root)

    train_full = load_region_dataset(dataset_name, data_root, train=True, download=download)
    test_full = load_region_dataset(dataset_name, data_root, train=False, download=download)
    pairs = sample_pairs(train_full, config.region_pairs, config.probe_seed)

    region_rows, accuracy_rows, loss_rows = [], [], []
    for width in widths:
        for seed in seeds:
            seed_all(seed)
            ttfs = PositiveTTFSClassifier(
                info.raw_dim,
                width,
                num_classes=info.num_classes,
                depth=1,
                delays_on=config.delays_on,
                train_delays=False,
                weight_train_bounds=(config.weight_train_min, config.weight_train_max),
                theta_train_bounds=(config.theta_train_min, config.theta_train_max),
                delay_train_bounds=(config.delay_train_min, config.delay_train_max),
            ).to(device)
            relu = ReLUClassifier(info.raw_dim, width, info.num_classes, depth=1).to(device)
            initialize_ttfs_classifier_init1(ttfs, seed, init_config)
            initialize_relu_classifier(relu, seed)
            ttfs_opt = torch.optim.AdamW(
                ttfs.parameters(), lr=config.learning_rate,
                weight_decay=config.weight_decay_mnist if "mnist" in info.name else config.weight_decay_cifar10,
            )
            relu_opt = torch.optim.AdamW(
                relu.parameters(), lr=config.learning_rate,
                weight_decay=config.weight_decay_mnist if "mnist" in info.name else config.weight_decay_cifar10,
            )
            train_loader_ttfs = _make_loader(train_full, config.train_batch_size, seed + 100)
            train_loader_relu = _make_loader(train_full, config.train_batch_size, seed + 100)
            test_loader = _make_loader(test_full, 256, seed, shuffle=False)

            for epoch in range(config.epochs + 1):
                if epoch in config.checkpoints:
                    for model_name, model, bs in [
                        ("ttfs_positive", ttfs, config.region_batch_size_snn),
                        ("relu", relu, config.region_batch_size_relu),
                    ]:
                        for pair_id, (i, j) in enumerate(pairs):
                            x0, _ = train_full[i]
                            x1, _ = train_full[j]
                            regions, ceiling = count_regions_classifier(
                                model, x0.float(), x1.float(), config.region_steps, bs, device
                            )
                            region_rows.append({
                                "dataset": info.name,
                                "model_family": model_name,
                                "init_name": "generic_positive" if model_name.startswith("ttfs") else "he_nonzero_bias",
                                "seed": seed,
                                "width": width,
                                "depth": 1,
                                "epoch": epoch,
                                "pair_id": pair_id,
                                "regions": regions,
                                "ceiling_fraction": ceiling,
                            })
                    accuracy_rows.extend([
                        {"dataset": info.name, "model_family": "ttfs_positive", "init_name": "generic_positive", "seed": seed, "width": width, "depth": 1, "epoch": epoch, "test_accuracy": _accuracy(ttfs, test_loader, device)},
                        {"dataset": info.name, "model_family": "relu", "init_name": "he_nonzero_bias", "seed": seed, "width": width, "depth": 1, "epoch": epoch, "test_accuracy": _accuracy(relu, test_loader, device)},
                    ])
                if epoch == config.epochs:
                    break
                loss_ttfs = _train_epoch(ttfs, train_loader_ttfs, ttfs_opt, device, config.grad_clip, project_snn=True)
                loss_relu = _train_epoch(relu, train_loader_relu, relu_opt, device, config.grad_clip, project_snn=False)
                loss_rows.extend([
                    {"dataset": info.name, "model_family": "ttfs_positive", "init_name": "generic_positive", "seed": seed, "width": width, "depth": 1, "epoch": epoch + 1, "train_loss": loss_ttfs},
                    {"dataset": info.name, "model_family": "relu", "init_name": "he_nonzero_bias", "seed": seed, "width": width, "depth": 1, "epoch": epoch + 1, "train_loss": loss_relu},
                ])

    # Accuracy-only Init 3 arm.
    if config.run_training_oriented_accuracy:
        to_train, to_calib, to_test = load_training_oriented_datasets(dataset_name, data_root, download=download)
        label_smoothing = 0.0 if "mnist" in info.name else 0.05
        for width in widths:
            for seed in seeds:
                seed_all(seed)
                model = PositiveTTFSClassifier(
                    info.raw_dim, width, info.num_classes, depth=1,
                    delays_on=False, train_delays=False,
                    weight_train_bounds=(config.weight_train_min, config.weight_train_max),
                    theta_train_bounds=(config.theta_train_min, config.theta_train_max),
                ).to(device)
                calib_x = calibration_batch(
                    to_calib, config.to_encoder_calibration_samples, seed + 200, device=device, flatten=True
                )
                initialize_ttfs_classifier_init3(model, calib_x, seed, config, init_config)
                optimizer = torch.optim.AdamW(
                    model.parameters(), lr=config.to_learning_rate, weight_decay=config.to_weight_decay
                )
                train_loader = _make_loader(to_train, config.train_batch_size, seed + 100)
                test_loader = _make_loader(to_test, 256, seed, shuffle=False)
                for epoch in range(config.epochs + 1):
                    if epoch in config.checkpoints:
                        accuracy_rows.append({
                            "dataset": info.name,
                            "model_family": "ttfs_positive",
                            "init_name": "training_oriented",
                            "seed": seed,
                            "width": width,
                            "depth": 1,
                            "epoch": epoch,
                            "test_accuracy": _accuracy(model, test_loader, device),
                        })
                    if epoch == config.epochs:
                        break
                    loss = _train_epoch(
                        model, train_loader, optimizer, device, config.to_grad_clip,
                        label_smoothing=label_smoothing, project_snn=True
                    )
                    loss_rows.append({
                        "dataset": info.name,
                        "model_family": "ttfs_positive",
                        "init_name": "training_oriented",
                        "seed": seed,
                        "width": width,
                        "depth": 1,
                        "epoch": epoch + 1,
                        "train_loss": loss,
                    })

    region_df = pd.DataFrame(region_rows)
    accuracy_df = pd.DataFrame(accuracy_rows)
    loss_df = pd.DataFrame(loss_rows)
    prefix = info.name
    paths = {
        "regions": results_root / f"{prefix}_depth1_training_regions.csv",
        "accuracy": results_root / f"{prefix}_depth1_training_accuracy.csv",
        "loss": results_root / f"{prefix}_depth1_training_loss.csv",
    }
    region_df.to_csv(paths["regions"], index=False)
    accuracy_df.to_csv(paths["accuracy"], index=False)
    loss_df.to_csv(paths["loss"], index=False)
    return {"regions": region_df, "accuracy": accuracy_df, "loss": loss_df, "paths": paths}
