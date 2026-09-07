from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .configs import RegionInitConfig, TrainingConfig
from .models import (
    FrozenPerNeuronAffine,
    PositiveTTFSClassifier,
    PositiveTTFSNet,
    ReLUClassifier,
    ReLUNet,
    SharedTTFSNet,
    SignedTTFSNet,
)
from .utils import cpu_generator


@torch.no_grad()
def fill_nonzero_normal_(tensor: torch.Tensor, mean: float, std: float, generator: torch.Generator, eps: float = 1e-8) -> None:
    values = torch.randn(tuple(tensor.shape), generator=generator) * std + mean
    tiny = values.abs() < eps
    if tiny.any():
        signs = torch.where(values[tiny] >= 0, 1.0, -1.0)
        signs[signs == 0] = 1.0
        values[tiny] = signs * eps
    tensor.copy_(values.to(tensor.device))


@torch.no_grad()
def init_encoder_nonzero(enc: nn.Linear, seed: int) -> None:
    """Encoder initialization used in the TTFS width/depth notebooks."""
    fan_in, fan_out = enc.weight.shape[1], enc.weight.shape[0]
    sd = math.sqrt(2.0 / (fan_in + fan_out))
    W = torch.randn(tuple(enc.weight.shape), generator=cpu_generator(seed + 1)) * sd
    tiny = W.abs() < 1e-8
    W[tiny] = 1e-6
    enc.weight.copy_(W.to(enc.weight.device))
    mag = 0.005 + 0.045 * torch.rand(enc.bias.numel(), generator=cpu_generator(seed + 2))
    sign = torch.where(
        torch.rand(enc.bias.numel(), generator=cpu_generator(seed + 3)) > 0.5,
        1.0,
        -1.0,
    )
    enc.bias.copy_((mag * sign).to(enc.bias.device))


@torch.no_grad()
def he_initialize_linear(layer: nn.Linear, seed: int, bias_std: float = 0.01) -> None:
    fan_in = layer.weight.shape[1]
    W = torch.randn(tuple(layer.weight.shape), generator=cpu_generator(seed)) * math.sqrt(2.0 / fan_in)
    tiny = W.abs() < 1e-8
    W[tiny] = 1e-8
    layer.weight.copy_(W.to(layer.weight.device))
    fill_nonzero_normal_(layer.bias, 0.0, bias_std, cpu_generator(seed + 1))


@torch.no_grad()
def initialize_relu_region_model(model: ReLUNet, seed: int, bias_std: float = 0.01) -> list[dict]:
    he_initialize_linear(model.enc, seed + 1_000, bias_std=bias_std)
    rows = []
    all_layers = [(0, "encoder", model.enc)] + [
        (ell, "hidden", layer) for ell, layer in enumerate(model.layers, start=1)
    ]
    for ell, component, layer in all_layers:
        if component == "hidden":
            he_initialize_linear(layer, seed + 10_000 * ell, bias_std=bias_std)
        rows.append({
            "component": component,
            "layer": ell,
            "input_dim": layer.weight.shape[1],
            "output_dim": layer.weight.shape[0],
            "weight_std_target": math.sqrt(2.0 / layer.weight.shape[1]),
            "weight_mean_realized": float(layer.weight.mean().item()),
            "weight_std_realized": float(layer.weight.std().item()),
            "bias_mean_realized": float(layer.bias.mean().item()),
            "bias_std_realized": float(layer.bias.std().item()),
        })
    for name, p in model.named_parameters():
        if torch.any(p == 0):
            raise RuntimeError(f"Parameter {name} contains exact zero entries")
    return rows


@torch.no_grad()
def _init_positive_layer(layer, sigma: float, seed: int, cfg: RegionInitConfig, delays_on: bool) -> None:
    z = torch.randn(tuple(layer.w.shape), generator=cpu_generator(seed + 4))
    layer.w.copy_(torch.exp(sigma * z - 0.5 * sigma * sigma).to(layer.w.device))
    if delays_on:
        d = cfg.delay_min + (cfg.delay_max - cfg.delay_min) * torch.rand(
            layer.input_dim, generator=cpu_generator(seed + 5)
        )
        layer.input_delay.copy_(d.to(layer.input_delay.device))
    else:
        layer.input_delay.zero_()


@torch.no_grad()
def _generic_theta(layer, seed: int, cfg: RegionInitConfig) -> None:
    u = torch.rand(layer.output_dim, generator=cpu_generator(seed + 6))
    theta = torch.exp(
        torch.tensor(math.log(cfg.theta_min))
        + u * (math.log(cfg.theta_max) - math.log(cfg.theta_min))
    )
    layer.threshold.copy_(theta.to(layer.threshold.device))


def _stratified_k(nin: int, nout: int, cfg: RegionInitConfig) -> torch.Tensor:
    kmax = max(1, min(nin - 1, int(round(cfg.boundary_k_max_fraction * nin))))
    return torch.round(torch.linspace(1, kmax, steps=nout)).long().clamp(1, nin - 1)


@torch.no_grad()
def _boundary_theta(layer, h: torch.Tensor, seed: int, cfg: RegionInitConfig) -> None:
    B, N = h.shape
    O = layer.output_dim
    arrivals = h + layer.input_delay.view(1, -1)
    t_sorted, order = torch.sort(arrivals, dim=1)
    W = layer.w.unsqueeze(0).expand(B, -1, -1)
    W_sorted = torch.gather(W, 1, order.unsqueeze(-1).expand(B, N, O))
    cum_w = torch.cumsum(W_sorted, dim=1)
    cum_wt = torch.cumsum(W_sorted * t_sorted.unsqueeze(-1), dim=1)
    ks = _stratified_k(N, O, cfg).to(h.device)
    ik = (ks - 1).view(1, 1, O).expand(B, 1, O)
    sw = torch.gather(cum_w, 1, ik).squeeze(1)
    swt = torch.gather(cum_wt, 1, ik).squeeze(1)
    t_next = torch.gather(t_sorted, 1, ks.view(1, O).expand(B, O))
    theta_star = (t_next * sw - swt).clamp_min(1e-6)
    med = theta_star.median(dim=0).values.detach().cpu()
    eps = cfg.boundary_log_noise_std * torch.randn(O, generator=cpu_generator(seed + 7))
    layer.threshold.copy_((med * torch.exp(eps)).clamp_min(1e-5).to(layer.threshold.device))


def _norm_targets(h: torch.Tensor, nout: int):
    if h.shape[1] == nout:
        return h.mean(0), h.std(0).clamp_min(1e-4)
    mu, sd = float(h.mean()), float(h.std().clamp_min(1e-4))
    return (
        torch.full((nout,), mu, device=h.device),
        torch.full((nout,), sd, device=h.device),
    )


@torch.no_grad()
def _fit_norm(norm: FrozenPerNeuronAffine, raw: torch.Tensor, target_mean: torch.Tensor, target_std: torch.Tensor, max_gain: float) -> None:
    raw_mean, raw_std = raw.mean(0), raw.std(0).clamp_min(1e-6)
    gain = torch.clamp(target_std / raw_std, max=max_gain)
    norm.set_(gain, target_mean - gain * raw_mean)


@torch.no_grad()
def initialize_positive_region_model(
    model: PositiveTTFSNet,
    init_name: str,
    calibration_x: torch.Tensor,
    seed: int,
    cfg: RegionInitConfig | None = None,
    delays_on: bool = False,
) -> list[dict]:
    """Initialize Init 1 (generic_positive) or Init 2 (max_regions)."""
    cfg = cfg or RegionInitConfig()
    aliases = {"init1": "generic_positive", "init2": "max_regions"}
    init_name = aliases.get(init_name.lower(), init_name)
    if init_name not in {"generic_positive", "max_regions"}:
        raise ValueError(f"Unknown positive TTFS init: {init_name}")

    init_encoder_nonzero(model.enc, seed)
    h = model.encode_times(calibration_x)
    rows = []
    for ell, (layer, norm) in enumerate(zip(model.layers, model.norms), start=1):
        layer_seed = seed + 10_000 * ell + 100 * layer.input_dim + layer.output_dim
        sigma = cfg.max_sigma if init_name == "max_regions" else cfg.generic_sigma
        _init_positive_layer(layer, sigma, layer_seed, cfg, delays_on)
        if init_name == "max_regions":
            _boundary_theta(layer, h, layer_seed + 1_000_000, cfg)
        else:
            _generic_theta(layer, layer_seed + 1_000_000, cfg)
        raw, cm = layer(h, return_causal=True)
        if init_name == "max_regions":
            target_mean, target_std = _norm_targets(h, layer.output_dim)
            _fit_norm(norm, raw, target_mean, target_std, cfg.max_gain)
        h = norm(raw)
        rows.append({
            "layer": ell,
            "input_dim": layer.input_dim,
            "output_dim": layer.output_dim,
            "init_name": init_name,
            "weight_sigma": sigma,
            "theta_mean": float(layer.threshold.mean()),
            "theta_std": float(layer.threshold.std()),
            "delay_min": float(layer.input_delay.min()),
            "delay_max": float(layer.input_delay.max()),
            "prefix_mean": float(cm.sum(-1).float().mean()),
            "prefix_std": float(cm.sum(-1).float().std()),
        })
    return rows


@torch.no_grad()
def initialize_shared_region_model(
    model: SharedTTFSNet,
    seed: int,
    cfg: RegionInitConfig | None = None,
    delays_on: bool = False,
) -> list[dict]:
    cfg = cfg or RegionInitConfig()
    # Preserve the shared-weight notebook's encoder seed convention.
    fan_in, fan_out = model.enc.weight.shape[1], model.enc.weight.shape[0]
    sd = math.sqrt(2.0 / (fan_in + fan_out))
    W = torch.randn(tuple(model.enc.weight.shape), generator=cpu_generator(seed + 1)) * sd
    tiny = W.abs() < 1e-8
    if tiny.any():
        signs = torch.where(
            torch.rand(int(tiny.sum()), generator=cpu_generator(seed + 2)) >= 0.5,
            1.0,
            -1.0,
        )
        W[tiny] = signs * 1e-6
    model.enc.weight.copy_(W.to(model.enc.weight.device))
    mag = 0.005 + 0.045 * torch.rand(model.enc.bias.numel(), generator=cpu_generator(seed + 3))
    sign = torch.where(
        torch.rand(model.enc.bias.numel(), generator=cpu_generator(seed + 4)) >= 0.5,
        1.0,
        -1.0,
    )
    model.enc.bias.copy_((mag * sign).to(model.enc.bias.device))

    rows = []
    for ell, layer in enumerate(model.layers, start=1):
        layer_seed = seed + 10_000 * ell + 100 * layer.input_dim + layer.output_dim
        z = torch.randn(layer.input_dim, generator=cpu_generator(layer_seed + 5))
        v = torch.exp(cfg.generic_sigma * z - 0.5 * cfg.generic_sigma**2)
        layer.shared_w.copy_(v.to(layer.shared_w.device))
        if delays_on:
            d = cfg.delay_min + (cfg.delay_max - cfg.delay_min) * torch.rand(
                layer.input_dim, generator=cpu_generator(layer_seed + 5)
            )
            layer.input_delay.copy_(d.to(layer.input_delay.device))
        else:
            layer.input_delay.zero_()
        u = torch.rand(layer.output_dim, generator=cpu_generator(layer_seed + 7))
        theta = torch.exp(
            math.log(cfg.theta_min)
            + u * (math.log(cfg.theta_max) - math.log(cfg.theta_min))
        )
        # Preserve strict distinctness safeguard from notebook.
        theta_sorted, order = torch.sort(theta)
        for k in range(1, theta_sorted.numel()):
            if theta_sorted[k] <= theta_sorted[k - 1]:
                theta_sorted[k] = theta_sorted[k - 1] * (1.0 + 1e-6)
        theta_distinct = torch.empty_like(theta_sorted)
        theta_distinct[order] = theta_sorted
        layer.threshold.copy_(theta_distinct.to(layer.threshold.device))
        rows.append({
            "model_family": "ttfs_shared",
            "init_name": "shared_generic_positive",
            "layer": ell,
            "input_dim": layer.input_dim,
            "output_dim": layer.output_dim,
            "shared_weight_mean": float(layer.shared_w.mean()),
            "shared_weight_std": float(layer.shared_w.std()),
            "theta_mean": float(layer.threshold.mean()),
            "theta_std": float(layer.threshold.std()),
        })
    for name, p in model.named_parameters():
        if "input_delay" not in name and torch.any(p == 0):
            raise RuntimeError(f"Parameter {name} contains exact zero entries")
    return rows


@torch.no_grad()
def initialize_signed_region_model(
    model: SignedTTFSNet,
    seed: int,
    cfg: RegionInitConfig | None = None,
    delays_on: bool = False,
) -> list[dict]:
    cfg = cfg or RegionInitConfig()
    init_encoder_nonzero(model.enc, seed)
    rows = []
    for ell, layer in enumerate(model.layers, start=1):
        layer_seed = seed + 10_000 * ell + 100 * layer.input_dim + layer.output_dim
        z = torch.randn(tuple(layer.w.shape), generator=cpu_generator(layer_seed + 4))
        W = cfg.signed_sigma * z
        tiny = W.abs() < 1e-8
        if tiny.any():
            replacement = torch.where(
                z[tiny] >= 0,
                torch.full_like(z[tiny], 1e-8),
                torch.full_like(z[tiny], -1e-8),
            )
            W[tiny] = replacement
        layer.w.copy_(W.to(layer.w.device))
        if delays_on:
            d = cfg.delay_min + (cfg.delay_max - cfg.delay_min) * torch.rand(
                layer.input_dim, generator=cpu_generator(layer_seed + 5)
            )
            layer.input_delay.copy_(d.to(layer.input_delay.device))
        else:
            layer.input_delay.zero_()
        _generic_theta(layer, layer_seed + 1_000_000, cfg)
        w = layer.w.detach()
        rows.append({
            "layer": ell,
            "input_dim": layer.input_dim,
            "output_dim": layer.output_dim,
            "model_family": "ttfs_signed_gaussian",
            "init_name": "generic_signed_gaussian",
            "weight_sigma": cfg.signed_sigma,
            "weight_mean": float(w.mean()),
            "weight_std": float(w.std()),
            "positive_weight_fraction": float((w > 0).float().mean()),
            "negative_weight_fraction": float((w < 0).float().mean()),
            "theta_mean": float(layer.threshold.mean()),
            "theta_std": float(layer.threshold.std()),
        })
    for name, p in model.named_parameters():
        if "input_delay" not in name and torch.any(p == 0):
            raise RuntimeError(f"Parameter {name} contains exact zero entries")
    return rows


# -----------------------------------------------------------------------------
# Training initializations
# -----------------------------------------------------------------------------

@torch.no_grad()
def initialize_ttfs_classifier_init1(
    model: PositiveTTFSClassifier,
    seed: int,
    cfg: RegionInitConfig | None = None,
) -> None:
    cfg = cfg or RegionInitConfig()
    init_encoder_nonzero(model.encoder, seed)
    for ell, layer in enumerate(model.hidden_layers, start=1):
        layer_seed = seed + 10_000 * ell + 100 * layer.input_dim + layer.output_dim
        _init_positive_layer(layer, cfg.generic_sigma, layer_seed, cfg, model.delays_on)
        _generic_theta(layer, layer_seed + 1_000_000, cfg)
    # Decoder is not part of the region signature.
    g = cpu_generator(seed + 9_000_000)
    fan_in, fan_out = model.decoder.weight.shape[1], model.decoder.weight.shape[0]
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    W = (2.0 * torch.rand(tuple(model.decoder.weight.shape), generator=g) - 1.0) * bound
    model.decoder.weight.copy_(W.to(model.decoder.weight.device))
    fill_nonzero_normal_(model.decoder.bias, 0.0, 0.01, cpu_generator(seed + 9_000_001))
    model.project_constraints_()


@torch.no_grad()
def initialize_relu_classifier(model: ReLUClassifier, seed: int) -> None:
    he_initialize_linear(model.encoder, seed + 1_000)
    for ell, layer in enumerate(model.hidden_layers, start=1):
        he_initialize_linear(layer, seed + 10_000 * ell)
    g = cpu_generator(seed + 9_000_000)
    fan_in, fan_out = model.decoder.weight.shape[1], model.decoder.weight.shape[0]
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    W = (2.0 * torch.rand(tuple(model.decoder.weight.shape), generator=g) - 1.0) * bound
    model.decoder.weight.copy_(W.to(model.decoder.weight.device))
    fill_nonzero_normal_(model.decoder.bias, 0.0, 0.01, cpu_generator(seed + 9_000_001))


@torch.no_grad()
def calibrate_encoder_time_scale(model: PositiveTTFSClassifier, calib_x: torch.Tensor, target_std: float = 1.0) -> None:
    X = calib_x.to(model.encoder.weight.device)
    t = -model.encoder(X)
    mu = t.mean(dim=0)
    std = t.std(dim=0, unbiased=False).clamp_min(1e-4)
    scale = target_std / std
    model.encoder.weight.mul_(scale.view(-1, 1))
    model.encoder.bias.copy_(scale * (model.encoder.bias + mu))


@torch.no_grad()
def initialize_ttfs_classifier_init3(
    model: PositiveTTFSClassifier,
    calib_x: torch.Tensor,
    seed: int,
    training_cfg: TrainingConfig | None = None,
    init_cfg: RegionInitConfig | None = None,
) -> None:
    """Training-oriented configuration used in the depth-one accuracy arm."""
    training_cfg = training_cfg or TrainingConfig()
    init_cfg = init_cfg or RegionInitConfig()
    if model.depth != 1:
        raise ValueError("The paper Init 3 accuracy arm is depth one.")
    init_encoder_nonzero(model.encoder, seed)
    calibrate_encoder_time_scale(
        model, calib_x, target_std=training_cfg.to_encoder_target_time_std
    )
    layer = model.hidden
    z = torch.randn(tuple(layer.w.shape), generator=cpu_generator(seed + 310_000))
    desired_w = torch.exp(init_cfg.generic_sigma * z - 0.5 * init_cfg.generic_sigma**2)
    desired_w = desired_w / desired_w.sum(dim=0, keepdim=True)
    desired_w = training_cfg.to_incoming_weight_sum * desired_w
    layer.w.copy_(desired_w.to(layer.w.device))
    layer.input_delay.zero_()

    X = calib_x.to(model.encoder.weight.device)
    arrivals = model.encode_times(X)
    sorted_arrivals, order = torch.sort(arrivals, dim=1)
    rng = np.random.default_rng(seed + 320_000)
    n, out = layer.input_dim, layer.output_dim
    desired_theta = torch.empty(out, device=layer.w.device)
    lo_frac, hi_frac = training_cfg.to_target_prefix_fraction
    for j in range(out):
        frac = float(rng.uniform(lo_frac, hi_frac))
        k = max(1, min(int(round(frac * n)), n - 1))
        wj = layer.w[:, j]
        w_sorted = wj[order]
        next_arrival = sorted_arrivals[:, k]
        theta_star = (
            w_sorted[:, :k]
            * (next_arrival.unsqueeze(1) - sorted_arrivals[:, :k])
        ).sum(dim=1)
        theta_med = torch.median(theta_star).clamp_min(model.theta_train_bounds[0])
        jitter = math.exp(float(rng.normal(0.0, training_cfg.to_threshold_log_jitter_std)))
        desired_theta[j] = theta_med * jitter
    layer.threshold.copy_(desired_theta)
    nn.init.xavier_uniform_(model.decoder.weight)
    model.decoder.bias.normal_(0.0, 0.01)
    model.project_constraints_()

# -----------------------------------------------------------------------------
# Exact-2D notebook initialization (kept separate because its seed convention
# differs from the width/depth and depth-one training notebooks).
# -----------------------------------------------------------------------------

@torch.no_grad()
def init_xavier_nonzero(layer: nn.Linear, seed: int) -> None:
    fan_in, fan_out = layer.weight.shape[1], layer.weight.shape[0]
    bound = math.sqrt(6.0 / (fan_in + fan_out))
    W = (2.0 * torch.rand(tuple(layer.weight.shape), generator=cpu_generator(seed)) - 1.0) * bound
    W[W.abs() < 1e-8] = 1e-8
    layer.weight.copy_(W.to(layer.weight.device))
    b = torch.randn(layer.bias.numel(), generator=cpu_generator(seed + 1)) * 0.01
    b[b.abs() < 1e-8] = 1e-8
    layer.bias.copy_(b.to(layer.bias.device))


@torch.no_grad()
def initialize_exact_ttfs(model: PositiveTTFSClassifier, seed: int, exact_cfg=None) -> None:
    from .configs import Exact2DConfig
    exact_cfg = exact_cfg or Exact2DConfig()
    init_xavier_nonzero(model.encoder, seed + 100)
    for ell, layer in enumerate(model.hidden_layers):
        base = seed + 1000 + 100 * ell
        z = torch.randn(tuple(layer.w.shape), generator=cpu_generator(base))
        W = torch.exp(exact_cfg.weight_sigma * z - 0.5 * exact_cfg.weight_sigma**2)
        layer.w.copy_(W.to(layer.w.device))
        u = torch.rand(layer.output_dim, generator=cpu_generator(base + 1))
        theta = torch.exp(
            math.log(exact_cfg.theta_min)
            + u * (math.log(exact_cfg.theta_max) - math.log(exact_cfg.theta_min))
        )
        layer.threshold.copy_(theta.to(layer.threshold.device))
        if exact_cfg.delays_on:
            d = exact_cfg.delay_min + (exact_cfg.delay_max - exact_cfg.delay_min) * torch.rand(
                layer.input_dim, generator=cpu_generator(base + 2)
            )
            layer.input_delay.copy_(d.to(layer.input_delay.device))
        else:
            layer.input_delay.zero_()
    init_xavier_nonzero(model.decoder, seed + 9000)
    model.project_constraints_()


@torch.no_grad()
def initialize_exact_relu(model: ReLUClassifier, seed: int) -> None:
    for ell, layer in enumerate([model.encoder, *model.hidden_layers]):
        fan_in = layer.weight.shape[1]
        W = torch.randn(tuple(layer.weight.shape), generator=cpu_generator(seed + 2000 + 100 * ell)) * math.sqrt(2.0 / fan_in)
        W[W.abs() < 1e-8] = 1e-8
        layer.weight.copy_(W.to(layer.weight.device))
        b = torch.randn(layer.bias.numel(), generator=cpu_generator(seed + 2001 + 100 * ell)) * 0.01
        b[b.abs() < 1e-8] = 1e-8
        layer.bias.copy_(b.to(layer.bias.device))
    init_xavier_nonzero(model.decoder, seed + 9500)
