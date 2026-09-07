from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field


PAPER_SEEDS = (42, 96, 2, 0, 12345)
WIDTH_VALUES = (16, 32, 64, 128, 256, 512)
DEPTH_VALUES = tuple(range(1, 9))

# These schedules match the current MNIST/CIFAR-10 width/depth notebooks.
DEPTH_SCHEDULES = OrderedDict([
    ("32", [32, 32, 32, 32, 32, 32, 32, 32]),
    ("64", [64, 64, 64, 64, 64, 64, 64, 64]),
    ("128", [128, 128, 128, 128, 128, 128, 128, 128]),
    ("256", [256, 256, 256, 256, 256, 256, 256, 256]),
    ("Gradual", [256, 256, 256, 128, 128, 128, 64, 64]),
    ("More gradual", [512, 256, 256, 128, 128, 64, 64, 64]),
])


@dataclass(frozen=True)
class RegionInitConfig:
    generic_sigma: float = 0.5
    max_sigma: float = 1.5
    theta_min: float = 0.25
    theta_max: float = 4.0
    delay_min: float = 0.05
    delay_max: float = 1.25
    boundary_k_max_fraction: float = 0.50
    boundary_log_noise_std: float = 0.08
    max_gain: float = 20.0
    signed_sigma: float = 0.5
    signed_last_spike: float = 100.0
    denom_eps: float = 1e-8


@dataclass(frozen=True)
class WidthDepthConfig:
    steps: int = 20_000
    seeds: tuple[int, ...] = PAPER_SEEDS
    num_pairs: int = 10
    calibration_size: int = 128
    probe_seed: int = 2026
    widths: tuple[int, ...] = WIDTH_VALUES
    schedules: OrderedDict = field(default_factory=lambda: DEPTH_SCHEDULES.copy())
    ttfs_batch_size: int = 24
    relu_batch_size: int = 256
    shared_batch_size: int = 64
    signed_batch_size: int = 24
    delays_on: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 100
    checkpoints: tuple[int, ...] = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    widths: tuple[int, ...] = (64, 128, 256)
    seeds: tuple[int, ...] = PAPER_SEEDS
    region_steps: int = 20_000
    region_pairs: int = 10
    probe_seed: int = 2026
    train_batch_size: int = 128
    region_batch_size_snn: int = 16
    region_batch_size_relu: int = 1024
    # The current attached MNIST and CIFAR-10 training notebooks both use 1e-4.
    learning_rate: float = 1e-4
    weight_decay_mnist: float = 1e-4
    weight_decay_cifar10: float = 5e-4
    grad_clip: float = 5.0
    delays_on: bool = False
    weight_train_min: float = 1e-5
    weight_train_max: float = 10.0
    theta_train_min: float = 1e-4
    theta_train_max: float = 10.0
    delay_train_min: float = 1e-4
    delay_train_max: float = 5.0
    run_training_oriented_accuracy: bool = True
    to_learning_rate: float = 1e-3
    to_weight_decay: float = 1e-4
    to_grad_clip: float = 5.0
    to_encoder_calibration_samples: int = 512
    to_encoder_target_time_std: float = 1.0
    to_incoming_weight_sum: float = 1.0
    to_target_prefix_fraction: tuple[float, float] = (0.15, 0.40)
    to_threshold_log_jitter_std: float = 0.05


@dataclass(frozen=True)
class Exact2DConfig:
    full_dim_tol: float = 1e-9
    cross_tol: float = 1e-8
    cross_step_frac: float = 1e-5
    geom_tol: float = 1e-8
    max_cells: int = 100_000
    validation_grid_res: int = 90
    margin: float = 0.30
    delays_on: bool = True
    train_delays: bool = False
    weight_sigma: float = 0.5
    theta_min: float = 0.25
    theta_max: float = 4.0
    delay_min: float = 0.05
    delay_max: float = 1.25
    weight_train_min: float = 1e-5
    weight_train_max: float = 50.0
    theta_train_min: float = 1e-5
    theta_train_max: float = 50.0
    delay_train_min: float = 1e-4
    delay_train_max: float = 5.0
