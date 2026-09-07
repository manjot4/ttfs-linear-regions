from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import torch

from .models import PositiveTTFSNet, ReLUNet, SharedTTFSNet, SignedTTFSNet


def _chunked_trajectory(x0: torch.Tensor, x1: torch.Tensor, steps: int, batch_size: int, device: torch.device):
    npts = steps + 1
    x0 = x0.to(device)
    x1 = x1.to(device)
    for start in range(0, npts, batch_size):
        end = min(npts, start + batch_size)
        t = torch.arange(start, end, device=device, dtype=torch.float32) / steps
        t = t.view((end - start,) + (1,) * x0.ndim)
        xb = (1 - t) * x0.unsqueeze(0) + t * x1.unsqueeze(0)
        yield start, xb


@torch.no_grad()
def evaluate_pair_layerwise(model, x0: torch.Tensor, x1: torch.Tensor, steps: int, batch_size: int, device: torch.device | str) -> list[dict]:
    """Count traversed regions cumulatively by layer.

    The count is ``1 +`` the number of consecutive pattern changes, not the
    number of globally unique patterns along the sampled path.
    """
    device = torch.device(device)
    layer_boundaries = [np.zeros(steps, dtype=bool) for _ in range(model.depth)]
    previous_mask = [None] * model.depth
    signed_nonspike_counts = [0] * model.depth
    signed_event_counts = [0] * model.depth

    for start, xb in _chunked_trajectory(x0, x1, steps, batch_size, device):
        if isinstance(model, ReLUNet):
            h = model.encode(xb)
            for ell, layer in enumerate(model.layers):
                z = layer(h)
                mask = z > 0
                if previous_mask[ell] is None:
                    seq, offset = mask, start
                else:
                    seq = torch.cat([previous_mask[ell].unsqueeze(0), mask], dim=0)
                    offset = start - 1
                if seq.shape[0] > 1:
                    changed = (seq[1:] != seq[:-1]).any(dim=1).cpu().numpy()
                    layer_boundaries[ell][offset:offset + len(changed)] = changed
                previous_mask[ell] = mask[-1].detach().clone()
                h = torch.relu(z)
            continue

        h = model.encode_times(xb)
        if isinstance(model, SignedTTFSNet):
            layer_norms = [None] * model.depth
        else:
            layer_norms = getattr(model, "norms", [None] * model.depth)

        for ell, layer in enumerate(model.layers):
            if isinstance(model, SignedTTFSNet):
                raw, mask, spiked = layer(h, return_causal=True)
                signed_nonspike_counts[ell] += int((~spiked).sum().item())
                signed_event_counts[ell] += int(spiked.numel())
            else:
                raw, mask = layer(h, return_causal=True)
            if previous_mask[ell] is None:
                seq, offset = mask, start
            else:
                seq = torch.cat([previous_mask[ell].unsqueeze(0), mask], dim=0)
                offset = start - 1
            if seq.shape[0] > 1:
                changed = (seq[1:] != seq[:-1]).any(dim=2).any(dim=1).cpu().numpy()
                layer_boundaries[ell][offset:offset + len(changed)] = changed
            previous_mask[ell] = mask[-1].detach().clone()
            norm = layer_norms[ell]
            h = raw if norm is None else norm(raw)

    rows = []
    cumulative = np.zeros(steps, dtype=bool)
    for ell in range(model.depth):
        new_boundary = layer_boundaries[ell] & ~cumulative
        cumulative |= layer_boundaries[ell]
        regions = 1 + int(cumulative.sum())
        row = {
            "depth": ell + 1,
            "regions": regions,
            "new_regions_this_layer": int(new_boundary.sum()),
            "layer_boundary_steps": int(layer_boundaries[ell].sum()),
            "ceiling_fraction": regions / (steps + 1),
        }
        if isinstance(model, SignedTTFSNet):
            row["nonspiking_fraction"] = signed_nonspike_counts[ell] / max(1, signed_event_counts[ell])
        rows.append(row)
    return rows


@torch.no_grad()
def evaluate_pairs_layerwise(model, dataset, pairs: Iterable[tuple[int, int]], steps: int, batch_size: int, device) -> pd.DataFrame:
    rows = []
    for pair_id, (i, j) in enumerate(pairs):
        x0, _ = dataset[i]
        x1, _ = dataset[j]
        for row in evaluate_pair_layerwise(model, x0.float(), x1.float(), steps, batch_size, device):
            row["pair_id"] = pair_id
            rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def count_regions_classifier(model, x0: torch.Tensor, x1: torch.Tensor, steps: int, batch_size: int, device) -> tuple[int, float]:
    """Trajectory count for a classifier using its concatenated hidden region mask."""
    device = torch.device(device)
    boundaries = np.zeros(steps, dtype=bool)
    previous = None
    for start, xb in _chunked_trajectory(x0, x1, steps, batch_size, device):
        mask = model.region_mask(xb)
        if previous is None:
            seq, offset = mask, start
        else:
            seq = torch.cat([previous.unsqueeze(0), mask], dim=0)
            offset = start - 1
        if seq.shape[0] > 1:
            changed = (seq[1:] != seq[:-1]).any(dim=1).cpu().numpy()
            boundaries[offset:offset + len(changed)] = changed
        previous = mask[-1].detach().clone()
    regions = 1 + int(boundaries.sum())
    return regions, regions / (steps + 1)
