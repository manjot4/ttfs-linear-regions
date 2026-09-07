from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from scipy.optimize import linprog

from .configs import Exact2DConfig
from .models import PositiveTTFSClassifier, ReLUClassifier


class EnumerationLimit(RuntimeError):
    pass


@dataclass
class Slice2D:
    x0: torch.Tensor
    e1: torch.Tensor
    e2: torch.Tensor
    bounds: tuple[float, float, float, float]
    ref_uv: np.ndarray | None = None

    @property
    def box_A(self) -> np.ndarray:
        u0, u1, v0, v1 = self.bounds
        return np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    @property
    def box_b(self) -> np.ndarray:
        u0, u1, v0, v1 = self.bounds
        return np.array([u1, -u0, v1, -v0])

    @property
    def box_seed(self) -> np.ndarray:
        u0, u1, v0, v1 = self.bounds
        return np.array([0.5 * (u0 + u1), 0.5 * (v0 + v1)])

    def points(self, uv: torch.Tensor) -> torch.Tensor:
        return (
            self.x0.view(1, -1)
            + uv[:, 0:1] * self.e1.view(1, -1)
            + uv[:, 1:2] * self.e2.view(1, -1)
        )


def tight_square_bounds(ref_uv: np.ndarray | torch.Tensor, margin: float = 0.30, scale: float = 1.0):
    uv = np.asarray(ref_uv, dtype=float)
    umin, umax = float(uv[:, 0].min()), float(uv[:, 0].max())
    vmin, vmax = float(uv[:, 1].min()), float(uv[:, 1].max())
    uc, vc = 0.5 * (umin + umax), 0.5 * (vmin + vmax)
    span = max(umax - umin, vmax - vmin, 0.001)
    half = 0.5 * span * (1.0 + 2.0 * margin) * float(scale)
    return (uc - half, uc + half, vc - half, vc + half)


def make_slice_from_three(x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor, margin: float = 0.30, scale: float = 1.0) -> Slice2D:
    x0 = x0.view(-1).float().cpu()
    x1 = x1.view(-1).float().cpu()
    x2 = x2.view(-1).float().cpu()
    d1 = x1 - x0
    d1_norm = d1.norm()
    if float(d1_norm) < 1e-10:
        raise RuntimeError("Reference samples 0 and 1 are identical.")
    e1 = d1 / d1_norm
    d2 = x2 - x0
    d2_perp = d2 - torch.dot(d2, e1) * e1
    d2_norm = d2_perp.norm()
    if float(d2_norm) < 1e-10:
        raise RuntimeError("The three reference points are nearly collinear.")
    e2 = d2_perp / d2_norm
    ref_uv = torch.stack([
        torch.tensor([0.0, 0.0]),
        torch.tensor([torch.dot(x1 - x0, e1), torch.dot(x1 - x0, e2)]),
        torch.tensor([torch.dot(x2 - x0, e1), torch.dot(x2 - x0, e2)]),
    ]).numpy()
    bounds = tight_square_bounds(ref_uv, margin=margin, scale=scale)
    return Slice2D(x0=x0, e1=e1, e2=e2, bounds=bounds, ref_uv=ref_uv)


def torch_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(float, copy=False)


def affine_from_linear_on_slice(layer: torch.nn.Linear, sl: Slice2D, sign: float = 1.0) -> np.ndarray:
    W, b = torch_to_np(layer.weight), torch_to_np(layer.bias)
    x0, e1, e2 = torch_to_np(sl.x0), torch_to_np(sl.e1), torch_to_np(sl.e2)
    return np.stack([
        sign * (W @ e1),
        sign * (W @ e2),
        sign * (W @ x0 + b),
    ], axis=1)


def chebyshev_center(A: np.ndarray, b: np.ndarray):
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    norms = np.linalg.norm(A, axis=1)
    A3 = np.column_stack([A, norms])
    c = np.array([0.0, 0.0, -1.0])
    res = linprog(
        c,
        A_ub=A3,
        b_ub=b,
        bounds=[(None, None), (None, None), (0.0, None)],
        method="highs",
    )
    if not res.success:
        return None, -np.inf
    return res.x[:2], float(res.x[2])


def full_dimensional(A: np.ndarray, b: np.ndarray, tol: float = 1e-9):
    x, radius = chebyshev_center(A, b)
    return (x is not None and radius > tol, x, radius)


def clip_polygon_halfspace(polygon: np.ndarray, a: np.ndarray, rhs: float, tol: float = 1e-8) -> np.ndarray:
    if len(polygon) == 0:
        return polygon
    a = np.asarray(a, dtype=float)
    out = []
    for i in range(len(polygon)):
        S, E = polygon[i], polygon[(i + 1) % len(polygon)]
        fS, fE = float(a @ S - rhs), float(a @ E - rhs)
        inside_S, inside_E = fS <= tol, fE <= tol
        if inside_S and inside_E:
            out.append(E)
        elif inside_S and not inside_E:
            denom = fS - fE
            if abs(denom) > tol:
                out.append(S + (fS / denom) * (E - S))
        elif not inside_S and inside_E:
            denom = fS - fE
            if abs(denom) > tol:
                out.append(S + (fS / denom) * (E - S))
            out.append(E)
    if not out:
        return np.empty((0, 2), dtype=float)
    cleaned = []
    for p in out:
        if not cleaned or np.linalg.norm(p - cleaned[-1]) > tol:
            cleaned.append(p)
    if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= tol:
        cleaned.pop()
    return np.asarray(cleaned, dtype=float)


def polygon_from_constraints(A: np.ndarray, b: np.ndarray, bounds, tol: float = 1e-8) -> np.ndarray:
    u0, u1, v0, v1 = bounds
    poly = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]], dtype=float)
    for a, rhs in zip(np.asarray(A), np.asarray(b)):
        poly = clip_polygon_halfspace(poly, a, rhs, tol=tol)
        if len(poly) == 0:
            break
    return poly


def causal_pattern_at_point(uv: np.ndarray, arrival_coeff: np.ndarray, w: np.ndarray, theta: float):
    uv = np.asarray(uv, dtype=float)
    arrival = arrival_coeff[:, :2] @ uv + arrival_coeff[:, 2]
    order = np.argsort(arrival)
    sum_w = 0.0
    sum_wt = 0.0
    for k, idx in enumerate(order):
        sum_w += float(w[idx])
        sum_wt += float(w[idx]) * float(arrival[idx])
        t = (float(theta) + sum_wt) / sum_w
        if k == len(order) - 1 or t < arrival[order[k + 1]]:
            S = np.zeros(len(order), dtype=bool)
            S[order[: k + 1]] = True
            return tuple(bool(x) for x in S)
    raise RuntimeError("Positive TTFS prefix search failed.")


def causal_constraints(arrival_coeff: np.ndarray, w: np.ndarray, theta: float, causal_pattern):
    S = np.asarray(causal_pattern, dtype=bool)
    sum_w = float(w[S].sum())
    if sum_w <= 0:
        raise RuntimeError("Positive-weight enumeration received non-positive causal sum.")
    numerator = (w[S, None] * arrival_coeff[S]).sum(axis=0).copy()
    numerator[2] += float(theta)
    t_coeff = numerator / sum_w
    H, h = [], []
    for i in range(len(w)):
        # S: t_i <= t_S.  not S: t_S <= t_i.
        diff = arrival_coeff[i] - t_coeff if S[i] else t_coeff - arrival_coeff[i]
        H.append(diff[:2])
        h.append(-diff[2])
    return np.asarray(H), np.asarray(h), t_coeff


def enumerate_single_ttfs_neuron(
    parent_A,
    parent_b,
    parent_seed,
    arrival_coeff,
    w,
    theta,
    config: Exact2DConfig,
):
    start = causal_pattern_at_point(parent_seed, arrival_coeff, w, theta)
    queue, found = [start], {}
    while queue:
        S = queue.pop()
        if S in found:
            continue
        H, h, t_coeff = causal_constraints(arrival_coeff, w, theta, S)
        A, b = np.vstack([parent_A, H]), np.concatenate([parent_b, h])
        feasible, x_in, radius = full_dimensional(A, b, config.full_dim_tol)
        if not feasible:
            continue
        found[S] = {"A": A, "b": b, "seed": x_in, "radius": radius, "t_coeff": t_coeff}

        # Facets are navigation devices only. Crossing is accepted only if the
        # actual TTFS forward rule returns a different causal set.
        for i in range(len(w)):
            H_other, h_other = np.delete(H, i, axis=0), np.delete(h, i, axis=0)
            A_other, b_other = np.vstack([parent_A, H_other]), np.concatenate([parent_b, h_other])
            res = linprog(
                -H[i],
                A_ub=A_other,
                b_ub=b_other,
                bounds=[(None, None), (None, None)],
                method="highs",
            )
            if not res.success:
                continue
            x_out = res.x
            violation = float(H[i] @ x_out - h[i])
            if violation <= config.cross_tol:
                continue
            inside_value = float(H[i] @ x_in - h[i])
            denom = violation - inside_value
            if denom <= 0:
                continue
            alpha = -inside_value / denom
            neighbor = None
            for frac in (config.cross_step_frac, 1e-4, 1e-3, 1e-2, 5e-2):
                alpha2 = min(alpha + frac * (1.0 - alpha), 1.0)
                x_cross = x_in + alpha2 * (x_out - x_in)
                S2 = causal_pattern_at_point(x_cross, arrival_coeff, w, theta)
                if S2 != S:
                    neighbor = S2
                    break
            if neighbor is not None and neighbor not in found:
                queue.append(neighbor)
    return found


def _enumerate_ttfs_layer_for_parent(parent, layer, layer_index: int, config: Exact2DConfig, completed_cells: int = 0):
    arrival_coeff = parent["feature_coeffs"].copy()
    arrival_coeff[:, 2] += torch_to_np(layer.input_delay)
    W, theta = torch_to_np(layer.w), torch_to_np(layer.threshold)
    subcells = [{
        "A": parent["A"],
        "b": parent["b"],
        "seed": parent["seed"],
        "signature": parent["signature"],
        "out_coeffs": tuple(),
    }]
    for j in range(W.shape[1]):
        new_subcells = []
        for sub in subcells:
            local_regions = enumerate_single_ttfs_neuron(
                sub["A"], sub["b"], sub["seed"], arrival_coeff, W[:, j], float(theta[j]), config
            )
            for S, local in local_regions.items():
                new_subcells.append({
                    "A": local["A"],
                    "b": local["b"],
                    "seed": local["seed"],
                    "signature": sub["signature"] + (S,),
                    "out_coeffs": sub["out_coeffs"] + (local["t_coeff"],),
                })
                if completed_cells + len(new_subcells) > config.max_cells:
                    raise EnumerationLimit(
                        f"TTFS enumeration exceeded max_cells={config.max_cells} "
                        f"in layer {layer_index + 1}, neuron {j + 1}."
                    )
        subcells = new_subcells
        if not subcells:
            break
    for sub in subcells:
        sub["feature_coeffs"] = np.stack(sub.pop("out_coeffs"), axis=0)
    return subcells


def enumerate_ttfs(model: PositiveTTFSClassifier, sl: Slice2D, config: Exact2DConfig | None = None, verbose: bool = False):
    config = config or Exact2DConfig()
    cells = [{
        "A": sl.box_A.copy(),
        "b": sl.box_b.copy(),
        "seed": sl.box_seed.copy(),
        "signature": tuple(),
        "feature_coeffs": affine_from_linear_on_slice(model.encoder, sl, sign=-1.0),
    }]
    for ell, layer in enumerate(model.hidden_layers):
        next_cells = []
        for pidx, parent in enumerate(cells):
            next_cells.extend(_enumerate_ttfs_layer_for_parent(parent, layer, ell, config, len(next_cells)))
            if len(next_cells) > config.max_cells:
                raise EnumerationLimit(f"TTFS enumeration exceeded max_cells={config.max_cells} in layer {ell + 1}.")
        cells = next_cells
        if verbose:
            print(f"TTFS layer {ell + 1}/{model.depth}: {len(cells)} exact cells")
    return cells


def _relu_preactivation_coefficients(layer, input_coeffs):
    z = torch_to_np(layer.weight) @ input_coeffs
    z[:, 2] += torch_to_np(layer.bias)
    return z


def _enumerate_relu_layer_for_parent(parent, layer, layer_index: int, config: Exact2DConfig, completed_cells: int = 0):
    z_coeff = _relu_preactivation_coefficients(layer, parent["feature_coeffs"])
    subcells = [{
        "A": parent["A"],
        "b": parent["b"],
        "seed": parent["seed"],
        "signature": parent["signature"],
        "out_coeffs": tuple(),
    }]
    zero = np.zeros(3, dtype=float)
    for j, (p, q, r) in enumerate(z_coeff):
        new_subcells = []
        for sub in subcells:
            # inactive: p u + q v + r <= 0
            A0 = np.vstack([sub["A"], [p, q]])
            b0 = np.concatenate([sub["b"], [-r]])
            ok0, seed0, _ = full_dimensional(A0, b0, config.full_dim_tol)
            if ok0:
                new_subcells.append({
                    "A": A0,
                    "b": b0,
                    "seed": seed0,
                    "signature": sub["signature"] + (False,),
                    "out_coeffs": sub["out_coeffs"] + (zero.copy(),),
                })
            # active: p u + q v + r >= 0
            A1 = np.vstack([sub["A"], [-p, -q]])
            b1 = np.concatenate([sub["b"], [r]])
            ok1, seed1, _ = full_dimensional(A1, b1, config.full_dim_tol)
            if ok1:
                new_subcells.append({
                    "A": A1,
                    "b": b1,
                    "seed": seed1,
                    "signature": sub["signature"] + (True,),
                    "out_coeffs": sub["out_coeffs"] + (z_coeff[j].copy(),),
                })
            if completed_cells + len(new_subcells) > config.max_cells:
                raise EnumerationLimit(
                    f"ReLU enumeration exceeded max_cells={config.max_cells} "
                    f"in layer {layer_index + 1}, neuron {j + 1}."
                )
        subcells = new_subcells
        if not subcells:
            break
    for sub in subcells:
        sub["feature_coeffs"] = np.stack(sub.pop("out_coeffs"), axis=0)
    return subcells


def enumerate_relu(model: ReLUClassifier, sl: Slice2D, config: Exact2DConfig | None = None, verbose: bool = False):
    config = config or Exact2DConfig()
    cells = [{
        "A": sl.box_A.copy(),
        "b": sl.box_b.copy(),
        "seed": sl.box_seed.copy(),
        "signature": tuple(),
        "feature_coeffs": affine_from_linear_on_slice(model.encoder, sl, sign=1.0),
    }]
    for ell, layer in enumerate(model.hidden_layers):
        next_cells = []
        for parent in cells:
            next_cells.extend(_enumerate_relu_layer_for_parent(parent, layer, ell, config, len(next_cells)))
            if len(next_cells) > config.max_cells:
                raise EnumerationLimit(f"ReLU enumeration exceeded max_cells={config.max_cells} in layer {ell + 1}.")
        cells = next_cells
        if verbose:
            print(f"ReLU layer {ell + 1}/{model.depth}: {len(cells)} exact cells")
    return cells


def logit_coefficients(model, cell, model_kind: str):
    features = cell["feature_coeffs"]
    if model_kind == "ttfs":
        features = -features
    elif model_kind != "relu":
        raise ValueError(model_kind)
    logits = torch_to_np(model.decoder.weight) @ features
    logits[:, 2] += torch_to_np(model.decoder.bias)
    return logits


def unique_points(points, tol: float = 1e-8):
    out = []
    for p in points:
        p = np.asarray(p, dtype=float)
        if not any(np.linalg.norm(p - q) <= tol for q in out):
            out.append(p)
    return out


def line_segment_in_polygon(poly: np.ndarray, line_coeff: np.ndarray, tol: float = 1e-8):
    if len(poly) < 2:
        return None
    a, b, c = line_coeff
    vals = poly @ np.array([a, b]) + c
    pts = []
    for i in range(len(poly)):
        P, Q = poly[i], poly[(i + 1) % len(poly)]
        fP, fQ = vals[i], vals[(i + 1) % len(poly)]
        if abs(fP) <= tol:
            pts.append(P)
        if fP * fQ < -tol**2:
            pts.append(P + (fP / (fP - fQ)) * (Q - P))
        elif abs(fP) <= tol and abs(fQ) <= tol:
            pts.append(Q)
    pts = unique_points(pts, tol)
    if len(pts) < 2:
        return None
    best, best_dist = None, -1.0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist > best_dist:
                best, best_dist = (pts[i], pts[j]), dist
    return None if best is None or best_dist <= tol else best


def exact_decision_boundary_pieces(model, cells, model_kind: str, sl: Slice2D, num_classes: int = 10, config: Exact2DConfig | None = None):
    """Return exact class decision-boundary segments inside enumerated cells.

    Pairwise logit equality is intersected with class-dominance inequalities;
    therefore these are genuine multiclass decision-boundary pieces, not all
    equal-logit lines.
    """
    config = config or Exact2DConfig()
    pieces = []
    for cell_index, cell in enumerate(cells):
        poly = polygon_from_constraints(cell["A"], cell["b"], sl.bounds, config.geom_tol)
        if len(poly) < 3:
            continue
        logits = logit_coefficients(model, cell, model_kind)
        for r in range(num_classes):
            for s in range(r + 1, num_classes):
                p = poly.copy()
                for k in range(num_classes):
                    if k in (r, s):
                        continue
                    # g_k <= g_r on the r/s boundary candidate.
                    diff = logits[k] - logits[r]
                    p = clip_polygon_halfspace(p, diff[:2], -diff[2], config.geom_tol)
                    if len(p) == 0:
                        break
                if len(p) == 0:
                    continue
                segment = line_segment_in_polygon(p, logits[r] - logits[s], config.geom_tol)
                if segment is not None:
                    pieces.append({
                        "cell_index": cell_index,
                        "class_a": r,
                        "class_b": s,
                        "p0": segment[0],
                        "p1": segment[1],
                    })
    return pieces


def make_validation_points(sl: Slice2D, resolution: int = 90) -> torch.Tensor:
    u = torch.linspace(sl.bounds[0], sl.bounds[1], resolution)
    v = torch.linspace(sl.bounds[2], sl.bounds[3], resolution)
    uu, vv = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([uu.reshape(-1), vv.reshape(-1)], dim=1)
    return sl.points(uv).float()


def _ttfs_mask_to_signature(model: PositiveTTFSClassifier, row: np.ndarray):
    signature = []
    offset = 0
    for layer in model.hidden_layers:
        size = layer.output_dim * layer.input_dim
        block = row[offset:offset + size].reshape(layer.output_dim, layer.input_dim)
        signature.extend(tuple(bool(x) for x in neuron_row) for neuron_row in block)
        offset += size
    return tuple(signature)


@torch.no_grad()
def sampled_signatures(model, sl: Slice2D, model_kind: str, resolution: int = 90, device: str | torch.device = "cpu"):
    X = make_validation_points(sl, resolution)
    patterns = []
    device = torch.device(device)
    for start in range(0, len(X), 1024):
        xb = X[start:start + 1024].to(device)
        mask = model.region_mask(xb).detach().cpu().numpy().astype(bool)
        for row in mask:
            if model_kind == "ttfs":
                patterns.append(_ttfs_mask_to_signature(model, row))
            elif model_kind == "relu":
                patterns.append(tuple(bool(x) for x in row))
            else:
                raise ValueError(model_kind)
    return set(patterns)


def validate_against_grid(model, cells, sl: Slice2D, model_kind: str, resolution: int = 90, device="cpu"):
    sampled = sampled_signatures(model, sl, model_kind, resolution, device)
    exact = {cell["signature"] for cell in cells}
    missing = sampled - exact
    if missing:
        raise AssertionError(f"Validation failed: {len(missing)} sampled patterns missing from exact enumeration")
    return {
        "sampled_unique_patterns": len(sampled),
        "exact_enumerated_cells": len(exact),
        "additional_exact_cells": max(0, len(exact) - len(sampled)),
    }
