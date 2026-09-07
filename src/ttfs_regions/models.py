from __future__ import annotations

import torch
import torch.nn as nn


class IdentityAffine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class FrozenPerNeuronAffine(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.register_buffer("gain", torch.ones(n))
        self.register_buffer("shift", torch.zeros(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain.view(1, -1) + self.shift.view(1, -1)

    @torch.no_grad()
    def set_(self, gain: torch.Tensor, shift: torch.Tensor) -> None:
        self.gain.copy_(gain)
        self.shift.copy_(shift)


class PositiveTTFSLayer(nn.Module):
    """Positive-weight TTFS layer using sorted causal-prefix accumulation."""

    def __init__(self, input_dim: int, output_dim: int, delays_trainable: bool = False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.w = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.threshold = nn.Parameter(torch.empty(self.output_dim))
        self.input_delay = nn.Parameter(
            torch.empty(self.input_dim), requires_grad=bool(delays_trainable)
        )

    def forward(self, x: torch.Tensor, return_causal: bool = False):
        B, N = x.shape
        O = self.output_dim
        if N != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {N}")

        arrivals = x + self.input_delay.view(1, -1)
        t_sorted, order = torch.sort(arrivals, dim=1)

        W = self.w.unsqueeze(0).expand(B, -1, -1)
        W_sorted = torch.gather(W, 1, order.unsqueeze(-1).expand(B, N, O))
        cum_w = torch.cumsum(W_sorted, dim=1)
        cum_wt = torch.cumsum(W_sorted * t_sorted.unsqueeze(-1), dim=1)
        candidate = (self.threshold.view(1, 1, O) + cum_wt) / (cum_w + 1e-12)

        valid = torch.ones((B, N, O), dtype=torch.bool, device=x.device)
        if N > 1:
            valid[:, :-1, :] = candidate[:, :-1, :] < t_sorted[:, 1:].unsqueeze(-1)

        prefix_index = torch.arange(N, device=x.device).view(1, N, 1)
        first = torch.where(valid, prefix_index, torch.full_like(prefix_index, N)).min(dim=1).values

        bi = torch.arange(B, device=x.device).unsqueeze(1).expand(B, O)
        oi = torch.arange(O, device=x.device).unsqueeze(0).expand(B, O)
        out = candidate[bi, first.long(), oi]

        if not return_causal:
            return out

        positions = torch.arange(N, device=x.device).view(1, N, 1)
        causal_sorted = positions <= first.unsqueeze(1)
        causal_original = torch.zeros((B, N, O), dtype=torch.bool, device=x.device)
        causal_original.scatter_(
            1, order.unsqueeze(-1).expand(B, N, O), causal_sorted
        )
        causal = causal_original.permute(0, 2, 1).contiguous()
        return out, causal


class SignedTTFSLayer(nn.Module):
    """TTFS layer with arbitrary signed weights.

    A prefix is admissible only if its cumulative weight is positive. If no
    prefix is admissible, the output is ``last_spike`` and the causal mask is
    all zero.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        delays_trainable: bool = False,
        denom_eps: float = 1e-8,
        last_spike: float = 100.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.denom_eps = float(denom_eps)
        self.last_spike = float(last_spike)
        self.w = nn.Parameter(torch.empty(self.input_dim, self.output_dim))
        self.threshold = nn.Parameter(torch.empty(self.output_dim))
        self.input_delay = nn.Parameter(
            torch.empty(self.input_dim), requires_grad=bool(delays_trainable)
        )

    def forward(self, x: torch.Tensor, return_causal: bool = False):
        B, N = x.shape
        O = self.output_dim
        arrivals = x + self.input_delay.view(1, -1)
        t_sorted, order = torch.sort(arrivals, dim=1)

        W = self.w.unsqueeze(0).expand(B, -1, -1)
        W_sorted = torch.gather(W, 1, order.unsqueeze(-1).expand(B, N, O))
        cum_w = torch.cumsum(W_sorted, dim=1)
        cum_wt = torch.cumsum(W_sorted * t_sorted.unsqueeze(-1), dim=1)

        denom_ok = cum_w > self.denom_eps
        safe_cum_w = torch.where(denom_ok, cum_w, torch.ones_like(cum_w))
        candidate = (self.threshold.view(1, 1, O) + cum_wt) / safe_cum_w

        valid = denom_ok.clone()
        if N > 1:
            valid[:, :-1, :] &= candidate[:, :-1, :] < t_sorted[:, 1:].unsqueeze(-1)

        idx = torch.arange(N, device=x.device).view(1, N, 1)
        invalid_index = torch.full_like(idx, N)
        first = torch.where(valid, idx, invalid_index).min(dim=1).values
        spiked = first < N
        safe_first = first.clamp(max=N - 1).long()

        bi = torch.arange(B, device=x.device).unsqueeze(1).expand(B, O)
        oi = torch.arange(O, device=x.device).unsqueeze(0).expand(B, O)
        out = candidate[bi, safe_first, oi]
        out = torch.where(spiked, out, torch.full_like(out, self.last_spike))

        if not return_causal:
            return out

        positions = torch.arange(N, device=x.device).view(1, N, 1)
        causal_sorted = (positions <= safe_first.unsqueeze(1)) & spiked.unsqueeze(1)
        causal_original = torch.zeros((B, N, O), dtype=torch.bool, device=x.device)
        causal_original.scatter_(
            1, order.unsqueeze(-1).expand(B, N, O), causal_sorted
        )
        causal = causal_original.permute(0, 2, 1).contiguous()
        return out, causal, spiked


class SharedTTFSLayer(nn.Module):
    """Positive TTFS layer sharing one incoming weight vector across outputs."""

    def __init__(self, input_dim: int, output_dim: int, delays_trainable: bool = False):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.shared_w = nn.Parameter(torch.empty(self.input_dim))
        self.threshold = nn.Parameter(torch.empty(self.output_dim))
        self.input_delay = nn.Parameter(
            torch.empty(self.input_dim), requires_grad=bool(delays_trainable)
        )

    def forward(self, x: torch.Tensor, return_causal: bool = False):
        B, N = x.shape
        O = self.output_dim
        arrivals = x + self.input_delay.view(1, -1)
        t_sorted, order = torch.sort(arrivals, dim=1)
        W = self.shared_w.view(1, N).expand(B, N)
        W_sorted = torch.gather(W, 1, order)
        cum_w = torch.cumsum(W_sorted, dim=1)
        cum_wt = torch.cumsum(W_sorted * t_sorted, dim=1)
        candidate = (
            self.threshold.view(1, 1, O) + cum_wt.unsqueeze(-1)
        ) / (cum_w.unsqueeze(-1) + 1e-12)

        valid = torch.ones((B, N, O), dtype=torch.bool, device=x.device)
        if N > 1:
            valid[:, :-1, :] = candidate[:, :-1, :] < t_sorted[:, 1:].unsqueeze(-1)
        idx = torch.arange(N, device=x.device).view(1, N, 1)
        first = torch.where(valid, idx, torch.full_like(idx, N)).min(dim=1).values
        bi = torch.arange(B, device=x.device).unsqueeze(1).expand(B, O)
        oi = torch.arange(O, device=x.device).unsqueeze(0).expand(B, O)
        out = candidate[bi, first.long(), oi]

        if not return_causal:
            return out
        pos = torch.arange(N, device=x.device).view(1, N, 1)
        causal_sorted = pos <= first.unsqueeze(1)
        causal_original = torch.zeros((B, N, O), dtype=torch.bool, device=x.device)
        causal_original.scatter_(
            1, order.unsqueeze(-1).expand(B, N, O), causal_sorted
        )
        return out, causal_original.permute(0, 2, 1).contiguous()


class PositiveTTFSNet(nn.Module):
    def __init__(
        self,
        raw_dim: int,
        encoder_width: int,
        widths: list[int] | tuple[int, ...],
        use_norm: bool = False,
        delays_trainable: bool = False,
    ):
        super().__init__()
        self.depth = len(widths)
        self.encoder_width = int(encoder_width)
        self.layer_widths = list(widths)
        self.enc = nn.Linear(raw_dim, encoder_width)
        input_dims = [encoder_width] + list(widths[:-1])
        self.layers = nn.ModuleList(
            [PositiveTTFSLayer(a, b, delays_trainable) for a, b in zip(input_dims, widths)]
        )
        self.norms = nn.ModuleList(
            [FrozenPerNeuronAffine(b) if use_norm else IdentityAffine() for b in widths]
        )

    def encode_times(self, x: torch.Tensor) -> torch.Tensor:
        return -self.enc(x.view(x.size(0), -1))


class SharedTTFSNet(nn.Module):
    def __init__(self, raw_dim: int, encoder_width: int, widths, delays_trainable: bool = False):
        super().__init__()
        self.depth = len(widths)
        self.encoder_width = int(encoder_width)
        self.layer_widths = list(widths)
        self.enc = nn.Linear(raw_dim, encoder_width)
        input_dims = [encoder_width] + list(widths[:-1])
        self.layers = nn.ModuleList(
            [SharedTTFSLayer(a, b, delays_trainable) for a, b in zip(input_dims, widths)]
        )
        self.norms = nn.ModuleList([IdentityAffine() for _ in widths])

    def encode_times(self, x: torch.Tensor) -> torch.Tensor:
        return -self.enc(x.view(x.size(0), -1))


class SignedTTFSNet(nn.Module):
    def __init__(
        self,
        raw_dim: int,
        encoder_width: int,
        widths,
        delays_trainable: bool = False,
        denom_eps: float = 1e-8,
        last_spike: float = 100.0,
    ):
        super().__init__()
        self.depth = len(widths)
        self.encoder_width = int(encoder_width)
        self.layer_widths = list(widths)
        self.enc = nn.Linear(raw_dim, encoder_width)
        input_dims = [encoder_width] + list(widths[:-1])
        self.layers = nn.ModuleList(
            [
                SignedTTFSLayer(a, b, delays_trainable, denom_eps, last_spike)
                for a, b in zip(input_dims, widths)
            ]
        )

    def encode_times(self, x: torch.Tensor) -> torch.Tensor:
        return -self.enc(x.view(x.size(0), -1))


class ReLUNet(nn.Module):
    def __init__(self, raw_dim: int, encoder_width: int, widths):
        super().__init__()
        self.depth = len(widths)
        self.encoder_width = int(encoder_width)
        self.layer_widths = list(widths)
        self.enc = nn.Linear(raw_dim, encoder_width)
        input_dims = [encoder_width] + list(widths[:-1])
        self.layers = nn.ModuleList([nn.Linear(a, b) for a, b in zip(input_dims, widths)])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc(x.view(x.size(0), -1))


class PositiveTTFSClassifier(nn.Module):
    def __init__(
        self,
        raw_dim: int,
        width: int,
        num_classes: int = 10,
        depth: int = 1,
        delays_on: bool = False,
        train_delays: bool = False,
        weight_train_bounds=(1e-5, 10.0),
        theta_train_bounds=(1e-4, 10.0),
        delay_train_bounds=(1e-4, 5.0),
    ):
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.delays_on = bool(delays_on)
        self.weight_train_bounds = weight_train_bounds
        self.theta_train_bounds = theta_train_bounds
        self.delay_train_bounds = delay_train_bounds
        self.encoder = nn.Linear(raw_dim, width)
        self.hidden_layers = nn.ModuleList(
            [
                PositiveTTFSLayer(width, width, delays_trainable=(delays_on and train_delays))
                for _ in range(depth)
            ]
        )
        self.decoder = nn.Linear(width, num_classes)

    @property
    def enc(self):
        return self.encoder

    @property
    def hidden(self):
        return self.hidden_layers[0]

    def encode_times(self, x: torch.Tensor) -> torch.Tensor:
        return -self.encoder(x.view(x.size(0), -1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.encode_times(x)
        for layer in self.hidden_layers:
            t = layer(t)
        return self.decoder(-t)

    def region_mask(self, x: torch.Tensor) -> torch.Tensor:
        t = self.encode_times(x)
        masks = []
        for layer in self.hidden_layers:
            t, cm = layer(t, return_causal=True)
            masks.append(cm.reshape(cm.size(0), -1))
        return torch.cat(masks, dim=1)

    @torch.no_grad()
    def project_constraints_(self) -> None:
        for layer in self.hidden_layers:
            layer.w.clamp_(*self.weight_train_bounds)
            layer.threshold.clamp_(*self.theta_train_bounds)
            if self.delays_on:
                layer.input_delay.clamp_(*self.delay_train_bounds)
            else:
                layer.input_delay.zero_()


class ReLUClassifier(nn.Module):
    def __init__(self, raw_dim: int, width: int, num_classes: int = 10, depth: int = 1):
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.encoder = nn.Linear(raw_dim, width)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.decoder = nn.Linear(width, num_classes)

    @property
    def enc(self):
        return self.encoder

    @property
    def hidden(self):
        return self.hidden_layers[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x.view(x.size(0), -1))
        for layer in self.hidden_layers:
            h = torch.relu(layer(h))
        return self.decoder(h)

    def region_mask(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x.view(x.size(0), -1))
        masks = []
        for layer in self.hidden_layers:
            z = layer(h)
            masks.append(z > 0)
            h = torch.relu(z)
        return torch.cat(masks, dim=1)
