from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from e3nn import o3
from e3nn.nn._fc import _Layer as E3NNFCLayer
from torch import nn


def build_lora_irreps(
    irreps_in: o3.Irreps, irreps_out: o3.Irreps, rank: int
) -> o3.Irreps:
    """
    Choose an equivariant bottleneck irreps that preserves symmetry: for every irrep
    present in BOTH input and output, allocate `rank` copies.
    """
    in_set = {ir for _, ir in o3.Irreps(irreps_in)}
    out_set = {ir for _, ir in o3.Irreps(irreps_out)}
    shared = sorted(in_set & out_set, key=lambda ir: (ir.l, ir.p))
    if not shared:
        raise ValueError(
            f"No shared irreps between input ({irreps_in}) and output ({irreps_out}); cannot build equivariant LoRA."
        )
    parts = [f"{rank}x{ir}" for ir in shared]
    return o3.Irreps(" + ".join(parts))


class LoRAO3Linear(nn.Module):
    """LoRA for equivariant o3.Linear-like layers (preserves O(3) equivariance).

    Uses fused weight computation: W_merged = W_base + scaling * (W_A @ W_B)
    with automatic caching during inference (when grad is disabled).
    """

    def __init__(self, base_linear: o3.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.irreps_in = self.base.irreps_in
        self.irreps_out = self.base.irreps_out
        self.scaling = float(alpha) / float(rank)
        self.lora_irreps = build_lora_irreps(self.irreps_in, self.irreps_out, rank)

        layer_type = type(self.base)
        self.lora_A = layer_type(
            self.irreps_in, self.lora_irreps, internal_weights=True, biases=False
        )
        self.lora_B = layer_type(
            self.lora_irreps, self.irreps_out, internal_weights=True, biases=False
        )

        base_param = next(self.base.parameters())
        self.lora_A.to(dtype=base_param.dtype, device=base_param.device)
        self.lora_B.to(dtype=base_param.dtype, device=base_param.device)

        self._cached_merged_weight: torch.Tensor | None = None
        self._build_instruction_mapping()

        with torch.no_grad():
            for p in self.lora_B.parameters():
                p.zero_()
            for p in self.lora_A.parameters():
                if p.dim() >= 2:
                    p.normal_(mean=0.0, std=1e-3)

    def _build_instruction_mapping(self) -> None:
        """Build lookup tables for matching instructions between base, A, and B."""
        self._A_by_i_in = {}
        for idx, instr in enumerate(self.lora_A.instructions):
            self._A_by_i_in[instr.i_in] = (idx, instr.i_out, instr.path_weight)

        self._B_by_in_out = {}
        for idx, instr in enumerate(self.lora_B.instructions):
            self._B_by_in_out[(instr.i_in, instr.i_out)] = (idx, instr.path_weight)

    @staticmethod
    def _extract_weight_blocks(linear: o3.Linear) -> dict[int, torch.Tensor]:
        """Extract weight blocks indexed by instruction."""
        blocks = {}
        offset = 0
        for idx, instr in enumerate(linear.instructions):
            if instr.i_in == -1:
                continue
            size = math.prod(instr.path_shape)
            block = linear.weight[offset : offset + size].reshape(instr.path_shape)
            blocks[idx] = block
            offset += size
        return blocks

    def compute_merged_weight(self) -> torch.Tensor:
        """Compute W_base + scaling * composed(W_A, W_B) in weight space."""
        base_blocks = self._extract_weight_blocks(self.base)
        A_blocks = self._extract_weight_blocks(self.lora_A)
        B_blocks = self._extract_weight_blocks(self.lora_B)

        merged_blocks = []
        for base_idx, base_instr in enumerate(self.base.instructions):
            i_in_base = base_instr.i_in
            i_out_base = base_instr.i_out
            pw_base = base_instr.path_weight

            if i_in_base == -1:
                continue

            if i_in_base not in self._A_by_i_in:
                merged_blocks.append(base_blocks[base_idx])
                continue

            A_idx, i_mid, pw_A = self._A_by_i_in[i_in_base]
            B_key = (i_mid, i_out_base)
            if B_key not in self._B_by_in_out:
                merged_blocks.append(base_blocks[base_idx])
                continue

            B_idx, pw_B = self._B_by_in_out[B_key]
            ratio = (pw_A * pw_B) / pw_base
            delta = A_blocks[A_idx] @ B_blocks[B_idx]
            merged = base_blocks[base_idx] + self.scaling * ratio * delta
            merged_blocks.append(merged)

        return torch.cat([b.flatten() for b in merged_blocks])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            self._cached_merged_weight = None
            return self.base(x) + self.scaling * self.lora_B(self.lora_A(x))

        if self._cached_merged_weight is None:
            self._cached_merged_weight = self.compute_merged_weight()

        original_weight = self.base.weight.data
        self.base.weight.data = self._cached_merged_weight
        try:
            return self.base(x)
        finally:
            self.base.weight.data = original_weight

    def merge_into_base(self) -> o3.Linear:
        """Permanently merge LoRA weights into base and return the base layer."""
        with torch.no_grad():
            self.base.weight.copy_(self.compute_merged_weight())
        return self.base


class LoRADenseLinear(nn.Module):
    """LoRA for torch.nn.Linear.

    Uses fused weight computation: W_merged = W_base + scaling * (W_B @ W_A)
    with automatic caching during inference (when grad is disabled).
    """

    def __init__(self, base_linear: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.scaling = float(alpha) / float(rank)

        self.lora_A = nn.Linear(self.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, self.out_features, bias=False)

        base_param = next(self.base.parameters())
        self.lora_A.to(dtype=base_param.dtype, device=base_param.device)
        self.lora_B.to(dtype=base_param.dtype, device=base_param.device)

        self._cached_delta: torch.Tensor | None = None

        with torch.no_grad():
            nn.init.zeros_(self.lora_B.weight)
            nn.init.normal_(self.lora_A.weight, mean=0.0, std=1e-3)

    def compute_delta(self) -> torch.Tensor:
        """Compute the LoRA weight delta: W_B @ W_A."""
        return self.lora_B.weight @ self.lora_A.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            self._cached_delta = None
            delta = self.compute_delta()
        else:
            if self._cached_delta is None:
                self._cached_delta = self.compute_delta()
            delta = self._cached_delta

        merged_weight = self.base.weight + self.scaling * delta
        return F.linear(x, merged_weight, self.base.bias)

    def merge_into_base(self) -> nn.Linear:
        """Permanently merge LoRA weights into base and return the base layer."""
        with torch.no_grad():
            self.base.weight.add_(self.scaling * self.compute_delta())
        return self.base


class LoRAFCLayer(nn.Module):
    """LoRA for e3nn.nn._fc._Layer used by FullyConnectedNet (scalar MLP).

    Uses fused weight computation: W_merged = W_base + scaling * (A @ B)
    with automatic caching during inference (when grad is disabled).

    Note: e3nn uses (in, out) weight layout, so delta = A @ B (not B @ A).
    """

    def __init__(self, base_layer: nn.Module, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError("LoRAFCLayer requires a layer with a 'weight' parameter")
        self.base = base_layer

        w = self.base.weight  # type: ignore[attr-defined]
        in_f, out_f = int(w.shape[0]), int(w.shape[1])
        self.scaling = float(alpha) / float(rank)

        self.lora_A = nn.Parameter(
            torch.empty(in_f, rank, device=w.device, dtype=w.dtype)
        )
        self.lora_B = nn.Parameter(
            torch.empty(rank, out_f, device=w.device, dtype=w.dtype)
        )
        self._cached_delta: torch.Tensor | None = None

        with torch.no_grad():
            nn.init.normal_(self.lora_A, mean=0.0, std=1e-3)
            nn.init.zeros_(self.lora_B)

    def compute_delta(self) -> torch.Tensor:
        """Compute the LoRA weight delta: A @ B."""
        return self.lora_A @ self.lora_B

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            self._cached_delta = None
            delta = self.compute_delta()
        else:
            if self._cached_delta is None:
                self._cached_delta = self.compute_delta()
            delta = self._cached_delta

        merged_weight = self.base.weight + self.scaling * delta

        w_orig = self.base.weight
        del self.base._parameters["weight"]  # pylint: disable=protected-access
        self.base.weight = merged_weight
        try:
            return self.base(x)
        finally:
            self.base.weight = w_orig
            self.base._parameters["weight"] = w_orig  # pylint: disable=protected-access

    def merge_into_base(self) -> nn.Module:
        """Permanently merge LoRA weights into base and return the base layer."""
        with torch.no_grad():
            self.base.weight.add_(self.scaling * self.compute_delta())
        return self.base


def inject_lora(
    module: nn.Module,
    rank: int = 4,
    alpha: float = 1.0,
    wrap_equivariant: bool = True,
    wrap_dense: bool = True,
    _is_root: bool = True,
) -> None:
    """Recursively replace eligible linears with LoRA-wrapped versions."""
    for child_name, child in list(module.named_children()):
        if isinstance(child, (LoRAO3Linear, LoRADenseLinear, LoRAFCLayer)):
            continue
        if wrap_equivariant and isinstance(child, o3.Linear):
            try:
                wrapped = LoRAO3Linear(child, rank=rank, alpha=alpha)
            except ValueError:
                continue
            setattr(module, child_name, wrapped)
        if wrap_dense and isinstance(child, nn.Linear):
            wrapped = LoRADenseLinear(child, rank=rank, alpha=alpha)
            setattr(module, child_name, wrapped)
            continue
        if wrap_dense and isinstance(child, E3NNFCLayer):
            wrapped = LoRAFCLayer(child, rank=rank, alpha=alpha)
            setattr(module, child_name, wrapped)
            continue
        inject_lora(child, rank, alpha, wrap_equivariant, wrap_dense, _is_root=False)

    if _is_root:
        for name, p in module.named_parameters():
            p.requires_grad = ("lora_A" in name) or ("lora_B" in name)


def inject_LoRAs(model: nn.Module, rank: int = 4, alpha: int = 1):
    inject_lora(model, rank=rank, alpha=alpha, wrap_equivariant=True, wrap_dense=True)
    return model


def has_lora_layers(model: nn.Module) -> bool:
    return any(
        isinstance(
            module,
            (
                LoRAO3Linear,
                LoRADenseLinear,
                LoRAFCLayer,
            ),
        )
        for module in model.modules()
    )


def merge_lora_weights(model: nn.Module, inplace: bool = True) -> nn.Module:
    """
    Merge LoRA weights into base weights and replace LoRA wrappers with merged base modules.

    This eliminates the inference overhead from LoRA by folding the low-rank
    adaptations directly into the original weight matrices. After merging:
    - LoRADenseLinear -> nn.Linear (with merged weights)
    - LoRAFCLayer -> e3nn _Layer (with merged weights)
    - LoRAO3Linear -> o3.Linear (with merged weights)

    Args:
        model: Model containing LoRA layers to merge.
        inplace: If True, modifies the model in place. If False, works on a deep copy.

    Returns:
        Model with LoRA weights merged into base layers. All parameters will have
        requires_grad=True after merging.
    """
    if not inplace:
        import copy

        model = copy.deepcopy(model)

    def merge_recursive(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, (LoRADenseLinear, LoRAFCLayer, LoRAO3Linear)):
                setattr(module, name, child.merge_into_base())
            else:
                merge_recursive(child)

    merge_recursive(model)

    for param in model.parameters():
        param.requires_grad = True

    return model
