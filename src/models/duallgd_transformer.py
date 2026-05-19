import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src import utils
from src.models.transformer_model import XEyTransformerLayer


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FiLMConditioner(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(cond).unsqueeze(1)
        beta = self.beta(cond).unsqueeze(1)
        return (1.0 + gamma) * x + beta


def _iid_gaussian_random_matrix(
    num_features: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    return torch.randn(num_features, head_dim, device=device, dtype=dtype)


def _orthogonal_gaussian_random_matrix(
    num_features: int, head_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    num_full_blocks = num_features // head_dim
    blocks = []

    for _ in range(num_full_blocks):
        q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, device=device, dtype=dtype), mode="reduced")
        blocks.append(q.transpose(0, 1))

    remainder = num_features - num_full_blocks * head_dim
    if remainder > 0:
        q, _ = torch.linalg.qr(torch.randn(head_dim, head_dim, device=device, dtype=dtype), mode="reduced")
        blocks.append(q.transpose(0, 1)[:remainder])

    if len(blocks) == 0:
        raise ValueError("num_features must be >= 1")

    matrix = torch.cat(blocks, dim=0)
    scale = math.sqrt(num_full_blocks + remainder / head_dim) if (num_full_blocks > 0 or remainder > 0) else 1.0
    matrix = matrix / max(scale, 1e-8)
    return matrix


class PerformerFeatureMap(nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_features: int,
        kernel: str = "positive",
        feature_type: str = "orthogonal",
        redraw_on_forward: bool = False,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_features = max(1, int(num_features))
        self.kernel = kernel.lower()
        self.feature_type = feature_type.lower()
        self.redraw_on_forward = redraw_on_forward

        if self.kernel not in {"positive", "sincos"}:
            raise ValueError(f"Unsupported performer kernel: {self.kernel}")
        if self.feature_type not in {"orthogonal", "iid"}:
            raise ValueError(f"Unsupported performer feature_type: {self.feature_type}")

        self.register_buffer("random_features", torch.empty(0), persistent=False)

    def _build_random_features(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.feature_type == "orthogonal":
            return _orthogonal_gaussian_random_matrix(self.num_features, self.head_dim, device, dtype)
        return _iid_gaussian_random_matrix(self.num_features, self.head_dim, device, dtype)

    def _get_random_features(self, x: torch.Tensor) -> torch.Tensor:
        needs_init = (
            self.random_features.numel() == 0
            or self.random_features.device != x.device
            or self.random_features.dtype != x.dtype
            or self.redraw_on_forward
        )
        if needs_init:
            self.random_features = self._build_random_features(x.device, x.dtype)
        return self.random_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, N, D]
        normalizer = self.head_dim ** -0.25
        x = x * normalizer

        random_features = self._get_random_features(x)
        proj = torch.einsum("bhnd,md->bhnm", x, random_features)

        # Keep exponentials in numerically safe range.
        proj = proj.clamp(min=-20.0, max=20.0)
        x_norm_sq = x.square().sum(dim=-1, keepdim=True).clamp(max=30.0)

        if self.kernel == "positive":
            h = torch.exp(-0.5 * x_norm_sq)
            mapped = h * torch.exp(proj) / math.sqrt(self.num_features)
        else:
            h = torch.exp(0.5 * x_norm_sq)
            phase = 2.0 * math.pi * proj
            mapped = h / math.sqrt(self.num_features) * torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)

        return mapped


class PerformerSelfAttention(nn.Module):
    """Performer-style linear attention with configurable random features."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        performer_num_features: int,
        performer_kernel: str,
        performer_feature_type: str,
        performer_redraw_on_forward: bool,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = 1e-8

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.feature_map = PerformerFeatureMap(
            head_dim=self.head_dim,
            num_features=performer_num_features,
            kernel=performer_kernel,
            feature_type=performer_feature_type,
            redraw_on_forward=performer_redraw_on_forward,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, num_tokens, dim = x.shape

        q = self.q_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        token_mask_f = token_mask.unsqueeze(1).unsqueeze(-1).to(q.dtype)

        q = self.feature_map(q) * token_mask_f
        k = self.feature_map(k) * token_mask_f
        v = v * token_mask_f

        kv = torch.einsum("bhnm,bhnd->bhmd", k, v)
        k_sum = k.sum(dim=2)

        numerator = torch.einsum("bhnm,bhmd->bhnd", q, kv)
        denominator = torch.einsum("bhnm,bhm->bhn", q, k_sum).unsqueeze(-1)
        out = numerator / (denominator + self.eps)
        out = out.permute(0, 2, 1, 3).reshape(bsz, num_tokens, dim)

        out = self.out_proj(out)
        out = self.dropout(out)
        out = out * token_mask.unsqueeze(-1).to(out.dtype)
        return out


class ELUPlusOneSelfAttention(nn.Module):
    """Linear attention via ELU+1 feature map."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.eps = 1e-8

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x) + 1.0

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, dim = x.shape

        q = self.q_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        token_mask_f = token_mask.unsqueeze(1).unsqueeze(-1).to(q.dtype)
        q = self._phi(q) * token_mask_f
        k = self._phi(k) * token_mask_f
        v = v * token_mask_f

        kv = torch.einsum("bhnm,bhnd->bhmd", k, v)
        k_sum = k.sum(dim=2)

        numerator = torch.einsum("bhnm,bhmd->bhnd", q, kv)
        denominator = torch.einsum("bhnm,bhm->bhn", q, k_sum).unsqueeze(-1)
        out = numerator / (denominator + self.eps)
        out = out.permute(0, 2, 1, 3).reshape(bsz, num_tokens, dim)

        out = self.out_proj(out)
        out = self.dropout(out)
        out = out * token_mask.unsqueeze(-1).to(out.dtype)
        return out


class SoftmaxSelfAttention(nn.Module):
    """Standard dense self-attention with the same projections as Performer."""

    def __init__(self, dim: int, num_heads: int, dropout: float):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        bsz, num_tokens, dim = x.shape

        q = self.q_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).view(bsz, num_tokens, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale

        valid_pairs = token_mask.unsqueeze(1).unsqueeze(-1) & token_mask.unsqueeze(1).unsqueeze(-2)
        valid_rows = valid_pairs.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~valid_pairs, torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(~valid_rows, 0.0)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn = attn * valid_rows.to(attn.dtype)

        v = v * token_mask.unsqueeze(1).unsqueeze(-1).to(v.dtype)
        out = torch.einsum("bhij,bhjd->bhid", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(bsz, num_tokens, dim)

        out = self.out_proj(out)
        out = self.dropout(out)
        out = out * token_mask.unsqueeze(-1).to(out.dtype)
        return out


class DualAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        global_dim: int,
        num_heads: int,
        dropout: float,
        performer_num_features: int,
        performer_kernel: str,
        performer_feature_type: str,
        performer_redraw_on_forward: bool,
        attention_type: str,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        attention_type = attention_type.lower()
        if attention_type == "performer":
            self.attn = PerformerSelfAttention(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                performer_num_features=performer_num_features,
                performer_kernel=performer_kernel,
                performer_feature_type=performer_feature_type,
                performer_redraw_on_forward=performer_redraw_on_forward,
            )
        elif attention_type == "elu_plus_one":
            self.attn = ELUPlusOneSelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        elif attention_type == "softmax":
            self.attn = SoftmaxSelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        else:
            raise ValueError(
                f"Unsupported attention_type: {attention_type}. "
                "Supported: 'performer', 'elu_plus_one', 'softmax'."
            )
        self.ffn = FeedForward(dim, dim * 4, dropout)
        self.global_proj = nn.Linear(dim, global_dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.norm1(x), token_mask)
        x = x + self.ffn(self.norm2(x))
        x = x * token_mask.unsqueeze(-1).to(x.dtype)

        denom = token_mask.sum(dim=1, keepdim=True).clamp_min(1).to(x.dtype)
        pooled = (x * token_mask.unsqueeze(-1).to(x.dtype)).sum(dim=1) / denom
        y = y + self.global_proj(pooled)
        return x, y


class MaskedCrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if q_dim % num_heads != 0:
            raise ValueError(f"q_dim ({q_dim}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out_proj = nn.Linear(q_dim, q_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q_x: torch.Tensor,
        kv_x: torch.Tensor,
        q_mask: torch.Tensor,
        kv_mask: torch.Tensor,
        local_mask: torch.Tensor,
        return_attn: bool = False,
    ):
        bsz, q_tokens, q_dim = q_x.shape
        kv_tokens = kv_x.size(1)

        q = self.q_proj(q_x).view(bsz, q_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(kv_x).view(bsz, kv_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(kv_x).view(bsz, kv_tokens, self.num_heads, self.head_dim)
        kv_mask_f = kv_mask.unsqueeze(-1).unsqueeze(-1).to(k.dtype)
        k = k * kv_mask_f
        v = v * kv_mask_f

        scores = torch.einsum("bihd,bjhd->bhij", q, k) * self.scale

        valid_pairs = q_mask.unsqueeze(-1) & kv_mask.unsqueeze(-2) & local_mask
        valid_rows = valid_pairs.any(dim=-1, keepdim=True)

        scores = scores.masked_fill(~valid_pairs.unsqueeze(1), torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(~valid_rows.unsqueeze(1), 0.0)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        attn = attn * valid_rows.unsqueeze(1).to(attn.dtype)

        out = torch.einsum("bhij,bjhd->bihd", attn, v).reshape(bsz, q_tokens, q_dim)
        out = self.out_proj(out)
        out = out * q_mask.unsqueeze(-1).to(out.dtype)
        if return_attn:
            return out, attn
        return out


class DualLGD(nn.Module):
    """
    Dual-stream architecture built on top of DiffMS' native transformer block
    (`XEyTransformerLayer`) for both primal and dual streams.
    """

    def __init__(
        self,
        n_layers: int,
        input_dims: Dict[str, int],
        hidden_dims: Dict[str, int],
        hidden_mlp_dims: Dict[str, int],
        output_dims: Dict[str, int],
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        act_fn_in: nn.Module = nn.ReLU(),
        act_fn_out: nn.Module = nn.ReLU(),
        duallgd: Optional[Dict] = None,
        **kwargs,
    ):
        super().__init__()
        cfg = duallgd or {}

        self.n_layers = n_layers
        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        self.hidden_primal = hidden_dims["dx"]
        self.hidden_dual = hidden_dims.get("de", hidden_dims["dx"])
        self.hidden_global = hidden_dims["dy"]

        self.use_primal_stream = bool(cfg.get("use_primal_stream", True))
        self.use_dual_stream = bool(cfg.get("use_dual_stream", True))
        self.use_dual_graph = self.use_dual_stream
        # Cross-attention requires an instantiated dual graph. Disable it when the
        # caller explicitly ablates the full dual path.
        self.use_cross_attention = bool(cfg.get("use_cross_attention", True)) and self.use_dual_graph
        self.use_dual_linegraph_mask = bool(cfg.get("use_dual_linegraph_mask", True))
        self.use_linear_dual_attention = bool(cfg.get("use_linear_dual_attention", False))
        self.use_attention_only_dual_block = bool(cfg.get("use_attention_only_dual_block", False))
        self.linear_attention_type = str(cfg.get("linear_attention_type", "performer")).lower()
        self.dual_attention_type = str(cfg.get("dual_attention_type", self.linear_attention_type)).lower()
        requested_cross_attention = bool(cfg.get("use_cross_attention", True))
        self.cross_attention_mode = str(cfg.get("cross_attention_mode", "bidirectional")).lower()
        self.cross_attention_use_incidence_mask = bool(cfg.get("cross_attention_use_incidence_mask", True))
        self.edge_readout_source = str(cfg.get("edge_readout_source", "dual")).lower()
        self.performer_kernel = str(cfg.get("performer_kernel", "positive")).lower()
        self.performer_feature_type = str(cfg.get("performer_feature_type", "orthogonal")).lower()
        self.performer_num_features = int(cfg.get("performer_num_features", 128))
        self.performer_redraw_on_forward = bool(cfg.get("performer_redraw_on_forward", False))
        self.dynamic_dual_edge_from_primal = bool(cfg.get("dynamic_dual_edge_from_primal", True))
        self.max_dense_dual_nodes = int(cfg.get("max_dense_dual_nodes", 4096))
        self.edge_diffusion_only = bool(cfg.get("edge_diffusion_only", True))

        if self.cross_attention_mode not in {"bidirectional", "dual_to_primal", "primal_to_dual", "none"}:
            raise ValueError(
                f"Unsupported cross_attention_mode: {self.cross_attention_mode}. "
                "Supported: 'bidirectional', 'dual_to_primal', 'primal_to_dual', 'none'."
            )
        if self.edge_readout_source not in {"dual", "primal"}:
            raise ValueError(
                f"Unsupported edge_readout_source: {self.edge_readout_source}. "
                "Supported: 'dual', 'primal'."
            )
        if not self.use_dual_graph:
            self.edge_readout_source = "primal"
        self.use_cross_attention = (
            requested_cross_attention
            and self.cross_attention_mode != "none"
            and self.use_dual_graph
        )
        self.enable_p_from_d = self.use_cross_attention and self.cross_attention_mode in {"bidirectional", "dual_to_primal"}
        self.enable_d_from_p = self.use_cross_attention and self.cross_attention_mode in {"bidirectional", "primal_to_dual"}
        self.use_stable_cross_updates = not self.cross_attention_use_incidence_mask

        self.max_spd = int(cfg.get("max_spd", 8))
        self.max_centrality = int(cfg.get("max_centrality", 16))

        primal_heads = int(cfg.get("primal_heads", hidden_dims.get("n_head", num_heads)))
        dual_heads = int(cfg.get("dual_heads", hidden_dims.get("n_head", num_heads)))
        cross_heads = int(cfg.get("cross_heads", hidden_dims.get("n_head", num_heads)))

        primal_ffx = int(hidden_dims.get("dim_ffX", self.hidden_primal * 4))
        primal_ffe = int(hidden_dims.get("dim_ffE", self.hidden_dual * 2))
        primal_ffy = int(hidden_dims.get("dim_ffy", self.hidden_global * 2))

        dual_ffx = int(cfg.get("dual_dim_ffX", self.hidden_dual * 4))
        dual_ffe = int(cfg.get("dual_dim_ffE", self.hidden_dual * 2))
        dual_ffy = int(cfg.get("dual_dim_ffy", self.hidden_global * 2))

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], self.hidden_primal),
            act_fn_in,
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], self.hidden_dual),
            act_fn_in,
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], self.hidden_global),
            act_fn_in,
        )

        if self.use_dual_graph:
            self.dual_node_init = nn.Sequential(
                nn.Linear(self.hidden_dual + 2 * self.hidden_primal, self.hidden_dual),
                nn.GELU(),
                nn.Linear(self.hidden_dual, self.hidden_dual),
            )
        else:
            self.dual_node_init = None

        # Dense dual-edge tensors are only needed when Dual stream uses dense XEyTransformer layers.
        self._use_dense_dual_edges = self.use_dual_graph and (
            not self.use_linear_dual_attention and not self.use_attention_only_dual_block
        )
        if self._use_dense_dual_edges:
            # Dual-edge features are built from the shared primal atom (center atom)
            # when two dual nodes correspond to two primal edges sharing one endpoint.
            self.shared_atom_to_dual_edge = nn.Sequential(
                nn.Linear(self.hidden_primal, self.hidden_dual),
                nn.GELU(),
                nn.Linear(self.hidden_dual, self.hidden_dual),
            )
            # Fallback feature for non-shared pairs (used only when full dual connectivity is enabled).
            self.dual_non_shared_edge = nn.Parameter(torch.zeros(self.hidden_dual))
        else:
            self.shared_atom_to_dual_edge = None
            self.register_parameter("dual_non_shared_edge", None)

        self.primal_struct_proj = nn.Linear(2, self.hidden_dual)
        self.film_primal = FiLMConditioner(self.hidden_primal, self.hidden_global)
        self.film_dual = FiLMConditioner(self.hidden_dual, self.hidden_global) if self.use_dual_graph else None

        if self.use_primal_stream:
            self.primal_layers = nn.ModuleList(
                [
                    XEyTransformerLayer(
                        dx=self.hidden_primal,
                        de=self.hidden_dual,
                        dy=self.hidden_global,
                        n_head=primal_heads,
                        dim_ffX=primal_ffx,
                        dim_ffE=primal_ffe,
                        dim_ffy=primal_ffy,
                        dropout=dropout,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.primal_layers = nn.ModuleList([])

        if self.use_dual_graph:
            if self.use_linear_dual_attention or self.use_attention_only_dual_block:
                dual_attention_type = (
                    self.dual_attention_type if self.use_attention_only_dual_block else self.linear_attention_type
                )
                self.dual_layers = nn.ModuleList(
                    [
                        DualAttentionBlock(
                            dim=self.hidden_dual,
                            global_dim=self.hidden_global,
                            num_heads=dual_heads,
                            dropout=attention_dropout,
                            performer_num_features=self.performer_num_features,
                            performer_kernel=self.performer_kernel,
                            performer_feature_type=self.performer_feature_type,
                            performer_redraw_on_forward=self.performer_redraw_on_forward,
                            attention_type=dual_attention_type,
                        )
                        for _ in range(n_layers)
                    ]
                )
            else:
                self.dual_layers = nn.ModuleList(
                    [
                        XEyTransformerLayer(
                            dx=self.hidden_dual,
                            de=self.hidden_dual,
                            dy=self.hidden_global,
                            n_head=dual_heads,
                            dim_ffX=dual_ffx,
                            dim_ffE=dual_ffe,
                            dim_ffy=dual_ffy,
                            dropout=dropout,
                        )
                        for _ in range(n_layers)
                    ]
                )
        else:
            self.dual_layers = nn.ModuleList([])

        if self.enable_p_from_d:
            self.p_from_d = nn.ModuleList(
                [MaskedCrossAttention(self.hidden_primal, self.hidden_dual, cross_heads, attention_dropout) for _ in range(n_layers)]
            )
        else:
            self.p_from_d = nn.ModuleList([])
        if self.enable_d_from_p:
            self.d_from_p = nn.ModuleList(
                [MaskedCrossAttention(self.hidden_dual, self.hidden_primal, cross_heads, attention_dropout) for _ in range(n_layers)]
            )
        else:
            self.d_from_p = nn.ModuleList([])

        self.global_fuse = nn.ModuleList(
            [nn.Linear(2 * self.hidden_global + self.hidden_primal + self.hidden_dual, self.hidden_global) for _ in range(n_layers)]
        )
        self.global_norm = nn.ModuleList([nn.LayerNorm(self.hidden_global) for _ in range(n_layers)])

        if self.edge_diffusion_only:
            self.mlp_out_X = None
        else:
            self.mlp_out_X = nn.Sequential(
                nn.Linear(self.hidden_primal, hidden_mlp_dims["X"]),
                act_fn_out,
                nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
            )

        # This decoder is shared by the regular dual-token readout and the
        # no-dual-graph ablation path, where primal edge states are decoded directly.
        self.dual_out_E = nn.Sequential(
            nn.Linear(self.hidden_dual, hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )

        self.mlp_out_y = nn.Sequential(
            nn.Linear(self.hidden_global, hidden_mlp_dims["y"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
        )

        self._pair_index_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._pair_dense_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _get_pair_indices(self, num_nodes: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_nodes not in self._pair_index_cache:
            idx_i, idx_j = torch.triu_indices(num_nodes, num_nodes, offset=1)
            self._pair_index_cache[num_nodes] = (idx_i, idx_j)
        idx_i, idx_j = self._pair_index_cache[num_nodes]
        return idx_i.to(device), idx_j.to(device)

    def _get_dense_pair_cache(
        self, num_nodes: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if num_nodes not in self._pair_dense_cache:
            idx_i, idx_j = self._get_pair_indices(num_nodes, torch.device("cpu"))
            same_ii = idx_i[:, None] == idx_i[None, :]
            same_ij = idx_i[:, None] == idx_j[None, :]
            same_ji = idx_j[:, None] == idx_i[None, :]
            same_jj = idx_j[:, None] == idx_j[None, :]

            line_adj = same_ii | same_ij | same_ji | same_jj

            shared_atom = torch.full_like(same_ii, fill_value=-1, dtype=torch.long)
            shared_atom = torch.where(same_ii, idx_i[:, None], shared_atom)
            shared_atom = torch.where(same_ij, idx_i[:, None], shared_atom)
            shared_atom = torch.where(same_ji, idx_j[:, None], shared_atom)
            shared_atom = torch.where(same_jj, idx_j[:, None], shared_atom)

            self._pair_dense_cache[num_nodes] = (line_adj, shared_atom)

        line_adj, shared_atom = self._pair_dense_cache[num_nodes]
        return line_adj.to(device), shared_atom.to(device)

    def _build_dual_edge_features(
        self,
        h_primal: torch.Tensor,
        shared_atom_idx: torch.Tensor,
        dual_pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build dual edge features from the shared primal atom feature.
        shared_atom_idx: [M, M], -1 for non-shared dual-node pairs.
        """
        if self.shared_atom_to_dual_edge is None or self.dual_non_shared_edge is None:
            raise RuntimeError("Dense dual-edge feature builder is not initialized for current config.")
        safe_idx = shared_atom_idx.clamp_min(0)
        shared_feat = h_primal[:, safe_idx, :]  # [B, M, M, hidden_primal]

        shared_valid = (shared_atom_idx >= 0).unsqueeze(0)
        shared_feat = shared_feat * shared_valid.unsqueeze(-1).to(shared_feat.dtype)

        dual_E = self.shared_atom_to_dual_edge(shared_feat)

        non_shared_mask = dual_pair_mask & (~shared_valid)
        if non_shared_mask.any():
            dual_E = dual_E + non_shared_mask.unsqueeze(-1).to(dual_E.dtype) * self.dual_non_shared_edge.view(
                1, 1, 1, -1
            )

        dual_E = dual_E * dual_pair_mask.unsqueeze(-1).to(dual_E.dtype)
        return dual_E

    def _compute_primal_structural_features(self, E_input: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        feats = []
        edge_classes = E_input[..., : self.out_dim_E].argmax(dim=-1)
        adj = (edge_classes != 0) & node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)

        bsz, num_nodes, _ = adj.shape
        diag = torch.eye(num_nodes, device=adj.device, dtype=torch.bool).unsqueeze(0)
        adj = adj & (~diag)

        inf = self.max_spd + 1
        dist = torch.full((bsz, num_nodes, num_nodes), inf, device=adj.device, dtype=torch.long)
        dist[adj] = 1
        dist[diag.expand(bsz, -1, -1)] = 0

        for k in range(num_nodes):
            dist = torch.minimum(dist, dist[:, :, k].unsqueeze(-1) + dist[:, k, :].unsqueeze(1))

        dist = dist.clamp_max(inf).float() / float(inf)
        feats.append(dist.unsqueeze(-1))

        degree = adj.long().sum(dim=-1).clamp_max(self.max_centrality).float() / float(max(self.max_centrality, 1))
        cent = degree.unsqueeze(2) + degree.unsqueeze(1)
        feats.append(cent.unsqueeze(-1))

        struct_feat = torch.cat(feats, dim=-1)
        valid_pairs = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        struct_feat = struct_feat * valid_pairs.unsqueeze(-1).to(struct_feat.dtype)
        return struct_feat

    def _dual_masks_and_incidence(
        self,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        node_mask: torch.Tensor,
        line_adj: Optional[torch.Tensor] = None,
        build_pair_mask: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        bsz, num_nodes = node_mask.shape
        num_pairs = idx_i.numel()

        dual_mask = node_mask[:, idx_i] & node_mask[:, idx_j]

        incidence = torch.zeros(
            bsz,
            num_nodes,
            num_pairs,
            device=node_mask.device,
            dtype=torch.bool,
        )
        pair_ids = torch.arange(num_pairs, device=node_mask.device)
        incidence[:, idx_i, pair_ids] = True
        incidence[:, idx_j, pair_ids] = True
        incidence = incidence & node_mask.unsqueeze(-1) & dual_mask.unsqueeze(1)

        dual_pair_mask = None
        if build_pair_mask:
            if self.use_dual_linegraph_mask and line_adj is not None:
                dual_pair_mask = dual_mask.unsqueeze(-1) & dual_mask.unsqueeze(-2) & line_adj.unsqueeze(0)
            else:
                dual_pair_mask = dual_mask.unsqueeze(-1) & dual_mask.unsqueeze(-2)

        return dual_mask, dual_pair_mask, incidence

    def _build_cross_attention_masks(
        self,
        node_mask: torch.Tensor,
        dual_mask: torch.Tensor,
        incidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cross_attention_use_incidence_mask:
            return incidence, incidence.transpose(1, 2)

        p_from_d_mask = node_mask.unsqueeze(-1) & dual_mask.unsqueeze(-2)
        d_from_p_mask = dual_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        return p_from_d_mask, d_from_p_mask

    @staticmethod
    def _normalize_cross_input(x: torch.Tensor) -> torch.Tensor:
        # Parameter-free normalization to keep the ablation parameter count unchanged.
        return F.layer_norm(x, (x.size(-1),))

    @staticmethod
    def _cross_residual_scale(
        current_mask: torch.Tensor,
        reference_mask: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        current_fanin = current_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
        reference_fanin = reference_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(dtype)
        ratio = (reference_fanin / current_fanin).clamp(max=1.0)
        return torch.sqrt(ratio)

    def _scatter_dual_to_dense(
        self,
        dual_logits: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
        num_nodes: int,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, _, edge_dim = dual_logits.shape
        out = torch.zeros(bsz, num_nodes, num_nodes, edge_dim, device=dual_logits.device, dtype=dual_logits.dtype)

        out[:, idx_i, idx_j, :] = dual_logits
        out[:, idx_j, idx_i, :] = dual_logits

        valid_pairs = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        out = out * valid_pairs.unsqueeze(-1).to(out.dtype)

        diag = torch.eye(num_nodes, device=out.device, dtype=torch.bool).unsqueeze(0)
        out = out.masked_fill(diag.unsqueeze(-1), 0.0)
        return out

    def _decode_primal_edges_to_dense(
        self,
        e_primal: torch.Tensor,
        E_to_out: torch.Tensor,
        node_mask: torch.Tensor,
    ) -> torch.Tensor:
        edge_logits = self.dual_out_E(e_primal)
        valid_pairs = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        edge_mask = valid_pairs.unsqueeze(-1).to(edge_logits.dtype)

        E_out = (edge_logits + E_to_out) * edge_mask
        diag = torch.eye(E_out.size(1), device=E_out.device, dtype=torch.bool).unsqueeze(0)
        E_out = E_out.masked_fill(diag.unsqueeze(-1), 0.0)
        E_out = 0.5 * (E_out + E_out.transpose(1, 2))
        return E_out

    @staticmethod
    def _normalize_analysis_layers(layer_ids) -> set[int]:
        if layer_ids is None:
            return set()
        if isinstance(layer_ids, int):
            return {int(layer_ids)}
        return {int(layer_id) for layer_id in layer_ids}

    def _edge_probe_logits(
        self,
        h_dual: Optional[torch.Tensor],
        e_primal: torch.Tensor,
        idx_i: Optional[torch.Tensor],
        idx_j: Optional[torch.Tensor],
    ) -> torch.Tensor:
        edge_repr = self._edge_probe_repr(h_dual=h_dual, e_primal=e_primal, idx_i=idx_i, idx_j=idx_j)
        return self.dual_out_E(edge_repr)

    def _edge_probe_repr(
        self,
        h_dual: Optional[torch.Tensor],
        e_primal: torch.Tensor,
        idx_i: Optional[torch.Tensor],
        idx_j: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.use_dual_graph and self.edge_readout_source == "dual":
            if h_dual is None:
                raise RuntimeError("Dual edge probe requested without dual states.")
            return h_dual

        if idx_i is None or idx_j is None:
            raise RuntimeError("Primal edge probe requested without pair indices.")
        return e_primal[:, idx_i, idx_j, :]

    def edge_target_repr_from_classes(
        self,
        pair_classes: torch.Tensor,
        X: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build an external target representation for each pair from the
        ground-truth bond class, without referencing any later-layer states.

        For primal edge readout this is simply the embedded bond-type one-hot.
        For dual-token readout, map that embedded bond-type together with the
        current endpoint atom embeddings through the same dual-node initializer
        used to create dual tokens.
        """
        pair_onehot = F.one_hot(pair_classes.clamp_min(0), num_classes=self.out_dim_E).to(X.dtype)
        edge_target = self.mlp_in_E(pair_onehot)

        if self.use_dual_graph and self.edge_readout_source == "dual":
            if self.dual_node_init is None:
                raise RuntimeError("Dual target representation requested without dual_node_init.")
            h_primal = self.mlp_in_X(X)
            xi = h_primal[:, idx_i, :]
            xj = h_primal[:, idx_j, :]
            edge_target = self.dual_node_init(torch.cat([edge_target, xi, xj], dim=-1))

        return edge_target

    def forward(
        self,
        X: torch.Tensor,
        E: torch.Tensor,
        y: torch.Tensor,
        node_mask: torch.Tensor,
        analysis: Optional[Dict] = None,
    ):
        _, num_nodes = X.shape[:2]

        X_to_out = X[..., : self.out_dim_X]
        E_to_out = E[..., : self.out_dim_E]
        y_to_out = y[..., : self.out_dim_y]

        h_primal = self.mlp_in_X(X)
        e_primal = self.mlp_in_E(E)
        e_primal = 0.5 * (e_primal + e_primal.transpose(1, 2))
        y_state = self.mlp_in_y(y)

        struct_feat = self._compute_primal_structural_features(E, node_mask)
        e_primal = e_primal + self.primal_struct_proj(struct_feat)

        dual_mask: Optional[torch.Tensor]
        dual_pair_mask: Optional[torch.Tensor]
        shared_atom_idx: Optional[torch.Tensor]
        dual_E: Optional[torch.Tensor]
        incidence: Optional[torch.Tensor]
        h_dual: Optional[torch.Tensor]
        idx_i: Optional[torch.Tensor]
        idx_j: Optional[torch.Tensor]
        probe_layers = self._normalize_analysis_layers(None if analysis is None else analysis.get("probe_layers"))
        p_from_d_attn_layers = self._normalize_analysis_layers(
            None if analysis is None else analysis.get("p_from_d_attn_layers")
        )
        d_from_p_attn_layers = self._normalize_analysis_layers(
            None if analysis is None else analysis.get("d_from_p_attn_layers")
        )
        analysis_records = None if analysis is None else analysis.setdefault("records", {})

        if self.use_dual_graph:
            idx_i, idx_j = self._get_pair_indices(num_nodes, X.device)
            num_pairs = idx_i.numel()
            line_adj: Optional[torch.Tensor] = None
            cached_shared_atom_idx: Optional[torch.Tensor] = None
            if self._use_dense_dual_edges:
                line_adj, cached_shared_atom_idx = self._get_dense_pair_cache(num_nodes, X.device)

            primal_edge_pairs = e_primal[:, idx_i, idx_j, :]
            xi = h_primal[:, idx_i, :]
            xj = h_primal[:, idx_j, :]
            assert self.dual_node_init is not None
            h_dual = self.dual_node_init(torch.cat([primal_edge_pairs, xi, xj], dim=-1))

            if not self._use_dense_dual_edges:
                shared_atom_idx = None
                dual_mask, dual_pair_mask, incidence = self._dual_masks_and_incidence(
                    idx_i=idx_i,
                    idx_j=idx_j,
                    node_mask=node_mask,
                    line_adj=None,
                    build_pair_mask=False,
                )
                dual_E = None
            else:
                if num_pairs > self.max_dense_dual_nodes:
                    raise RuntimeError(
                        f"Dual dense edge path disabled for M={num_pairs} dual nodes (max_dense_dual_nodes="
                        f"{self.max_dense_dual_nodes}). Set model.duallgd.use_linear_dual_attention=true "
                        "or increase max_dense_dual_nodes if you intentionally want dense MxM dual edges."
                    )
                assert line_adj is not None
                assert cached_shared_atom_idx is not None
                shared_atom_idx = cached_shared_atom_idx
                dual_mask, dual_pair_mask, incidence = self._dual_masks_and_incidence(
                    idx_i=idx_i,
                    idx_j=idx_j,
                    node_mask=node_mask,
                    line_adj=line_adj,
                    build_pair_mask=True,
                )
                assert dual_pair_mask is not None
                dual_E = self._build_dual_edge_features(
                    h_primal=h_primal,
                    shared_atom_idx=shared_atom_idx,
                    dual_pair_mask=dual_pair_mask,
                )
        else:
            idx_i = None
            idx_j = None
            incidence = None
            h_dual = None
            dual_mask = None
            dual_pair_mask = None
            shared_atom_idx = None
            dual_E = None

        if analysis is not None:
            analysis["edge_probe_source"] = "dual" if (self.use_dual_graph and self.edge_readout_source == "dual") else "primal"
            if idx_i is not None and idx_j is not None:
                analysis["pair_index"] = torch.stack((idx_i, idx_j), dim=-1)

        h_primal = h_primal * node_mask.unsqueeze(-1).to(h_primal.dtype)
        e_primal = e_primal * (node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)).unsqueeze(-1).to(e_primal.dtype)
        if h_dual is not None and dual_mask is not None:
            h_dual = h_dual * dual_mask.unsqueeze(-1).to(h_dual.dtype)
        if dual_E is not None and dual_pair_mask is not None:
            dual_E = dual_E * dual_pair_mask.unsqueeze(-1).to(dual_E.dtype)

        dual_tokens = h_dual.size(1) if h_dual is not None else 0

        for layer_idx in range(self.n_layers):
            capture_probe = layer_idx in probe_layers
            capture_p_from_d_attn = layer_idx in p_from_d_attn_layers
            capture_d_from_p_attn = layer_idx in d_from_p_attn_layers
            layer_record = None
            if analysis_records is not None and (capture_probe or capture_p_from_d_attn or capture_d_from_p_attn):
                layer_record = analysis_records.setdefault(int(layer_idx), {})
                layer_record["layer_idx"] = int(layer_idx)

            if self.use_primal_stream:
                h_primal = self.film_primal(h_primal, y_state)
            if self.use_dual_graph and dual_tokens > 0:
                assert self.film_dual is not None
                assert h_dual is not None
                h_dual = self.film_dual(h_dual, y_state)

            y_primal = y_state
            if self.use_dual_graph:
                y_dual = y_state
            else:
                y_dual = y_state.new_zeros(y_state.shape)

            if self.use_primal_stream:
                h_primal, e_primal, y_primal = self.primal_layers[layer_idx](
                    h_primal, e_primal, y_state, node_mask
                )

            if self.use_dual_graph and dual_tokens > 0:
                assert h_dual is not None
                assert dual_mask is not None
                if self.use_linear_dual_attention or self.use_attention_only_dual_block:
                    h_dual, y_dual = self.dual_layers[layer_idx](
                        h_dual,
                        y_state,
                        dual_mask,
                    )
                else:
                    assert dual_E is not None
                    assert dual_pair_mask is not None
                    assert shared_atom_idx is not None
                    if self.dynamic_dual_edge_from_primal:
                        shared_dual_E = self._build_dual_edge_features(
                            h_primal=h_primal,
                            shared_atom_idx=shared_atom_idx,
                            dual_pair_mask=dual_pair_mask,
                        )
                        dual_E = 0.5 * (dual_E + shared_dual_E)
                    h_dual, dual_E, y_dual = self.dual_layers[layer_idx](h_dual, dual_E, y_state, dual_mask)
                    dual_E = dual_E * dual_pair_mask.unsqueeze(-1).to(dual_E.dtype)

            if capture_probe and layer_record is not None:
                layer_record["pre_cross_edge_repr"] = self._edge_probe_repr(
                    h_dual=h_dual,
                    e_primal=e_primal,
                    idx_i=idx_i,
                    idx_j=idx_j,
                ).clone()
                layer_record["pre_cross_edge_logits"] = self._edge_probe_logits(
                    h_dual=h_dual,
                    e_primal=e_primal,
                    idx_i=idx_i,
                    idx_j=idx_j,
                )

            if (self.enable_p_from_d or self.enable_d_from_p) and dual_tokens > 0:
                assert h_dual is not None
                assert dual_mask is not None
                assert incidence is not None
                p_from_d_mask, d_from_p_mask = self._build_cross_attention_masks(
                    node_mask=node_mask,
                    dual_mask=dual_mask,
                    incidence=incidence,
                )
                if self.use_stable_cross_updates:
                    h_primal_src = self._normalize_cross_input(h_primal)
                    h_dual_src = self._normalize_cross_input(h_dual)
                    primal_delta = None
                    dual_delta = None

                    if self.enable_p_from_d:
                        primal_update = self.p_from_d[layer_idx](
                            q_x=h_primal_src,
                            kv_x=h_dual_src,
                            q_mask=node_mask,
                            kv_mask=dual_mask,
                            local_mask=p_from_d_mask,
                            return_attn=capture_p_from_d_attn,
                        )
                        if capture_p_from_d_attn:
                            primal_delta, p_from_d_attn = primal_update
                            if layer_record is not None:
                                layer_record["p_from_d_attn"] = p_from_d_attn
                        else:
                            primal_delta = primal_update
                        if not self.cross_attention_use_incidence_mask:
                            primal_delta = primal_delta * self._cross_residual_scale(
                                current_mask=p_from_d_mask,
                                reference_mask=incidence,
                                dtype=primal_delta.dtype,
                            )

                    if self.enable_d_from_p:
                        dual_update = self.d_from_p[layer_idx](
                            q_x=h_dual_src,
                            kv_x=h_primal_src,
                            q_mask=dual_mask,
                            kv_mask=node_mask,
                            local_mask=d_from_p_mask,
                            return_attn=capture_d_from_p_attn,
                        )
                        if capture_d_from_p_attn:
                            dual_delta, d_from_p_attn = dual_update
                            if layer_record is not None:
                                layer_record["d_from_p_attn"] = d_from_p_attn
                        else:
                            dual_delta = dual_update
                        if not self.cross_attention_use_incidence_mask:
                            dual_delta = dual_delta * self._cross_residual_scale(
                                current_mask=d_from_p_mask,
                                reference_mask=incidence.transpose(1, 2),
                                dtype=dual_delta.dtype,
                            )

                    if primal_delta is not None:
                        h_primal = h_primal + primal_delta
                    if dual_delta is not None:
                        h_dual = h_dual + dual_delta
                else:
                    if self.enable_p_from_d:
                        primal_update = self.p_from_d[layer_idx](
                            q_x=h_primal,
                            kv_x=h_dual,
                            q_mask=node_mask,
                            kv_mask=dual_mask,
                            local_mask=p_from_d_mask,
                            return_attn=capture_p_from_d_attn,
                        )
                        if capture_p_from_d_attn:
                            primal_delta, p_from_d_attn = primal_update
                            if layer_record is not None:
                                layer_record["p_from_d_attn"] = p_from_d_attn
                        else:
                            primal_delta = primal_update
                        h_primal = h_primal + primal_delta

                    if self.enable_d_from_p:
                        dual_update = self.d_from_p[layer_idx](
                            q_x=h_dual,
                            kv_x=h_primal,
                            q_mask=dual_mask,
                            kv_mask=node_mask,
                            local_mask=d_from_p_mask,
                            return_attn=capture_d_from_p_attn,
                        )
                        if capture_d_from_p_attn:
                            dual_delta, d_from_p_attn = dual_update
                            if layer_record is not None:
                                layer_record["d_from_p_attn"] = d_from_p_attn
                        else:
                            dual_delta = dual_update
                        h_dual = h_dual + dual_delta

                h_primal = h_primal * node_mask.unsqueeze(-1).to(h_primal.dtype)
                h_dual = h_dual * dual_mask.unsqueeze(-1).to(h_dual.dtype)

            if capture_probe and layer_record is not None:
                layer_record["post_cross_edge_repr"] = self._edge_probe_repr(
                    h_dual=h_dual,
                    e_primal=e_primal,
                    idx_i=idx_i,
                    idx_j=idx_j,
                ).clone()
                layer_record["post_cross_edge_logits"] = self._edge_probe_logits(
                    h_dual=h_dual,
                    e_primal=e_primal,
                    idx_i=idx_i,
                    idx_j=idx_j,
                )

            primal_denom = node_mask.sum(dim=1, keepdim=True).clamp_min(1).to(h_primal.dtype)
            pooled_primal = (h_primal * node_mask.unsqueeze(-1).to(h_primal.dtype)).sum(dim=1) / primal_denom

            if dual_tokens > 0:
                assert h_dual is not None
                assert dual_mask is not None
                dual_denom = dual_mask.sum(dim=1, keepdim=True).clamp_min(1).to(h_dual.dtype)
                pooled_dual = (h_dual * dual_mask.unsqueeze(-1).to(h_dual.dtype)).sum(dim=1) / dual_denom
            else:
                pooled_dual = h_primal.new_zeros(h_primal.size(0), self.hidden_dual)

            fusion_in = torch.cat([y_primal, y_dual, pooled_primal, pooled_dual], dim=-1)
            y_state = self.global_norm[layer_idx](y_state + self.global_fuse[layer_idx](fusion_in))

        if self.edge_diffusion_only:
            X_out = X_to_out
        else:
            assert self.mlp_out_X is not None
            X_out = self.mlp_out_X(h_primal) + X_to_out

        if self.edge_readout_source == "dual" and self.use_dual_graph:
            assert h_dual is not None
            assert idx_i is not None
            assert idx_j is not None
            dual_logits = self.dual_out_E(h_dual)
            E_out = self._scatter_dual_to_dense(
                dual_logits=dual_logits,
                idx_i=idx_i,
                idx_j=idx_j,
                num_nodes=num_nodes,
                node_mask=node_mask,
            )
            E_out = E_out + E_to_out
            E_out = 0.5 * (E_out + E_out.transpose(1, 2))
        else:
            E_out = self._decode_primal_edges_to_dense(
                e_primal=e_primal,
                E_to_out=E_to_out,
                node_mask=node_mask,
            )

        y_out = self.mlp_out_y(y_state) + y_to_out

        return utils.PlaceHolder(X=X_out, E=E_out, y=y_out).mask(node_mask)
