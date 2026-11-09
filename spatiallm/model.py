from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    architecture: str = "spatial"
    vocab_size: int = 257
    dim: int = 96
    layers: int = 2
    heads: int = 4
    max_seq_len: int = 256
    local_window: int = 8
    dropout: float = 0.0
    hybrid_every: int = 2
    ff_dim: int | None = None

    def __post_init__(self):
        if self.architecture not in {"spatial", "transformer", "hybrid"}:
            raise ValueError("architecture must be spatial, transformer or hybrid")
        if self.ff_dim is not None and (not isinstance(self.ff_dim, int) or self.ff_dim < 1):
            raise ValueError("ff_dim must be a positive integer or None")
        if self.architecture == "hybrid" and (
                not isinstance(self.hybrid_every, int) or not 2 <= self.hybrid_every <= self.layers):
            raise ValueError("hybrid_every must be between 2 and layers")
        if min(self.dim, self.layers, self.heads, self.max_seq_len, self.local_window) < 1:
            raise ValueError("model dimensions must be positive")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if not isinstance(self.vocab_size, int) or self.vocab_size < 2:
            raise ValueError("vocab_size must be an integer >= 2")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class SpatialAttention(nn.Module):
    """O(B*T*K*D) geometric mixing; never constructs a T by T score tensor.

    Fixed local + dyadic causal edges are reweighted by learned 3-D distances.
    Edge selection is deliberately static: global nearest-neighbour search would
    introduce quadratic work and complicate strict causal inference.
    """

    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.dim // cfg.heads
        offsets = sorted(set(range(min(cfg.local_window, cfg.max_seq_len))) |
                         {2 ** i for i in range(cfg.max_seq_len.bit_length())
                          if 2 ** i < cfg.max_seq_len})
        self.register_buffer("offsets", torch.tensor(offsets), persistent=False)
        positions = torch.arange(cfg.max_seq_len)[:, None]
        indices = positions - self.offsets[None, :]
        self.register_buffer("indices", indices.clamp_min(0), persistent=False)
        self.register_buffer("valid", indices >= 0, persistent=False)
        self.coordinates = nn.Linear(cfg.dim, 2 * cfg.heads * 3)
        self.values = nn.Linear(cfg.dim, cfg.dim)
        self.project = nn.Linear(cfg.dim, cfg.dim)
        self.log_scale = nn.Parameter(torch.zeros(cfg.heads))
        self.edge_bias = nn.Parameter(torch.zeros(cfg.heads, len(offsets)))
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        b, t, d = x.shape
        coords = self.coordinates(x).view(b, t, 2, self.heads, 3).tanh()
        q, k = coords[:, :, 0], coords[:, :, 1]
        idx = self.indices[:t]
        neighbours = k[:, idx]  # B,T,K,H,3
        distance = (q[:, :, None].float() - neighbours.float()).square().sum(-1)
        scale = self.log_scale.float().clamp(-5, 5).exp()
        scores = -distance * scale + self.edge_bias.float().T[None, None]
        scores = scores.masked_fill(~self.valid[:t][None, :, :, None], -torch.inf)
        weights = self.dropout(scores.softmax(dim=2)).to(x.dtype)
        v = self.values(x).view(b, t, self.heads, self.head_dim)
        mixed = (weights[..., None] * v[:, idx]).sum(dim=2).reshape(b, t, d)
        return self.project(mixed)


class TransformerAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads, self.head_dim = cfg.heads, cfg.dim // cfg.heads
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim)
        self.project = nn.Linear(cfg.dim, cfg.dim)
        self.dropout = cfg.dropout

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).unbind(2)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        return self.project(y.transpose(1, 2).contiguous().view(b, t, d))


class Block(nn.Module):
    def __init__(self, cfg, index=0):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.dim), nn.LayerNorm(cfg.dim)
        spatial = cfg.architecture == "spatial" or (
            cfg.architecture == "hybrid" and (index + 1) % cfg.hybrid_every != 0)
        self.mix = (SpatialAttention if spatial else TransformerAttention)(cfg)
        ff_dim = cfg.ff_dim or 4 * cfg.dim
        self.ff = nn.Sequential(nn.Linear(cfg.dim, ff_dim), nn.GELU(),
                                nn.Linear(ff_dim, cfg.dim), nn.Dropout(cfg.dropout))
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = x + self.drop(self.mix(self.norm1(x)))
        return x + self.ff(self.norm2(x))


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.position = nn.Embedding(cfg.max_seq_len, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.layers)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward_hidden(self, tokens):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.max_seq_len:
            raise ValueError("tokens must be [batch, time], 1 <= time <= max_seq_len")
        x = self.embedding(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)

    def forward(self, tokens, targets=None):
        logits = self.output(self.forward_hidden(tokens))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    def _chunk_loss(self, hidden, targets):
        return F.cross_entropy(self.output(hidden), targets, reduction="sum")

    def forward_loss(self, tokens, targets, chunk_size=512, recompute_head=True):
        """Exact CE with a bounded output matrix, including during backward.

        Non-reentrant checkpointing recomputes each output chunk during backward
        so softmax activations for all vocabulary logits are never retained.
        """
        if chunk_size < 1 or tokens.shape != targets.shape:
            raise ValueError("positive chunk_size and equally shaped tokens/targets required")
        hidden = self.forward_hidden(tokens).reshape(-1, self.config.dim)
        targets = targets.reshape(-1)
        losses = []
        for first in range(0, len(targets), chunk_size):
            h, y = hidden[first:first + chunk_size], targets[first:first + chunk_size]
            if recompute_head and torch.is_grad_enabled():
                losses.append(checkpoint(self._chunk_loss, h, y, use_reentrant=False))
            else:
                losses.append(self._chunk_loss(h, y))
        return torch.stack(losses).sum() / targets.numel()

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())

    def config_dict(self):
        return asdict(self.config)
