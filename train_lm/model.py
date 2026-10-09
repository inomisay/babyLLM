import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BabyLMConfig:
    vocab_size: int = 8192      # Proportional vocabulary for tiny LLM (prevents embedding bloat)
    d_model: int = 288          # Hidden dimension
    n_layers: int = 6           # Number of Transformer layers
    n_heads: int = 6            # Attention heads (d_head = 48)
    d_ff: int = 768             # SwiGLU hidden dim (~8/3 * d_model)
    max_seq_len: int = 256      # Context window size
    dropout: float = 0.1        # Dropout rate
    pad_token_id: int = 0       # Padding token ID
    bos_token_id: int = 1       # Beginning of sequence token ID
    eos_token_id: int = 2       # End of sequence token ID

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization for improved training stability."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for superior grammatical/syntactic awareness."""
    def __init__(self, dim: int, max_seq_len: int = 1024, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, dim)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        q_rot = self._apply_rotary(q, cos, sin)
        k_rot = self._apply_rotary(k, cos, sin)
        return q_rot, k_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)

class CausalSelfAttention(nn.Module):
    """Multi-Head Causal Self-Attention with RoPE and Scaled Dot-Product Attention."""
    def __init__(self, cfg: BabyLMConfig, rope: RotaryEmbedding):
        super().__init__()
        self.cfg = cfg
        self.rope = rope
        self.d_head = cfg.d_model // cfg.n_heads
        assert cfg.d_model % cfg.n_heads == 0, "d_model must be divisible by n_heads"

        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_dropout = cfg.dropout

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.cfg.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.cfg.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.cfg.n_heads, self.d_head).transpose(1, 2)

        q, k = self.rope(q, k, S)

        # PyTorch 2.0+ optimized scaled dot product with causal masking
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True if attention_mask is None else False,
        )

        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network providing higher expressive capacity for small models."""
    def __init__(self, cfg: BabyLMConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down_proj = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: BabyLMConfig, rope: RotaryEmbedding):
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg, rope)
        self.norm2 = RMSNorm(cfg.d_model)
        self.mlp = SwiGLUMLP(cfg)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x

class BabyTransformerLM(nn.Module):
    """Modern LLaMA-style Decoder-only Language Model (~15M parameters)."""
    def __init__(self, cfg: BabyLMConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_token_id)
        self.rope = RotaryEmbedding(cfg.d_model // cfg.n_heads, max_seq_len=cfg.max_seq_len)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, self.rope) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying to reduce parameter count and improve generalization
        self.lm_head.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, S = input_ids.shape
        x = self.token_emb(input_ids)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        x = self.norm(x)
        logits = self.lm_head(x)  # (B, S, vocab_size)

        loss = None
        if targets is not None:
            # Shift alignment check: logits predict next token
            # ignore_index=-100 masks out user prompts, system prompts, and pad tokens in SFT
            loss = F.cross_entropy(
                logits.view(-1, self.cfg.vocab_size),
                targets.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 60,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        stop_token_ids: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-k/top-p sampling and early stop on delimiter tokens.
        
        ATTENTION LOOKUP MECHANISM:
        At each generation step t, the query vector q_t = W_q x_t compares against all past key vectors
        k_1, ..., k_{t-1} across the entire multi-turn context. When generating pronouns ('she', 'her', 'it')
        or entity names ('Minik'), the attention weights softmax(q_t K^T / sqrt(d)) peak at the earlier
        tokens where the referent entity was introduced, retrieving its semantic features into value vector v.
        """
        self.eval()
        stop_ids = set(stop_token_ids or [self.cfg.eos_token_id])

        for _ in range(max_new_tokens):
            idx_cond = input_ids if input_ids.size(1) <= self.cfg.max_seq_len else input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float("Inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[:, indices_to_remove] = -float("Inf")

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)

            if next_token.item() in stop_ids or next_token.item() == self.cfg.eos_token_id:
                break

        return input_ids
