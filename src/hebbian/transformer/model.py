"""
GPT2 model for associative-recall experiments.

Includes RoPE, flash attention, pretrained-weight loading, generation, and the
optional BinaryMoE fact-expert path used by the fact-editing experiments.

References:
1) https://github.com/karpathy/nanoGPT
2) https://github.com/openai/gpt-2/blob/master/src/model.py
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from hebbian.transformer.norms import get_norm, FrozenRMSNorm

try:
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb,
    )
    from transformers import LlamaConfig

    HUGGINGFACE_ROPE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_ROPE_AVAILABLE = False


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 1
    n_head: int = 1
    n_embd: int = 64
    dropout: float = 0.0
    bias: bool = False
    # Residual connections
    mlp_residual: bool = False
    attn_residual: bool = False
    # RoPE and positional encoding
    use_rope: bool = True
    rope_base: float = 10000.0
    no_positional_encoding: bool = False  # if True, disables both RoPE and wpe
    # Normalization
    mlp_norm_type: str = "frozen_rmsnorm"
    attn_norm_type: str = "rmsnorm"
    lm_head_norm_type: str = "rmsnorm"
    # Embedding
    tie_embeddings: bool = True
    # Special init
    freeze_value_dense_identity: bool = True
    use_mlp_qk: bool = False
    # Use identity MLP (passes input through unchanged)
    use_identity_mlp: bool = False
    # Optional old-style BinaryMoE path used by fact-editing experiments.
    use_moe: bool = False
    moe_router_num_layers: int = 2
    moe_router_intermediate_dim: int | None = None
    moe_gate: bool = False
    moe_router_use_mlp_input: bool = False
    moe_convex: bool = True
    moe_mlp_type: str = "lora_linear"  # "moe" | "mlp" | "linear" | "lora_linear"
    moe_mlp_out_norm: bool = False
    moe_lora_linear_rank: int = 8


class AttentionProjectionMLP(nn.Module):
    """Old-style nonlinear Q/K projection used by the original fact-editing run."""

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        return self.mlp(self.norm(x))


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        if config.use_mlp_qk:
            self.c_q = AttentionProjectionMLP(config.n_embd)
            self.c_k = AttentionProjectionMLP(config.n_embd)
        else:
            self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # RoPE is disabled if no_positional_encoding is set
        self.use_rope = config.use_rope and not config.no_positional_encoding
        if self.use_rope:
            assert HUGGINGFACE_ROPE_AVAILABLE, (
                "RoPE requires HuggingFace transformers. Install with: pip install transformers"
            )
            head_dim = config.n_embd // config.n_head
            assert head_dim % 2 == 0
            self.rope_dim = head_dim

            rope_config = LlamaConfig(
                hidden_size=config.n_embd,
                num_attention_heads=config.n_head,
                max_position_embeddings=config.block_size,
                rope_theta=config.rope_base,
                rope_scaling=None,
            )
            self.rope = LlamaRotaryEmbedding(rope_config)

        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def _compute_rope_cos_sin(
        self,
        x_rope: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin with a CUDA-safe fallback.

        Some CUDA stacks fail in HF's LlamaRotaryEmbedding batched matmul path
        (CUBLAS_STATUS_INVALID_VALUE for batch size > 1). In that case, use an
        equivalent einsum-based computation.
        """
        if x_rope.is_cuda and position_ids.shape[0] > 1:
            inv_freq = self.rope.inv_freq.to(device=x_rope.device, dtype=torch.float32)
            pos = position_ids.to(device=x_rope.device, dtype=torch.float32)
            freqs = torch.einsum("d,bt->btd", inv_freq, pos)
            emb = torch.cat((freqs, freqs), dim=-1)
            scale = getattr(self.rope, "attention_scaling", 1.0)
            cos = (emb.cos() * scale).to(dtype=x_rope.dtype)
            sin = (emb.sin() * scale).to(dtype=x_rope.dtype)
            return cos, sin

        return self.rope(x_rope, position_ids)

    def forward(self, x):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.use_rope:
            q_rope = q[..., : self.rope_dim]
            k_rope = k[..., : self.rope_dim]
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
            cos, sin = self._compute_rope_cos_sin(q_rope, position_ids)
            q_rope, k_rope = apply_rotary_pos_emb(q_rope, k_rope, cos, sin)
            if self.rope_dim < q.shape[-1]:
                q = torch.cat([q_rope, q[..., self.rope_dim :]], dim=-1)
                k = torch.cat([k_rope, k[..., self.rope_dim :]], dim=-1)
            else:
                q = q_rope
                k = k_rope

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class IdentityMLP(nn.Module):
    """Identity MLP that passes input through unchanged.
    
    Used to effectively disable the MLP component in transformer blocks
    while maintaining the architecture structure. Useful for attention-only
    experiments.
    """

    def forward(self, x):
        return x


class BinaryRouter(nn.Module):
    def __init__(
        self,
        inp_dim: int,
        hidden_dim: int | None = None,
        nlayers: int = 1,
        convex: bool = True,
    ):
        super().__init__()
        assert nlayers >= 1, "nlayers must be at least 1"
        self.convex = convex
        out_dim = 1 if convex else 2
        act_fn = nn.Sigmoid if convex else nn.GELU

        if nlayers == 1:
            self.mlp = nn.Sequential(nn.Linear(inp_dim, out_dim), act_fn())
        else:
            if hidden_dim is None:
                hidden_dim = inp_dim
            layers = [nn.Linear(inp_dim, hidden_dim), nn.GELU()]
            for _ in range(nlayers - 2):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers += [nn.Linear(hidden_dim, out_dim), act_fn()]
            self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        out = self.mlp(x)
        if self.convex:
            return out, 1 - out
        return out[..., :1], out[..., 1:]


class LowRankLinear(nn.Module):
    def __init__(self, input_output_dim: int, rank: int):
        super().__init__()
        self.input_output_dim = input_output_dim
        self.rank = rank
        self.U = nn.Parameter(torch.randn(input_output_dim, rank))
        self.V = nn.Parameter(torch.randn(rank, input_output_dim))

    def forward(self, x):
        return x @ self.U @ self.V


class BinaryMoE(nn.Module):
    def __init__(
        self,
        router: nn.Module,
        fact_expert: nn.Module,
        other_expert: nn.Module,
        *,
        gate: bool = False,
        use_mlp_input: bool = False,
        mlp_out_norm: bool = False,
    ):
        super().__init__()
        self.router = router
        self.fact_expert = fact_expert
        self.other_expert = other_expert
        self.gate = gate
        self.use_mlp_input = use_mlp_input
        self.mlp_out_norm = mlp_out_norm

    def forward(self, prev_x, new_x):
        router_input = new_x if self.use_mlp_input else prev_x
        expert_input = new_x if self.use_mlp_input else prev_x
        weight_fact, weight_other = self.router(router_input)
        other_output = self.other_expert(expert_input)
        fact_output = self.fact_expert(new_x)
        if self.mlp_out_norm:
            other_output = F.normalize(other_output, dim=-1)
            fact_output = F.normalize(fact_output, dim=-1)
        if self.gate:
            weight_other = torch.masked_fill(weight_other, weight_other <= 0.5, 0.0)
            weight_fact = torch.masked_fill(weight_fact, weight_fact < 0.5, 0.0)
        return weight_other * other_output + weight_fact * fact_output


def get_moe(config: GPTConfig) -> BinaryMoE:
    d_model = config.n_embd
    router = BinaryRouter(
        d_model,
        config.moe_router_intermediate_dim,
        config.moe_router_num_layers,
        convex=config.moe_convex,
    )
    if config.moe_mlp_type in {"moe", "mlp"}:
        other_expert = MLP(config)
    elif config.moe_mlp_type == "linear":
        other_expert = nn.Linear(d_model, d_model, bias=config.bias)
    elif config.moe_mlp_type == "lora_linear":
        other_expert = LowRankLinear(d_model, config.moe_lora_linear_rank)
    else:
        raise ValueError(f"MLP type {config.moe_mlp_type!r} not supported")
    fact_expert = MLP(config)
    return BinaryMoE(
        router,
        fact_expert,
        other_expert,
        gate=config.moe_gate,
        use_mlp_input=config.moe_router_use_mlp_input,
        mlp_out_norm=config.moe_mlp_out_norm,
    )


class Block(nn.Module):

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.ln_1 = get_norm(config.attn_norm_type, config.n_embd, config.bias)
        self.attn = CausalSelfAttention(config, layer_idx=layer_idx)
        self.ln_2 = get_norm(config.mlp_norm_type, config.n_embd, config.bias)
        if config.use_identity_mlp:
            self.mlp = IdentityMLP()
        elif config.use_moe:
            self.mlp = get_moe(config)
        else:
            self.mlp = MLP(config)
        self.mlp_residual = config.mlp_residual
        self.attn_residual = config.attn_residual

    def forward(self, x):
        attn_output = self.attn(self.ln_1(x))
        if self.attn_residual:
            x = x + attn_output
        else:
            x = attn_output

        mlp_input = self.ln_2(x)
        if isinstance(self.mlp, BinaryMoE):
            mlp_output = self.mlp(x, mlp_input)
        else:
            mlp_output = self.mlp(mlp_input)
        if self.mlp_residual:
            x = x + mlp_output
        else:
            x = mlp_output
        return x

    def get_frozen_rms_scale(self, mlp_embeddings):
        if isinstance(self.ln_1, FrozenRMSNorm):
            self.ln_1.get_frozen_rms_scale(mlp_embeddings)
        if isinstance(self.ln_2, FrozenRMSNorm):
            self.ln_2.get_frozen_rms_scale(mlp_embeddings)


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        # Positional embedding: use wpe only if not using RoPE AND not disabled entirely
        use_pos_emb = not config.use_rope and not config.no_positional_encoding

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=(
                    nn.Embedding(config.block_size, config.n_embd)
                    if use_pos_emb
                    else None
                ),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList(
                    [Block(config, layer_idx=i) for i in range(config.n_layer)]
                ),
                ln_f=get_norm(config.lm_head_norm_type, config.n_embd, config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        if config.freeze_value_dense_identity:
            for block in self.transformer.h:
                torch.nn.init.eye_(block.attn.c_proj.weight)
                block.attn.c_proj.weight.requires_grad = False
                if block.attn.c_proj.bias is not None:
                    torch.nn.init.zeros_(block.attn.c_proj.bias)
                    block.attn.c_proj.bias.requires_grad = False
                torch.nn.init.eye_(block.attn.c_v.weight)
                block.attn.c_v.weight.requires_grad = False
                if block.attn.c_v.bias is not None:
                    torch.nn.init.zeros_(block.attn.c_v.bias)
                    block.attn.c_v.bias.requires_grad = False

    def get_frozen_rms_scale(self, mlp_embeddings):
        for block in self.transformer.h:
            block.get_frozen_rms_scale(mlp_embeddings)
        if isinstance(self.transformer.ln_f, FrozenRMSNorm):
            self.transformer.ln_f.get_frozen_rms_scale(mlp_embeddings)

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and self.transformer.wpe is not None:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, mask_index=-100):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        mask = idx != mask_index
        idx_for_embedding = idx.clone()
        idx_for_embedding[~mask] = 0

        tok_emb = self.transformer.wte(idx_for_embedding)

        if self.transformer.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        x = x * mask.unsqueeze(-1)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=mask_index,
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    def forward_hidden(self, idx, mask_index=-100):
        """Forward pass returning the hidden state after ln_f (before lm_head).

        Args:
            idx: Input token indices of shape (B, T).
            mask_index: Token index used as padding mask (default -100).

        Returns:
            Tensor of shape (B, T, n_embd): hidden states after ln_f.
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )

        mask = idx != mask_index
        idx_for_embedding = idx.clone()
        idx_for_embedding[~mask] = 0

        tok_emb = self.transformer.wte(idx_for_embedding)

        if self.transformer.wpe is not None:
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
        else:
            x = self.transformer.drop(tok_emb)

        x = x * mask.unsqueeze(-1)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        return x

    def crop_block_size(self, block_size):
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        if self.transformer.wpe is not None:
            self.transformer.wpe.weight = nn.Parameter(
                self.transformer.wpe.weight[:block_size]
            )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024
        config_args["bias"] = True
        config_args["use_rope"] = False
        config_args["mlp_residual"] = True
        config_args["attn_residual"] = True
        config_args["freeze_value_dense_identity"] = False
        config_args["mlp_norm_type"] = "layernorm"
        config_args["attn_norm_type"] = "layernorm"
        config_args["lm_head_norm_type"] = "layernorm"
        if "dropout" in override_args:
            config_args["dropout"] = override_args["dropout"]

        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(".attn.bias")]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()
        sd_keys_hf = [
            k
            for k in sd_hf.keys()
            if not k.endswith(".attn.masked_bias") and not k.endswith(".attn.bias")
        ]
        transposed = ["attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"]

        assert len(sd_keys_hf) == len(sd_keys)
        for k in sd_keys_hf:
            if k.endswith("attn.c_attn.weight"):
                original_weight = sd_hf[k].t()
                n_embd = original_weight.shape[0]
                with torch.no_grad():
                    sd[k.replace("c_attn.weight", "c_q.weight")].copy_(
                        original_weight[:, :n_embd].t()
                    )
                    sd[k.replace("c_attn.weight", "c_k.weight")].copy_(
                        original_weight[:, n_embd : 2 * n_embd].t()
                    )
                    sd[k.replace("c_attn.weight", "c_v.weight")].copy_(
                        original_weight[:, 2 * n_embd :].t()
                    )
            elif k.endswith("attn.c_attn.bias"):
                original_bias = sd_hf[k]
                n_embd = original_bias.shape[0] // 3
                with torch.no_grad():
                    sd[k.replace("c_attn.bias", "c_q.bias")].copy_(
                        original_bias[:n_embd]
                    )
                    sd[k.replace("c_attn.bias", "c_k.bias")].copy_(
                        original_bias[n_embd : 2 * n_embd]
                    )
                    sd[k.replace("c_attn.bias", "c_v.bias")].copy_(
                        original_bias[2 * n_embd :]
                    )
            elif any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
