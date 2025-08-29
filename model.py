# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import config
from plant_net import PlantLayer

class RotaryEmbedding(nn.Module):

    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def _update_cache(self, seq_len, device):
        if self.seq_len_cached is not None and self.seq_len_cached >= seq_len and self.cos_cached.device == device:
            return
        self.seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()[None, :, None, :]
        self.sin_cached = emb.sin()[None, :, None, :]

    def _apply_rotary_emb(self, x, cos, sin):
        x_reshaped = torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), dim=-1)
        return x * cos + x_reshaped * sin

    def forward(self, q, k):
        seq_len = q.shape[1]
        self._update_cache(seq_len, q.device)
        return (
            self._apply_rotary_emb(q, self.cos_cached[:, :seq_len, ...], self.sin_cached[:, :seq_len, ...]),
            self._apply_rotary_emb(k, self.cos_cached[:, :seq_len, ...], self.sin_cached[:, :seq_len, ...]),
        )

class SwiGLU(nn.Module):

    def __init__(self, dim, hidden_dim_multiplier, dropout):
        super().__init__()
        hidden_dim = int(hidden_dim_multiplier * ((2 * dim) / 3))
        hidden_dim = 256 * ((hidden_dim + 256 - 1) // 256)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class Attention(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = dim // n_heads
        assert (self.head_dim % 2) == 0, "head_dim must be even for RoPE"
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _repeat_kv(self, x):
        bs, seq_len, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, seq_len, n_kv_heads, self.n_rep, head_dim)
            .reshape(bs, seq_len, n_kv_heads * self.n_rep, head_dim)
        )

    def forward(self, x, rope, causal_mask, padding_mask: Optional[torch.Tensor] = None, global_hormones: Optional[dict] = None):
        bs, seq_len, dim = x.shape

        q = self.wq(x).view(bs, seq_len, self.n_heads, self.head_dim)
        k = self.wk(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(bs, seq_len, self.n_kv_heads, self.head_dim)

        q, k = rope(q, k)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        if global_hormones is not None:
            dopamin = global_hormones.get('dopamin', 1.0)
            v = v * dopamin

        final_mask = causal_mask
        if padding_mask is not None:
            final_mask = final_mask + padding_mask

        if final_mask is not None:
            final_mask = final_mask.to(q.dtype)

        output = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            attn_mask=final_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0
        )

        output = output.transpose(1, 2).contiguous().view(bs, seq_len, dim)
        return self.resid_dropout(self.wo(output))

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, n_kv_heads, ff_multiplier, dropout):
        super().__init__()
        self.attention = Attention(dim, n_heads, n_kv_heads, dropout)
        self.feed_forward = SwiGLU(dim, ff_multiplier, dropout)
        self.attention_norm = nn.LayerNorm(dim)
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(self, x, rope, causal_mask, padding_mask, global_hormones: Optional[dict] = None):
  
        residual_factor_attn = 1.0
        residual_factor_ffn = 1.0
        if global_hormones is not None:
            serotonin = global_hormones.get('serotonin', 1.0)
            kortizol = global_hormones.get('kortizol', 1.0)
            oxytocin = global_hormones.get('oxytocin', 1.0)
            kortizol_mod_factor = 1.0 - (kortizol - 1.0) * 0.5 # Sensitivity, can be fine-tuned
            
            residual_factor_attn = serotonin * oxytocin * kortizol_mod_factor
            residual_factor_ffn = serotonin * oxytocin * kortizol_mod_factor # can be another factor for FFN

        h = x + self.attention(self.attention_norm(x), rope, causal_mask, padding_mask, global_hormones) * residual_factor_attn

        out = h + self.feed_forward(self.ffn_norm(h)) * residual_factor_ffn

        return out

class Transformer(nn.Module):

    def __init__(self, vocab_size, dim, n_layers, n_heads, n_kv_heads, ff_multiplier, dropout):
        super().__init__()
        self.tok_embeddings = nn.Embedding(vocab_size, dim)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, n_kv_heads, ff_multiplier, dropout)
            for _ in range(n_layers)
        ])
            
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight
        
        self.rope = RotaryEmbedding(dim // n_heads)

        self.use_plant_net = config.USE_PLANT_NETWORK
        if self.use_plant_net:
            self.plant_layer = PlantLayer()

    def forward(self, tokens, pad_id: Optional[int] = None, prompt_text: str = None, return_hidden: bool = False):
        bs, seq_len = tokens.shape
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)

        causal_mask = torch.full((seq_len, seq_len), float("-inf"), device=tokens.device)
        causal_mask = torch.triu(causal_mask, diagonal=1)[None, None, :, :]

        padding_mask = None
        if pad_id is not None and pad_id >= 0:
            padding_mask = (tokens == pad_id)[:, None, None, :]
            padding_mask = padding_mask.to(h.dtype).masked_fill(padding_mask, float("-inf"))

        global_hormones = None
        if self.use_plant_net and self.plant_layer is not None:
            global_hormones = self.plant_layer.get_global_hormones()

        for layer in self.layers:
            h = layer(h, self.rope, causal_mask, padding_mask, global_hormones)

        h = self.norm(h)
        logits = self.output(h)

        if return_hidden:
            return logits, h
        return logits
