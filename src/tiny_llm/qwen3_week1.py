import mlx.core as mx
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding
from .quantize import dequantize_linear


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
    ):
        self.wq, self.wk, self.wv, self.wo = wq, wk, wv, wo
        self.rms_norm_eps = rms_norm_eps
        self.max_seq_len = max_seq_len
        self.hidden, self.Hq, self.H, self.D = hidden_size, num_heads, num_kv_heads, head_dim
        self.q_norm, self.k_norm = q_norm, k_norm
        self.rope = RoPE(dims=self.D, seq_len=self.max_seq_len,
                         base=theta, traditional=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        q = linear(x, self.wq).reshape(*x.shape[:-1], self.Hq, self.D)
        k = linear(x, self.wk).reshape(*x.shape[:-1], self.H, self.D)
        v = linear(x, self.wv).reshape(
            *x.shape[:-1], self.H, self.D).astype(mx.float32)
        q = mx.fast.rms_norm(q, self.q_norm, self.rms_norm_eps)
        k = mx.fast.rms_norm(k, self.k_norm, self.rms_norm_eps)
        q = self.rope(q, offset=slice(0, x.shape[-2])).astype(mx.float32)
        k = self.rope(k, offset=slice(0, x.shape[-2])).astype(mx.float32)
        q = mx.swapaxes(q, -3, -2)
        k = mx.swapaxes(k, -3, -2)
        v = mx.swapaxes(v, -3, -2)
        out = scaled_dot_product_attention_grouped(q, k, v, None, mask)
        out = mx.swapaxes(out, -2, -3)
        out = out.reshape(*out.shape[:2], self.Hq * self.D).astype(x.dtype)
        return linear(out, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
    ):
        self.gate, self.up, self.down = w_gate, w_up, w_down
        self.D = dim
        self.H = hidden_dim

    def __call__(self, x: mx.array) -> mx.array:
        up = linear(x, self.up)
        gate = linear(x, self.gate)
        si = silu(gate) * up
        return linear(si, self.down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
        q_norm: mx.array,
        k_norm: mx.array,
        w_gate: mx.array,
        w_up: mx.array,
        w_down: mx.array,
        w_input_layernorm: mx.array,
        w_post_attention_layernorm: mx.array,
        max_seq_len: int = 32768,
        theta: int = 1000000,
    ):
        pass

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        pass


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        pass

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        pass
