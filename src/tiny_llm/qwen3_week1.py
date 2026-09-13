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
        self.q_rms_norm = RMSNorm(
            dim=head_dim, weight=q_norm, eps=rms_norm_eps)
        self.k_rms_norm = RMSNorm(
            dim=head_dim, weight=k_norm, eps=rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        q = linear(x, self.wq).reshape(*x.shape[:-1], self.Hq, self.D)
        k = linear(x, self.wk).reshape(*x.shape[:-1], self.H, self.D)
        v = linear(x, self.wv).reshape(
            *x.shape[:-1], self.H, self.D).astype(mx.float32)
        q = self.q_rms_norm(q)
        k = self.k_rms_norm(k)
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
        self.attention = Qwen3MultiHeadAttention(hidden_size=hidden_size, num_heads=num_attention_heads, num_kv_heads=num_kv_heads,
                                                 head_dim=head_dim, wq=wq, wk=wk, wv=wv, wo=wo, q_norm=q_norm, k_norm=k_norm, max_seq_len=max_seq_len, theta=theta, rms_norm_eps=rms_norm_eps)
        self.mlp = Qwen3MLP(dim=hidden_size, hidden_dim=intermediate_size,
                            w_gate=w_gate, w_up=w_up, w_down=w_down)
        self.input_rms_norm = RMSNorm(
            dim=hidden_size, weight=w_input_layernorm, eps=rms_norm_eps)
        self.post_rms_norm = RMSNorm(
            dim=hidden_size, weight=w_post_attention_layernorm, eps=rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
    ) -> mx.array:
        x_rms = self.input_rms_norm(x)
        atten = x + self.attention(x_rms, mask=mask)
        atten_rms = self.post_rms_norm(atten)
        mlp = atten + self.mlp(atten_rms)
        return mlp


class Qwen3ModelWeek1:
    def __init__(self, mlx_model: Any):
        args = mlx_model.args
        inner = mlx_model.model

        self.num_hidden_layers = args.num_hidden_layers
        self.tie_word_embeddings = args.tie_word_embeddings

        # --- 1. Embedding：量化的，要反量化 ---
        self.embedding = Embedding(
            vocab_size=args.vocab_size,
            embedding_dim=args.hidden_size,      # hidden_size == embedding_dim
            weight=dequantize_linear(inner.embed_tokens),
        )

        # --- 2. 28 个 TransformerBlock ---
        self.layers_inner = []
        for i in range(args.num_hidden_layers):
            layer = inner.layers[i]
            self.layers_inner.append(
                Qwen3TransformerBlock(
                    num_attention_heads=args.num_attention_heads,
                    num_kv_heads=args.num_key_value_heads,
                    hidden_size=args.hidden_size,
                    head_dim=args.head_dim,
                    intermediate_size=args.intermediate_size,
                    rms_norm_eps=args.rms_norm_eps,
                    # QuantizedLinear → 必须反量化
                    wq=dequantize_linear(layer.self_attn.q_proj),
                    wk=dequantize_linear(layer.self_attn.k_proj),
                    wv=dequantize_linear(layer.self_attn.v_proj),
                    wo=dequantize_linear(layer.self_attn.o_proj),
                    # RMSNorm 的 weight 本来就是 bf16 普通数组 → 直接取
                    q_norm=layer.self_attn.q_norm.weight,
                    k_norm=layer.self_attn.k_norm.weight,
                    w_gate=dequantize_linear(layer.mlp.gate_proj),
                    w_up=dequantize_linear(layer.mlp.up_proj),
                    w_down=dequantize_linear(layer.mlp.down_proj),
                    w_input_layernorm=layer.input_layernorm.weight,
                    w_post_attention_layernorm=layer.post_attention_layernorm.weight,
                    max_seq_len=args.max_position_embeddings,
                    theta=args.rope_theta,
                )
            )

        # --- 3. 模型末尾那个 norm ---
        self.norm = RMSNorm(
            dim=args.hidden_size,
            weight=inner.norm.weight,
            eps=args.rms_norm_eps,
        )

        # --- 4. lm_head：tie 了就没有，靠 Embedding.as_linear ---
        self.w_lm_head = None
        if not args.tie_word_embeddings:
            self.w_lm_head = dequantize_linear(mlx_model.lm_head)

    def __call__(
        self,
        inputs: mx.array,
    ) -> mx.array:
        x = self.embedding(x=inputs)
        for layers in self.layers_inner:
            x = layers(x, mask='causal')

        x = self.norm(x)
        if self.w_lm_head is None:
            return self.embedding.as_linear(x)
        else:
            return linear(x, self.w_lm_head)
