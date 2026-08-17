import mlx.core as mx

from .basics import linear, softmax


def scaled_dot_product_attention_simple(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
) -> mx.array:
    # Attention = softmax(Q @ K^t / dk ^ 0.5 + M) @ V
    key_t = mx.swapaxes(key, -1, -2)
    out = mx.matmul(query, key_t)
    real_scale = 1.0 / mx.sqrt(query.shape[-1]) if scale is None else scale
    out = out * real_scale
    out = out + mask if mask is not None else out
    p = softmax(out, -1)
    return mx.matmul(p, value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: mx.array,
        wk: mx.array,
        wv: mx.array,
        wo: mx.array,
    ):
        '''
        query/key/value: N x L x E
        E is hidden_size or embed_dim or dims or model_dim
        H is num_heads
        D is head_dim
        L is seq_len, in PyTorch API it's S (source len)
        w_q/w_k/w_v: (H x D) x E
        output/input: N x L x E
        w_o: E x (H x D)
        '''
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.E = hidden_size
        self.H = num_heads
        self.D = hidden_size // num_heads

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        value: mx.array,
        mask: mx.array | None = None,
    ) -> mx.array:
        # qkv proj
        query_proj = mx.swapaxes(
            linear(query, self.wq).reshape(*query.shape[:-1], self.H, self.D), -2, -3)
        key_proj = mx.swapaxes(
            linear(key, self.wk).reshape(*key.shape[:-1], self.H, self.D), -2, -3)
        value_proj = mx.swapaxes(
            linear(value, self.wv).reshape(*value.shape[:-1], self.H, self.D), -2, -3)
        out = scaled_dot_product_attention_simple(
            query=query_proj, key=key_proj, value=value_proj, mask=mask)
        out = mx.swapaxes(out, -2, -3)
        out = out.reshape(*out.shape[:2], self.H * self.D)
        return linear(out, self.wo)


def causal_mask(L: int, S: int, dtype: mx.Dtype) -> mx.array:
    pass


def scaled_dot_product_attention_grouped(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass


def paged_attention(
    query: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float | None = None,
    mask: mx.array | str | None = None,
) -> mx.array:
    pass
