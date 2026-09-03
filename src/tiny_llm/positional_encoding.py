import mlx.core as mx


class RoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.max_seq_len = seq_len
        self.base = base
        self.traditional = traditional
        i = mx.arange(0, self.dims // 2)
        theta = self.base ** (-2 * i / self.dims)
        m = mx.arange(0, self.max_seq_len)
        # angle = mx.matmul(mx.expand_dims(m, -1), mx.expand_dims(theta, 0))
        angle = m[:, None] * theta[None, :]
        self.sin = mx.sin(angle)
        self.cos = mx.cos(angle)

    def __call__(
        self, x: mx.array, offset: list[slice] | slice | None = None
    ) -> mx.array:
        N, L, H, D = x.shape
        if offset is None:
            offset = slice(0, L)
        cos_offset = self.cos[offset].reshape(1, L, 1, D // 2)
        sin_offset = self.sin[offset].reshape(1, L, 1, D // 2)
        if self.traditional:
            x = x.reshape(N, L, H, D // 2, 2)
            x1 = x[..., 0]
            x2 = x[..., 1]
        else:
            x1, x2 = x[..., :self.dims // 2], x[..., self.dims // 2:]
        out0 = x1 * cos_offset - x2 * sin_offset
        out1 = x1 * sin_offset + x2 * cos_offset
        return mx.stack([out0, out1], axis=-1).reshape(N, L, H, D) if self.traditional else mx.concatenate([out0, out1], axis=-1)
