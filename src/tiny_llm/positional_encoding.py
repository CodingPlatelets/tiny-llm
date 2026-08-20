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
        # x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
        x_shape = x.shape
        x = x.reshape(*x.shape[:-1], self.dims // 2, 2)
        # sin_offset = self.sin[offset] if offset is not None else self.sin[:x_shape[1]]
        # cos_offset = self.cos[offset] if offset is not None else self.cos[:x_shape[1]]
        if offset is None:
            offset = slice(0, x_shape[1])
        sin_offset = self.sin[offset]
        cos_offset = self.cos[offset]
        out0 = x[..., 0] * cos_offset[None, :, None, :] + \
            x[..., 1] * (-sin_offset[None, :, None, :])
        out1 = x[..., 0] * sin_offset[None, :, None, :] + \
            x[..., 1] * cos_offset[None, :, None, :]
        return mx.stack([out0, out1], axis=-1).reshape(x_shape)
