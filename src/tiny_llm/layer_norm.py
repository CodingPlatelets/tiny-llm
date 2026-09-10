import mlx.core as mx


class RMSNorm:
    def __init__(self, dim: int, weight: mx.array, eps: float = 1e-5):
        self.weight = weight
        self.hidden = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        original_type = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x**2, axis=-1, keepdims=True)
        sq = mx.sqrt(mean + self.eps)
        return (x / sq).astype(original_type) * self.weight
