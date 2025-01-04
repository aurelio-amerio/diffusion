import jax
from jax import numpy as jnp
from flax import nnx

class SimpleTimeEmbedding(nnx.Module):
    def __init__(self):
        """Simple time embedding module. Mostly used to embed time.

        """
        return 
    def __call__(self, t):
        t = jnp.atleast_2d(t)
        out = jnp.concatenate([
            t - 0.5,
            jnp.cos(2 * jnp.pi * t),
            jnp.sin(2 * jnp.pi * t),
            -jnp.cos(4 * jnp.pi * t)
        ], axis=-1)
        return out


class SinusoidalEmbedding(nnx.Module):
    def __init__(self, output_dim: int = 128):
        """Sinusoidal embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
        """
        self.output_dim = output_dim
        return

    def __call__(self, inputs):
        half_dim = self.output_dim // 2 + 1
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = inputs[..., None] * emb[None, ...]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], -1)
        return emb[..., : self.output_dim]


class GaussianFourierEmbedding(nnx.Module):
    def __init__(
        self,
        output_dim: int = 128,
        learnable: bool = False,
        *,
        rngs: nnx.Rngs
    ):
        """Gaussian Fourier embedding module. Mostly used to embed time.

        Args:
            output_dim (int, optional): Output dimesion. Defaults to 128.
        """
        self.output_dim = output_dim
        self.B = nnx.Param(jax.random.normal(rngs.params(), [self.output_dim // 2 + 1]))
        self.learnable = learnable
        return

        
    def __call__(self, inputs):
        if not self.learnable:
            B = jax.lax.stop_gradient(self.B)
        else:
            B = self.B
        arg = 2 * jnp.pi * inputs[...,None]* B[None,...]
        term1 = jnp.cos(arg)
        term2 = jnp.sin(arg)
        out = jnp.concatenate([term1, term2], axis=-1)
        return out[..., : self.output_dim]