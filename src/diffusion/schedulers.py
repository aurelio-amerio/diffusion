from diffusers import FlaxScoreSdeVeScheduler
import jax 
from jax import numpy as jnp
# from flax import nnx

class FlaxScoreSdeVeScheduler(FlaxScoreSdeVeScheduler):
    def add_noise(
        self,
        original_samples: jnp.ndarray,
        noise: jnp.ndarray,
        timesteps: jnp.ndarray,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.discrete_sigmas[timesteps]
        key, subkey = jax.random.split(key,1)
        noise = (
            noise * sigmas[:, None, None, None]
            if noise is not None
            else jax.random.normal(subkey, original_samples.shape) * sigmas[:, None, None, None]
        )
        noisy_samples = noise + original_samples
        return noisy_samples