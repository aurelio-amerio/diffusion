#%%
import sys
sys.path.append("src")
#%%
import jax 
from jax import numpy as jnp
from flax import nnx
import diffusion
from diffusion.embedding import SimpleTimeEmbedding, SinusoidalEmbedding, GaussianFourierEmbedding
# %%
t = jnp.array([0.1, 0.2, 0.3])
#%%
ste = SimpleTimeEmbedding()

# %%
ste(t)
# %%
se = SinusoidalEmbedding(4)
# %%
se(t)
# %%
gft = GaussianFourierEmbedding(128, False, rngs=nnx.Rngs(0))
# %%
gft(t).shape
# %%
@nnx.jit
def test(t):
    return gft(t)
# %%
test(t)
# %%
