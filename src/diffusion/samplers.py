import jax
from jax import numpy as jnp
from jax import jit

# TODO still need to test
def edm_sampler(
    sde, model, x, *,
    key,
    condition_mask = None,
    condition_value = None,
    num_steps=18,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    method="heun",
    model_kwargs={},
):

    assert method in ["euler", "heun"], f"Unknown method: {method}"
    if condition_mask is not None:
            assert (
                condition_value is not None
            ), "Condition value must be provided if condition mask is provided"
    else:
        condition_mask = 0
        condition_value = 0

    # Time step discretization.
    step_indices = jnp.arange(num_steps)
    
    t_steps = sde.timesteps(step_indices, num_steps)
    t_steps = jnp.append(t_steps, 0)

    # Main sampling loop.
    x_next = x * t_steps[0]

    def one_step(carry, args):
        x_next, key = carry
        key, subkey = jax.random.split(key)
        i, t_cur, t_next = args
        x_curr = x_next

        # Increase noise temporarily.
        gamma = jnp.min(S_churn / num_steps, jnp.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = t_cur + gamma * t_cur # sigma at the specific time step
        x_hat = x_curr + jnp.sqrt(t_hat ** 2 - t_cur ** 2) * S_noise * jax.random.normal(subkey, x_curr.shape)
        x_hat = x_hat * (1 - condition_mask) + condition_value * condition_mask # Apply conditioning.

        # Euler step.
        denoised = sde.denoise(model, x_hat, t_hat, **model_kwargs)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        x_next = x_next * (1 - condition_mask) + condition_value * condition_mask # Apply conditioning.

        if method == "euler":
            return (x_next, key), None

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = sde.denoise(x_next, t_next, model, **model_kwargs)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            x_next = x_next * (1 - condition_mask) + condition_value * condition_mask # Apply conditioning.
        
        return (x_next, key), None
        
    i = jnp.arange(num_steps)
    args = (i, t_steps[:-1], t_steps[1:])
    carry, _ = jax.lax.scan(one_step, (x_next, key), args)
    return carry[0]