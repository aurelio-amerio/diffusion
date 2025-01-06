# we define a base class for the forward and reverse sde
# to avoid code duplication

# for each sde, we define the drift, diffusion and marginals
# for the reverse sde, we implement a sampler given a solver using diffrax
# the reverse sde is defined from the forward sde

import abc
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
import jax.random as random
from functools import partial
from diffrax import (
    diffeqsolve,
    ControlTerm,
    MultiTerm,
    ODETerm,
    VirtualBrownianTree,
    UnsafeBrownianPath,
    DirectAdjoint,
    Euler,
)

import numpyro.distributions as dist

from typing import Callable


class BaseSDE(abc.ABC):
    def __init__(self, dim, prior_dist):
        self.dim = dim
        self.prior_dist = prior_dist
        pass

    @abc.abstractmethod
    def drift(self, x, t):
        """
        Drift function of the SDE.
        """
        pass

    @abc.abstractmethod
    def diffusion(self, x, t):
        """
        Diffusion function of the SDE.
        """
        pass

    @abc.abstractmethod
    def marginal_dist(self, x0, t):
        """
        Marginal probability.
        """
        pass

    @abc.abstractmethod
    def reverse(self):
        """
        Return the reverse SDE.
        """
        pass

    @abc.abstractmethod
    def get_loss_function(self):
        """
        Get the loss function for denoising score matching.
        """
        pass

    @abc.abstractmethod
    def _tree_flatten(self):
        pass

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class ReverseSDE(abc.ABC):
    def __init__(self, forward_sde: BaseSDE, score: Callable, dim, **kwargs):
        self.dim = dim
        self.forward_sde = forward_sde
        self.score = score

    def drift(self, x, t):
        """
        Drift function of the reverse SDE.
        """
        forward_drift = self.forward_sde.drift
        diffusion = self.forward_sde.diffusion
        # to obtain a process for the reverse time, we need to reverse the drift and diffusion of the forward sde
        res = -forward_drift(x, 1 - t) + jnp.multiply(
            jnp.square(diffusion(x, 1 - t)), jnp.squeeze(self.score(x, 1 - t))
        )
        return res

    def diffusion(self, x, t):
        """
        Diffusion function of the reverse SDE.
        """
        return self.forward_sde.diffusion(x, 1 - t)

    def sample(self, rng, n_samples, solver=Euler(), safe=True):
        """
        Sample from the reverse SDE.
        """
        t0, t1 = 0.0, 0.999

        def diff(t, y, args):
            return jnp.diag(jnp.broadcast_to(self.diffusion(y, t), (self.dim,)))

        def drift(t, y, args):
            return self.drift(y, t)

        keys = jax.random.split(rng, n_samples)

        @jit
        def sample_one(key):
            keys = jax.random.split(key, 2)
            y0 = jax.random.normal(keys[0], (self.dim,))

            brownian_motion = VirtualBrownianTree(
                t0, t1, tol=1e-5, shape=(self.dim,), key=keys[1]
            )
            terms = MultiTerm(ODETerm(drift), ControlTerm(diff, brownian_motion))
            sol = diffeqsolve(terms, solver, t0, t1, dt0=0.001, y0=y0)
            return sol.ys

        @jit
        def sample_one_unsafe(key):
            keys = jax.random.split(key, 2)
            y0 = jax.random.normal(keys[0], (self.dim,))

            brownian_motion = UnsafeBrownianPath(shape=(self.dim,), key=keys[1])
            terms = MultiTerm(ODETerm(drift), ControlTerm(diff, brownian_motion))
            sol = diffeqsolve(
                terms, solver, t0, t1, dt0=0.001, y0=y0, adjoint=DirectAdjoint()
            )
            return sol.ys

        if safe:
            res = vmap(sample_one)(keys)
        else:
            res = vmap(sample_one_unsafe)(keys)

        return jnp.squeeze(res)

    @abc.abstractmethod
    def _tree_flatten(self):
        pass

    @classmethod
    def _tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)


class VPSDE(BaseSDE):
    """
    Variance Preseverving SDE, also known as the Ornstein-Uhlenbeck SDE.
    """

    def __init__(self, dim, beta_min: float = 0.001, beta_max: float = 3.):
        prior_dist = dist.MultivariateNormal(
            loc=jnp.zeros(dim), covariance_matrix=jnp.eye(dim)
        )
        super().__init__(dim, prior_dist)
        self.diff_steps = 1000
        self.beta_min = beta_min
        self.beta_max = beta_max
        return

    def beta_t(self, t):
        """
        Beta function of the VPSDE.
        """
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def alpha_t(self, t):
        """
        Alpha function of the VPSDE.
        """
        return t * self.beta_min + 0.5 * t**2 * (self.beta_max - self.beta_min)

    def drift(self, x, t):
        """
        Drift function of the VPSDE.
        """
        return -0.5 * self.beta_t(t) * x

    def diffusion(self, x, t):
        """
        Diffusion function of the VPSDE.
        """
        return jnp.sqrt(self.beta_t(t))

    def mean_factor(self, t):
        """
        t: time (number)
        returns m_t as above
        """
        return jnp.exp(-0.5 * self.alpha_t(t))

    def var(self, t):
        """
        t: time (number)
        returns v_t as above
        """
        return 1 - jnp.exp(-self.alpha_t(t))

    def marginal_dist(self, x0, t):
        """
        Marginal probability of the VPSDE.
        """
        mean_coeff = self.mean_factor(t)
        var_coeff = self.var(t)
        std = jnp.sqrt(var_coeff)
        covar = jnp.diag(std)
        return dist.MultivariateNormal(loc=mean_coeff * x0, covariance_matrix=covar)

    def reverse(self, score):
        """
        Return the reverse VPSDE.
        """
        return R_VPSDE(self, score, self.dim)

    def get_loss_function(self):
        """
        Get the loss function for denoising score matching.
        """

        def loss_fn(score_model, x0, rng):
            N_batch = x0.shape[0]
            rng, step_key = random.split(rng)
            t = random.randint(step_key, (N_batch, 1), 1, self.diff_steps) / (
                self.diff_steps - 1
            )
            mean_coeff = self.mean_factor(t)
            # is it right to have the square root here for the loss?
            vs = self.var(t)
            stds = jnp.sqrt(vs)
            rng, step_key = random.split(rng)
            noise = random.normal(step_key, x0.shape)
            xt = x0 * mean_coeff + noise * stds
            output = score_model(xt, t)
            loss = jnp.mean((noise + output * stds) ** 2)
            return loss

        return loss_fn

    def _tree_flatten(self):

        children = (None,)  # arrays / dynamic values

        aux_data = {"dim": self.dim, "prior_dist": self.prior_dist}  # static values

        return (children, aux_data)


class R_VPSDE(ReverseSDE):
    def __init__(self, forward_sde, score, dim, **kwargs):
        super().__init__(forward_sde, score, dim, **kwargs)
        return

    def _tree_flatten(self):

        children = (None,)  # arrays / dynamic values

        aux_data = {
            "dim": self.dim,
            "forward_sde": self.forward_sde,
            "score": self.score,
        }  # static values

        return (children, aux_data)
