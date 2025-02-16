import abc
import jax.numpy as jnp
import jax
from jax import grad, jit, vmap
import jax.random as random
from functools import partial

# we will create an abstract SDE class which can implement VP, VE, and EDM methods, following https://github.com/NVlabs/edm/
# we will then define a precondition function for each method


class AbstractSDE(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def timesteps(self, i, N):
        pass

    @abc.abstractmethod
    def sigma(self, t):
        # also known as the schedule, as in tab 1 of EDM paper
        pass

    @abc.abstractmethod
    def sigma_prime(self, t):
        # also known as the schedule derivative
        pass

    @abc.abstractmethod
    def s(self, t):
        # also known as scaling, as in tab 1 of EDM paper
        pass

    @abc.abstractmethod
    def s_prime(self, t):
        # also known as scaling derivative
        pass

    @abc.abstractmethod
    def c_skip(self, sigma):
        # c_skip for preconditioning
        pass

    @abc.abstractmethod
    def c_out(self, sigma):
        # c_out for preconditioning
        pass

    @abc.abstractmethod
    def c_in(self, sigma):
        # c_in for preconditioning
        pass

    @abc.abstractmethod
    def c_noise(self, sigma):
        # c_noise for preconditioning
        pass

    def sample_sigma(self, key, shape):
        # sample sigma from the prior noise distribution
        t = jax.random.uniform(key, shape, minval=self.T0, maxval=self.T)    
        return self.sigma(t)

    def sample_noise(self, key, shape, sigma):
        # sample noise from the prior noise distribution with noise scale sigma(t)
        n = jax.random.normal(key, shape)*sigma
        return n

    @abc.abstractmethod
    def loss_weight(self, sigma):
        # weight for the loss function, for MLE estimation, also known as λ(σ) in the EDM paper
        pass

    def f(self, x, t):
        # f(x, sigma) in the SDE, also known as drift term for the forward diffusion process
        return x * self.s_prime(t) / self.s(t)

    def g(self, t):
        # g(sigma) in the SDE, also known as diffusion term for the forward diffusion process
        return self.s(t) * jnp.sqrt(2 * self.sigma_prime(t) * self.sigma(t))

    def denoise(self, x, sigma, F):
        # denoise function, D in the EDM paper, which shares a connection with the score function:
        # ∇_x log p(x; σ) = (D(x; σ) − x)/σ^2

        # this function includes the preconditioning and is connected to the NN objective F:
        # D_θ(x; σ) = c_skip(σ) x + c_out(σ) F_θ (c_in(σ) x; c_noise(σ))
        return self.c_skip(sigma) * x + self.c_out(sigma) * F(
            self.c_in(sigma) * x, self.c_noise(sigma)
        )

    def get_score_function(self, F):
        # score function, ∇_x log p(x; σ) = (D(x; σ) − x)/σ^2
        def score(x, t):
            sigma = self.sigma(t)
            return (self.denoise(x, sigma, F) - x) / (sigma**2)

        return score

    def denoising_loss(self, F, x, sigma): #TODO add conditioning
        # a typical trainig loop will sample t from Unif(eps, 1), then get sigma(t) and compute the loss
        # get the denoising score matching loss, as Eq. 8 of the EDM paper
        lam = self.loss_weight(sigma)
        c_out = self.c_out(sigma)
        c_in = self.c_in(sigma)
        c_noise = self.c_noise(sigma)
        c_skip = self.c_skip(sigma)
        noise = self.sample_noise(x.shape)

        loss = (
            lam
            * c_out**2
            * (F(c_in * (x + noise), c_noise) - 1 / c_out * (x - c_skip * (x + noise)))
            ** 2
        )
        # we sum the loss on any dimension that is not the batch dimentsion, and then we compute the mean over the batch dimension (the first)
        return jnp.mean(jnp.sum(loss, axis=tuple(range(1, len(x.shape)))))


class VP(AbstractSDE):
    def __init__(self, beta_min=0.1, beta_d=19.9, e_s=1e-3, e_t=1e-5, M=1000):
        super().__init__()
        self.beta_min = beta_min
        self.beta_d = beta_d
        self.e_s = e_s
        self.e_t = e_t
        self.M = M
        return

    def timesteps(self, i, N):
        return 1 + i / (N - 1) * (self.e_s - 1)

    def sigma(self, t):
        # also known as the schedule, as in tab 1 of EDM paper
        return jnp.sqrt(jnp.exp(0.5 * self.beta_d * t**2 + self.beta_min * t) - 1)

    def sigma_inv(self, sigma):
        return (
            jnp.sqrt(self.beta_min**2 + 2 * self.beta_d * jnp.log(1 + sigma**2))
            - self.beta_min
        ) / self.beta_d

    def sigma_prime(self, t):
        # also known as the schedule derivative
        return jax.grad(self.sigma)(t)

    def s(self, t):
        # also known as scaling, as in tab 1 of EDM paper
        return 1 / jnp.sqrt(jnp.exp(0.5 * self.beta_d * t**2 + self.beta_min * t))

    def s_prime(self, t):
        # also known as scaling derivative
        return jax.grad(self.s)(t)

    def c_skip(self, sigma):
        # c_skip for preconditioning
        return 1

    def c_out(self, sigma):
        # c_out for preconditioning
        return -sigma

    def c_in(self, sigma):
        # c_in for preconditioning
        return 1 / jnp.sqrt(sigma**2 + 1)

    def c_noise(self, sigma):
        # c_noise for preconditioning
        return (self.M -1)*self.sigma_inv(sigma)

    def loss_weight(self, sigma):
        return 1/sigma**2


class VE(AbstractSDE):
    def __init__(self, sigma_min=1e-3, sigma_max=15.):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        return
    
    def timesteps(self, i, N):
        return self.sigma_max**2*(self.sigma_min/self.sigma_max)**(2*i/(N-1))

    def sigma(self, t):
        # also known as the schedule, as in tab 1 of EDM paper
        return jnp.sqrt(t)

    def sigma_prime(self, t):
        return 1/(2*jnp.sqrt(t))

    def s(self, t):
        # also known as scaling, as in tab 1 of EDM paper
        return 1
    
    def s_prime(self, t):
        # also known as scaling derivative
        return 0
    
    def sample_sigma(self, key, shape):
        # sample sigma from the prior noise distribution
        rnd_uniform = jax.random.uniform(key, shape)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        return sigma

    def c_skip(self, sigma):
        # c_skip for preconditioning
       return 1

    def c_out(self, sigma):
        # c_out for preconditioning
        return sigma

    def c_in(self, sigma):
        # c_in for preconditioning
        return 1

    def c_noise(self, sigma):
        # c_noise for preconditioning
        return jnp.log(0.5*sigma)

    def loss_weight(self, sigma):
        return 1/sigma**2

#TODO
class EDM(AbstractSDE):
    def __init__(self, beta_min=0.1, beta_d=19.9):
        super().__init__()
        self.beta_min = beta_min
        self.beta_d = beta_d
        return

    @abc.abstractmethod
    def timesteps(self, i, N):
        pass

    @abc.abstractmethod
    def sigma(self, t):
        # also known as the schedule, as in tab 1 of EDM paper
        pass

    @abc.abstractmethod
    def sigma_prime(self, t):
        # also known as the schedule derivative
        pass

    @abc.abstractmethod
    def s(self, t):
        # also known as scaling, as in tab 1 of EDM paper
        pass

    @abc.abstractmethod
    def s_prime(self, t):
        # also known as scaling derivative
        pass

    @abc.abstractmethod
    def c_skip(self, sigma):
        # c_skip for preconditioning
        pass

    @abc.abstractmethod
    def c_out(self, sigma):
        # c_out for preconditioning
        pass

    @abc.abstractmethod
    def c_in(self, sigma):
        # c_in for preconditioning
        pass

    @abc.abstractmethod
    def c_noise(self, sigma):
        # c_noise for preconditioning
        pass

    @abc.abstractmethod
    def sample_noise(self, shape):
        # sample noise from the prior noise distribution
        pass

    @abc.abstractmethod
    def loss_weight(self, sigma):
        # weight for the loss function, for MLE estimation, also known as λ(σ) in the EDM paper
        pass
