import jax.numpy as jnp
import jax.random as jrandom

def get_uniform_random(F, N, dim, seed):
    jrandom_key = jrandom.PRNGKey(seed)

    X_init = jrandom.uniform(jrandom_key, shape=(N, dim)) - 1/2.
    X = X_init * (F.bounds[:, 1] - F.bounds[:, 0]) + (F.bounds[:, 1] + F.bounds[:, 0])/2.
    return X

def get_uniform_grid(F, N, dim, seed):
    jrandom_key = jrandom.PRNGKey(seed)
    
    N_dim = int(N**(1/dim))
    if N_dim**dim != N:
        N_dim += 1

    X_dim = jnp.linspace(-1/2., 1/2., N_dim)
    X_init = jnp.array([x.ravel() for x in jnp.meshgrid(*(X_dim for _ in range(dim)))]).T
    jrandom_key, subkey = jrandom.split(jrandom_key)
    X_init = jrandom.choice(subkey, X_init, shape=(N,), replace=False)
    X = X_init * (F.bounds[:, 1] - F.bounds[:, 0]) + (F.bounds[:, 1] + F.bounds[:, 0])/2.
    return X