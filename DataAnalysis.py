import jax.numpy as jnp
from jax.lax import fori_loop
from jax import jit

from functools import partial


@partial(jit, static_argnames=['F'])
def curv(F, X):
    H = F.f2(X)

    curvs = jnp.zeros(shape=(X.shape[0],))
    def body_fun(i, val):
        val = val.at[i].set(jnp.linalg.norm(H[i], "fro") / x.shape[1])
        return val
    curvs = fori_loop(0, X.shape[0], body_fun, curvs)   
    return curvs