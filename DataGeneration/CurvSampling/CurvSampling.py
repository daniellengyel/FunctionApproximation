from jax.config import config
config.update("jax_enable_x64", True)

import jax.numpy as jnp 
from jax import jacfwd, grad
import jax.random as jrandom
from jax import jit
from jax.lax import fori_loop

import os 
import sys
HOME = "/rds/general/user/dl2119/home/ICLR_Interp" # Path(os.environ["PATH_INTP_FOLDER"])
sys.path.append(HOME + "/" + "DataGeneration")
from UniformData import get_uniform_random

from Barriers import SquarePolytopeBarrier

def create_cube_barrier(bounds, dim):
    ws = []
    bs = []
    
    I = jnp.eye(dim)
    
    for i in range(dim):
        ws.append(I[i])
        ws.append(-I[i])
        
        if len(bounds) != dim:
            b = bounds[0]
        else:
            b = bounds[i]
                    
        bs.append(b[1])
        bs.append(-b[0])
        
    return SquarePolytopeBarrier(ws, bs)

def curv(F, x):
    
    x = jnp.array(x).reshape(1, -1)

    H = F.f2(x).reshape(x.shape[1], x.shape[1])

    return jnp.linalg.norm(H, "fro") / x.shape[1]

def sample_curv(F, dim, N, num_steps, sig, seed, eps_barrier, curv_scaling, full_path=True):
    
    barrier = create_cube_barrier(F.bounds, dim)
    
    jrandom_key = jrandom.PRNGKey(seed)
    
    jrandom_key, subkey = jrandom.split(jrandom_key)
    xs = [get_uniform_random(F, N, dim, seed + 1)]
    
    curv_f1 = grad(lambda x: curv(F, x))
    
    def update_step(curr_xs):
        grad_update = jnp.zeros(shape=(N, dim,))
        def body_fun(i, val):
            val = val.at[i].set(curv_f1(curr_xs[i]))
            return val
        grad_update = fori_loop(0, N, body_fun, grad_update)
        return curv_scaling*grad_update - eps_barrier*barrier.f1(curr_xs) + sig*jrandom.normal(subkey, shape=(N, dim))

    jitted_update = jit(update_step)

    for _ in range(num_steps):
        jrandom_key, subkey = jrandom.split(jrandom_key)
        xs_delta = jitted_update(xs[-1])

        if full_path:
            xs.append(xs[-1] + xs_delta)
        else:
            xs = [xs[-1] + xs_delta]

    if full_path:
        return xs
    else:
        return xs[0]
