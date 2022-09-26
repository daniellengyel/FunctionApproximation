"""Here we are going to summarize all the methods and make it easy to call ? """

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

import json

import os 
import sys
HOME = "/rds/general/user/dl2119/home/ICLR_Interp"
# HOME = os.environ["PATH_INTP_FOLDER"]
os.environ["PATH_INTP_FOLDER"] = HOME
sys.path.append(HOME + "/")
sys.path.append(HOME + "/DataGeneration")
sys.path.append(HOME + "/DataGeneration/CurvSampling")
from Functions import get_func
from UniformData import get_uniform_grid, get_uniform_random
from CurvSampling import sample_curv
from save_load_data import save_reparam_curv, save_sampling_hess, save_simplify_mesh, save_uniform_grid, save_uniform_sampling

if "PBS_ARRAY_INDEX" in os.environ:
    ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) - 1
else:
    ARRAY_INDEX = 0 


"""Input for all should be just the function and N i think """

def get_uniform(funcs, dims, Ns, seed, verbose=True):
    for f in funcs:
        if verbose:
            print("f", f)
        F = get_func(f)

        for dim in dims:
            if ((dim % 2) == 1) and (f == "Rosenbrock"):
                continue
            if verbose:
                print("dim", dim)
            for N in Ns:
                N = int(N)
                dim = int(dim)
                data_x_grid = get_uniform_grid(F, N, dim, seed)
                data_y_grid = F.f(data_x_grid)
                data_x_random = get_uniform_random(F, N, dim, seed)
                data_y_random = F.f(data_x_random)

                save_uniform_sampling(data_x_random, data_y_random, f, dim, N, seed)
                save_uniform_grid(data_x_grid, data_y_grid, f, dim, N, seed)

        


def get_hess_sampling(funcs, dims, Ns, verbose=True):
    with open(HOME + "/DataGeneration/CurvSampling/configs.json", "r") as f:
        configs = json.load(f)
    
    for f in funcs:
        if verbose:
            print("f", f)
        F = get_func(f)
        config = configs[f]
        num_steps = config["num_steps"]
        sig = config["sig"]
        seed = config["seed"]
        eps_barrier = config["eps_barrier"]
        curv_scaling = config["curv_scaling"]
        for i, dim in enumerate(dims):
            if verbose:
                print("dim", dim)            
            for N in Ns:
                N = int(N)
                dim = int(dim)
                if type(sig) == list:
                    xs = sample_curv(F, dim, N, num_steps, sig[i], seed, eps_barrier, curv_scaling[i], full_path=False)
                else:
                    xs = sample_curv(F, dim, N, num_steps, sig, seed, eps_barrier, curv_scaling, full_path=False)
                ys = F.f(xs)
                save_sampling_hess(xs, ys, f, dim, N, config)
    

def get_mesh_simplification():
    pass

def get_curv_reparam():
    pass




if __name__ == "__main__":
    funcs = ["Ackley", "Michi", "Rosenbrock"]
    dims = [1, 2, 3, 4, 6] #, 10]
    Ns = [10, 1e2, 1e3, 1e4, 1e5] # , 1e6]

    if ARRAY_INDEX == 0:    
        get_uniform(funcs, dims, Ns, seed=0)

    if ARRAY_INDEX > 0:
        get_hess_sampling(funcs, dims, [Ns[ARRAY_INDEX - 1]])





