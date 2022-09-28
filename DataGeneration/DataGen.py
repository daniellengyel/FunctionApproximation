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
sys.path.append(HOME + "/DataGeneration/Mesh_simplification")

from Functions import get_func
from UniformData import get_uniform_grid, get_uniform_random
from CurvSampling import sample_curv
from save_load_data import save_curv_hess_reparam, save_sampling_hess, save_mesh_simplification, save_uniform_grid, save_uniform_sampling
from QuadricsMeshSimplifcation import get_quadrics_points
from CurvatureReparam import get_curv_hess_reparam

if "PBS_ARRAY_INDEX" in os.environ:
    ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) - 1
else:
    ARRAY_INDEX = 2

def generate_all_data_confs(funcs, dims, Ns_per_dim):
    res = []
    for f in funcs:
        for d in dims:
            for N in Ns_per_dim:
                res.append({"func_name": f, "dim": d, "N": N**d})
    return res


"""Input for all should be just the function and N i think """

def get_uniform(all_data_confs, seed, verbose=True):

    for data_conf in all_data_confs:
        print(data_conf)  
        f = data_conf["func_name"]     
        F = get_func(f)
        N = int(data_conf["N"])
        dim = int(data_conf["dim"])
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
    

def get_mesh_simplification(data_confs, N_high=100**2, threshold_prct=0.01, verbose=True):
    for data_conf in data_confs:
        if data_conf["dim"] != 2:
            continue
        print(data_conf)  

        F = get_func(data_conf["func_name"])
        X_data = np.array(get_quadrics_points(F, N_high, data_conf["N"], threshold_prct)[0])[:, :2]
        y_data = F.f(X_data)
        save_mesh_simplification(X_data, y_data, data_conf["func_name"], data_conf["N"], config={"N_high": N_high, "threshold_prct": threshold_prct})

def get_save_curv_hess_reparam(data_confs, N_high=100, verbose=True):
    for data_conf in data_confs:
        if data_conf["dim"] != 1:
            continue

        F = get_func(data_conf["func_name"])

        for reparm_type in ["curv", "hess"]:
            for use_norm in [True, False]:
                X_data = get_curv_hess_reparam(F, N_high, data_conf["N"], reparm_type=reparm_type, use_norm=use_norm).reshape(-1, 1)
                y_data = F.f(X_data)
                save_curv_hess_reparam(X_data, y_data, data_conf["func_name"], data_conf["N"], reparam_type=reparm_type, use_norm=use_norm, config={"N_high": N_high})


if __name__ == "__main__":
    funcs = ["Ackley", "Michi", "Dixon"]
    dims = [1, 2]
    Ns_per_dim = [4, 6, 8, 10, 12, 16, 20, 40] # per dim

    all_data_confs = generate_all_data_confs(funcs, dims, Ns_per_dim)

    if ARRAY_INDEX == 0:    
        get_uniform(all_data_confs, seed=0)

    elif ARRAY_INDEX == 1:
        get_save_curv_hess_reparam(all_data_confs, N_high=100, verbose=True)

    # elif ARRAY_INDEX == 2:
    #     get_mesh_simplification(all_data_confs, N_high=100**2, threshold_prct=0.01, verbose=True)





