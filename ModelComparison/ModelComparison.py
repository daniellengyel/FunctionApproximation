import sys, os
import ujson as json
from pathlib import Path 
HOME = "/rds/general/user/dl2119/home/ICLR_Interp" # Path(os.environ["PATH_INTP_FOLDER"])
os.environ["PATH_INTP_FOLDER"] = HOME
HOME = Path(HOME)
sys.path.append(str(HOME))
sys.path.append(str(HOME / "DataGeneration"))
sys.path.append(str(HOME / "ModelGeneration"))

from save_load_data import load_fast
from save_load_model import get_nn_tag, fast_load_nn
from save_load_comps import save_comps

from torch import Tensor

from scipy.interpolate import RBFInterpolator

from Functions import get_func

import jax.random as jrandom
import numpy as np

import torch
from torch import Tensor

if "PBS_ARRAY_INDEX" in os.environ:
    ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) - 1
    TOTAL_ARRAY = 210
else:
    ARRAY_INDEX = 10
    TOTAL_ARRAY = 210

def normalize(X_data, y_data, X_test, y_test):
    x_max, x_min = np.max(X_data), np.min(X_data)
    y_max, y_min = np.max(y_data), np.min(y_data)
    X_data = (X_data - (x_max + x_min)/2.) * 5. / ((x_max - x_min)/2.)
    y_data = (y_data - (y_max + y_min)/2.) * 5. / ((y_max - y_min)/2.)

    X_test = (X_test - (x_max + x_min)/2.) * 5. / ((x_max - x_min)/2.)
    y_test = (y_test - (y_max + y_min)/2.) * 5. / ((y_max - y_min)/2.)

    return X_data, y_data, X_test, y_test


def get_rbf_tag(rbf_kernel, neighbors, epsilon=None):
    rbf_tag = "rbf_{}_neighbors_{}".format(rbf_kernel, neighbors)
    if epsilon is not None:
        rbf_tag += "_eps_{}".format(epsilon)
    return rbf_tag

def rbf_loss(X_train, y_train, X_test, y_test):
    rbf_configs =  {"thin_plate_spline": {"neighbors": 1e3},
                    "linear": {"neighbors": 1e3}, 
                    "gaussian": {"neighbors": 1e3, "epsilon": [1, 5, 10]}}

    def get_rbf_loss(rbf):
        test_out = rbf(X_test)
        test_errs = np.abs(y_test - test_out)
        test_mean_err = np.mean(test_errs**2)
        return {"test_errs": test_errs.tolist(), "test_mean_err": test_mean_err.tolist()}
        
    res = {}

    for rbf_kernel in rbf_configs:
        neighbors = rbf_configs[rbf_kernel]["neighbors"]
        if rbf_kernel == "gaussian":
            for epsilon in rbf_configs[rbf_kernel]["epsilon"]:
                rbf = RBFInterpolator(X_train, y_train, kernel=rbf_kernel, neighbors=neighbors, epsilon=epsilon)
                rbf_tag = get_rbf_tag(rbf_kernel, neighbors, epsilon)
                res[rbf_tag] = get_rbf_loss(rbf)
        else:
            rbf = RBFInterpolator(X_train, y_train, kernel=rbf_kernel, neighbors=neighbors)
            rbf_tag = get_rbf_tag(rbf_kernel, neighbors)
            res[rbf_tag] = get_rbf_loss(rbf)

    return res



def nn_loss(func_name, dim, N, data_gen_method, X_train, y_train, X_test, y_test):
    X_train = Tensor(X_train)
    y_train = Tensor(y_train).unsqueeze(1)

    X_test = Tensor(X_test)
    y_test = Tensor(y_test).unsqueeze(1)

    batch_sizes = [16, 64, 256]

    depth_widths = [{'depth': 4, 'width': 4},
                    {'depth': 4, 'width': 32},
                    {'depth': 4, 'width': 128},
                    {'depth': 8, 'width': 32},
                    {'depth': 8, 'width': 128},
                    {'depth': 16, 'width': 32},
                    {'depth': 16, 'width': 128}]


    optimizers = {"Adam": {"lr": [0.001, 0.0001, 1e-5], "momentum": [0.]}, "SGD": {"lr": [1e-2, 1e-3, 1e-4, 1e-5], "momentum": [0., 0.9]}}

    weight_decays = [0, 1e-3, 1]

    criterion = torch.nn.MSELoss(reduction="mean")

    nn_num = 0
    res = {}
    for depth_width in depth_widths:
        for optimizer in optimizers:
            for batch_size in batch_sizes:
                for lr in optimizers[optimizer]["lr"]:
                    for momentum in optimizers[optimizer]["momentum"]:
                        for weight_decay in weight_decays:
                            print(nn_num)
                            depth = depth_width["depth"]
                            width = depth_width["width"]
                            net = fast_load_nn(func_name, dim, N, data_gen_method, depth, width, optimizer, lr, batch_size, weight_decay, momentum)
                            nn_tag = get_nn_tag(depth, width, optimizer, lr, batch_size, weight_decay, momentum)
                            if net is not None:
                                train_out = net(X_train)
                                test_out = net(X_test)
                                train_errs = torch.abs(y_train - train_out)
                                test_errs = torch.abs(y_test - test_out)

                                train_errs = train_errs.detach().numpy().tolist()
                                test_errs = test_errs.detach().numpy().tolist()

                                train_mean_err = float(criterion(train_out, y_train))
                                test_mean_err = float(criterion(test_out, y_test))
                                res[nn_tag] = {"train_errs": train_errs, "test_errs": test_errs,
                                                "train_mean_err": train_mean_err, "test_mean_err": test_mean_err}
                            else:
                                res[nn_tag] = None

                            nn_num += 1

    return res


if __name__ == "__main__":
    funcs = ["Schwefel", "Ackley", "Michi", "Rosenbrock"]
    dims = [1, 2, 3, 4, 6, 8, 10]
    Ns = [1e2, 1e3, 1e4, 1e5] #, 1e6]
    data_gen_methods = ["UniformRandom", "UniformGrid", "HessSampling"]

    curr_exp_n = 0

    N_test = int(1e3)
    seed_test = 0
    jrandom_key = jrandom.PRNGKey(seed_test)
    for func_name in funcs:
        F = get_func(func_name)
        for dim in dims:
            if func_name == "Rosenbrock" and ((dim % 2) == 1):
                continue
            
            X_test = np.array(jrandom.uniform(jrandom_key, minval=F.bounds[:, 0], maxval=F.bounds[:, 1],shape=(N_test, dim)))
            y_test = np.array(F.f(X_test))
            for N in Ns:
                for data_gen_method in data_gen_methods:
                    print("Exp num", curr_exp_n)
                    if (curr_exp_n % TOTAL_ARRAY) == ARRAY_INDEX:
                        X_train, y_train = load_fast(func_name, dim, N, data_gen_method)
                        X_train, y_train, X_test, y_test = normalize(X_train, y_train, X_test, y_test)
                        
                        nn_comps = nn_loss(func_name, dim, N, data_gen_method, X_train, y_train, X_test, y_test)
                        rbf_comps = rbf_loss(X_train, y_train, X_test, y_test)
        
                        save_comps(nn_comps, N_test, seed_test, dim, N, func_name, data_gen_method)
                        save_comps(rbf_comps, N_test, seed_test, dim, N, func_name, data_gen_method)

                    curr_exp_n += 1

