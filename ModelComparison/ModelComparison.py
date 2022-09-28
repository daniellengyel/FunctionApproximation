import sys, os
import ujson as json
from pathlib import Path 
HOME = "/rds/general/user/dl2119/home/ICLR_Interp" # Path(os.environ["PATH_INTP_FOLDER"])
os.environ["PATH_INTP_FOLDER"] = HOME
HOME = Path(HOME)
sys.path.append(str(HOME))
sys.path.append(str(HOME / "DataGeneration"))
sys.path.append(str(HOME / "ModelGeneration"))

from save_load_data import load_fast, get_all_data_configurations
from save_load_model import get_nn_tag, fast_load_nn, get_all_net_tags
from save_load_comps import save_comps

from Functions import get_func

import jax.random as jrandom
import numpy as np

from GModel import GModel

if "PBS_ARRAY_INDEX" in os.environ:
    ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) - 1
    TOTAL_ARRAY = 205
else:
    ARRAY_INDEX = 0
    TOTAL_ARRAY = 205


def get_rbf_tag(rbf_kernel, neighbors, epsilon=None):
    rbf_tag = "rbf_{}_neighbors_{}".format(rbf_kernel, neighbors)
    if epsilon is not None:
        rbf_tag += "_eps_{}".format(epsilon)
    return rbf_tag


def get_rbf_tags():
    rbf_configs =  {"thin_plate_spline": {"neighbors": 1e3},
                    "linear": {"neighbors": 1e3}, 
                    "gaussian": {"neighbors": 1e3, "epsilon": [1, 5, 10]}}
    res = []
    for rbf_kernel in rbf_configs:
        neighbors = rbf_configs[rbf_kernel]["neighbors"]
        if rbf_kernel == "gaussian":
            for epsilon in rbf_configs[rbf_kernel]["epsilon"]:
                res.append(get_rbf_tag(rbf_kernel, neighbors, epsilon))
        else:
            res.append(get_rbf_tag(rbf_kernel, neighbors))
    
    return res

def compute_loss(model, X_train, y_train, X_test, y_test):

    test_out = model.f(X_test)
    test_errs = (y_test - test_out)**2
    test_mean_err = np.mean(test_errs)

    res = {"test_errs": test_errs.tolist(), "test_mean_err": float(test_mean_err)}

    if model.tag_dict['model'] == "nn":
        train_out = model.f(X_train)
        train_errs = (y_train - train_out)**2
        train_mean_err = np.mean(train_errs)
        res["train_errs"] = train_errs.tolist()
        res["train_mean_err"] = float(train_mean_err)

    return res


def rbf_loss(func_name, dim, N, data_gen_method, X_train, y_train, X_test, y_test): 
    res = {}

    for rbf_tag in get_rbf_tags():
        rbf_model = GModel(func_name, dim, N, data_gen_method, rbf_tag)
        res[rbf_tag] = compute_loss(rbf_model, X_train, y_train, X_test, y_test)
    return res


def nn_loss(func_name, dim, N, data_gen_method, X_train, y_train, X_test, y_test):
    res = {}

    for model_tag in get_all_net_tags(func_name, dim, N, data_gen_method):
        net_model = GModel(func_name, dim, N, data_gen_method, model_tag)
        res[model_tag] = compute_loss(net_model, X_train, y_train, X_test, y_test)
    return res

if __name__ == "__main__":

    curr_exp_n = 0

    N_test = int(1e3)
    seed_test = 1
    jrandom_key = jrandom.PRNGKey(seed_test)

    all_data_confs = get_all_data_configurations()
    for data_conf in all_data_confs:
        func_name = data_conf["func_name"]
        dim = data_conf["dim"]
        N = data_conf["N"]
        data_gen_method = data_conf["data_gen_method"]


        print("Exp num", curr_exp_n)
        if (curr_exp_n % TOTAL_ARRAY) == ARRAY_INDEX:
            print("Comparing...")
            F = get_func(func_name)
            X_test = np.array(jrandom.uniform(jrandom_key, minval=F.bounds[:, 0], maxval=F.bounds[:, 1],shape=(N_test, dim)))
            y_test = np.array(F.f(X_test))
            X_train, y_train = load_fast(func_name, dim, N, data_gen_method)
            
            nn_comps = nn_loss(func_name, dim, N, data_gen_method, X_train, y_train, X_test, y_test)
            rbf_comps = rbf_loss(func_name, dim, N, data_gen_method, X_train, y_train, X_test, y_test)

            save_comps(nn_comps, N_test, seed_test, dim, N, func_name, data_gen_method)
            save_comps(rbf_comps, N_test, seed_test, dim, N, func_name, data_gen_method)

        curr_exp_n += 1
