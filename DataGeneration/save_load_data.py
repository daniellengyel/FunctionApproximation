import json 
import numpy as np

# Fix for curv sampling that not the whole path is returned
import os 
import sys
from pathlib import Path
HOME = Path("/rds/general/user/dl2119/home/ICLR_Interp") #os.environ["PATH_INTP_FOLDER"])
sys.path.append(str(HOME))
sys.path.append(str(HOME / "DataGeneration"))


import shutil 

def save_files(f, dim, N, data_gen_method, data, file_names):
    dim = int(dim)
    N = int(N)
    path = HOME / "Dataset" / str(f) / "dim_{}".format(str(dim)) / "N_{}".format(str(N)) / data_gen_method
    Path(path).mkdir(parents=True, exist_ok=True)

    for i in range(len(data)):
        if isinstance(data[i], np.ndarray):
            with open(path / (file_names[i] + ".npy"), "wb") as f:
                np.save(f, data[i])
        else:
            with open(path / (file_names[i] + ".json"), "w") as f:
                json.dump(data[i], f)


def save_curv_hess_reparam(data_x, data_y, func_name, N, reparam_type, use_norm, config):
    save_files(func_name, 1, N, "Reparam_{}_useNorm_{}".format(reparam_type, use_norm), [np.array(data_x), np.array(data_y), config], ["X_data", "y_data", "config"])

def save_mesh_simplification(data_x, data_y, func_name, N, config):
    save_files(func_name, 2, N, "MeshSimplify", [np.array(data_x), np.array(data_y), config], ["X_data", "y_data", "config"])

def save_uniform_sampling(data_x, data_y, func_name, dim, N, seed):
    save_files(func_name, dim, N, "UniformRandom", [np.array(data_x), np.array(data_y), {"seed": seed}], ["X_data", "y_data", "config"])

def save_uniform_grid(data_x, data_y, func_name, dim, N, seed):
    save_files(func_name, dim, N, "UniformGrid", [np.array(data_x), np.array(data_y), {"seed": seed}], ["X_data", "y_data", "config"])

def save_sampling_hess(data_x, data_y, func_name, dim, N, config):
    save_files(func_name, dim, N, "HessSampling", [np.array(data_x), np.array(data_y), config], ["X_data", "y_data", "config"])

def load_fast(func_name, dim, N, data_gen_method):
    dim = int(dim)
    N = int(N)
    path = HOME / "Dataset" / func_name / "dim_{}".format(dim) / "N_{}".format(N)
    if not os.path.isdir(path):
        return None, None

    to_load = data_gen_method
    # for curr_gen_method in os.listdir(path):
    #     if curr_gen_method.split("_")[0] == data_gen_method:
    #         to_load = curr_gen_method

    if to_load not in os.listdir(path):
        return None, None

    with open(path / to_load / "X_data.npy", "rb") as f:
        X_data = np.load(f)

    with open(path / to_load / "y_data.npy", "rb") as f:
        y_data = np.load(f)

    return X_data, y_data



def get_all_data_configurations():
    path = HOME / "Dataset"

    res = []
    for func in os.listdir(path):
        for dim_dir in os.listdir(path / func):
            for N_dir in os.listdir(path / func / dim_dir):
                dim = int(dim_dir.split("_")[1])
                N = int(N_dir.split("_")[1])
                for data_gen_method in os.listdir(path / func / dim_dir / N_dir):
                    res.append({"func_name": func, "dim": dim, "N": N, "data_gen_method": data_gen_method})

    return res


if __name__ == "__main__":
    path = HOME / "Dataset"

    res = []
    for func in os.listdir(path):
        for dim_dir in os.listdir(path / func):
            for N_dir in os.listdir(path / func / dim_dir):
                dim = int(dim_dir.split("_")[1])
                N = int(N_dir.split("_")[1])
                for data_gen_method in os.listdir(path / func / dim_dir / N_dir):
                    res.append({"func_name": func, "dim": dim, "N": N, "data_gen_method": data_gen_method})
                    # if "Reparam" in data_gen_method:
                    #     shutil.rmtree(str(path / func / dim_dir / N_dir / data_gen_method))