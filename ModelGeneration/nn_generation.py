
import os, sys 
import numpy as np
from pathlib import Path 
HOME = "/rds/general/user/dl2119/home/ICLR_Interp" # Path(os.environ["PATH_INTP_FOLDER"])
os.environ["PATH_INTP_FOLDER"] = HOME
HOME = Path(HOME)
sys.path.append(str(HOME))
sys.path.append(str(HOME / "DataGeneration"))

from save_load_data import load_fast, get_all_data_configurations
from nn_train import train
from save_load_model import save_nn, get_nn_tag, get_all_net_tags

if "PBS_ARRAY_INDEX" in os.environ:
    ARRAY_INDEX = int(os.environ["PBS_ARRAY_INDEX"]) - 1
    TOTAL_ARRAY = 384
else:
    ARRAY_INDEX = 384
    TOTAL_ARRAY = 384


import tracemalloc


def normalize(X_data, y_data):
    x_max, x_min = np.max(X_data), np.min(X_data)
    y_max, y_min = np.max(y_data), np.min(y_data)
    X_data = (X_data - (x_max + x_min)/2.) * 5. / ((x_max - x_min)/2.)
    y_data = (y_data - (y_max + y_min)/2.) * 5. / ((y_max - y_min)/2.)

    return X_data, y_data


def generate_nns(X_train, y_train, func_name, dim, N, data_gen_name):
    num_epochs = int(500)
    # num_steps = 1e5
    eps = 1e-6
    batch_sizes = [4, 16, 64]

    depth_widths = [{'depth': 2, 'width': 2},
                    {'depth': 2, 'width': 4},
                    {'depth': 2, 'width': 8},
                    {'depth': 2, 'width': 16},
                    {'depth': 4, 'width': 4},
                    {'depth': 4, 'width': 8},
                    {'depth': 4, 'width': 16}]


    optimizers = {"Adam": {"lr": [0.001, 0.0001, 1e-5], "momentum": [0.]}, "SGD": {"lr": [1e-2, 1e-3, 1e-4, 1e-5], "momentum": [0., 0.9]}}

    weight_decays = [0, 1e-3, 1]

    all_net_tags = get_all_net_tags(func_name, dim, N, data_gen_method)

    nn_num = 0
    for depth_width in depth_widths:
        for optimizer in optimizers:
            for batch_size in batch_sizes:
                # How often will i see each elem is the number of epochs
                # num_epochs = (num_steps * batch_size)/N
                # num_epochs = max(num_epochs, 10) # want to see every data point at least 5-times
                # num_epochs = min(num_epochs, 1e3) # no more than 1k epochs
                # num_epochs = int(num_epochs)
                for lr in optimizers[optimizer]["lr"]:
                    for momentum in optimizers[optimizer]["momentum"]:
                        for weight_decay in weight_decays:
                            if (nn_num % TOTAL_ARRAY) == ARRAY_INDEX:
                                depth = depth_width["depth"]
                                width = depth_width["width"]
                                net_tag = get_nn_tag(depth, width, optimizer, lr, batch_size, weight_decay, momentum)
                                if net_tag in all_net_tags:
                                    continue
                                    
                                print("NN num", nn_num)
                                print(depth)
                                print(width)
                                print(lr)
                                print(num_epochs)
                                print(batch_size)
                                
                                nn, loss_hist = train(dim, None, X_train, y_train, num_epochs, eps, width, depth, optimizer, lr, batch_size, weight_decay, momentum, verbose=False)
                                save_nn(nn, loss_hist, func_name, dim, N, data_gen_name, depth, width, optimizer, lr, batch_size, weight_decay, momentum)
                            nn_num += 1


def load_and_call(func, dim, N, data_gen_method):
    X_data, y_data = load_fast(func, dim, N, data_gen_method)
    X_data, y_data = normalize(X_data, y_data)
    
    generate_nns(X_data, y_data, func, dim, N, data_gen_method)

if __name__ == "__main__":


    curr_exp_n = 0

    all_data_confs = get_all_data_configurations()

    for data_conf in all_data_confs:
        func = data_conf["func_name"]
        dim = data_conf["dim"]
        N = data_conf["N"]
        data_gen_method = data_conf["data_gen_method"]
        load_and_call(func, dim, N, data_gen_method)

        print("Exp N", curr_exp_n)
        # if (curr_exp_n % TOTAL_ARRAY) == ARRAY_INDEX:
        #     print("Generating...")
        
        # print()
        curr_exp_n += 1

                    