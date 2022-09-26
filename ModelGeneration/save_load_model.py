import torch
import json 

import os 
import sys
from pathlib import Path
HOME = Path(os.environ["PATH_INTP_FOLDER"])

from pathlib import Path

from Nets import SimpleNet

def get_nn_tag(depth, width, optimizer, lr, batch_size, weight_decay, momentum=0):
    if optimizer == "SGD":
        optimizer += "_{}".format(momentum)

    nn_tag = "nn_depth_{}_width_{}_opt_{}_lr_{}_bs_{}_l2p_{}".format(depth, width, optimizer, lr, batch_size, weight_decay)
    return nn_tag

def save_nn(net, loss_hist, func_name, dim, N, data_gen_name, depth, width, optimizer, lr, batch_size, weight_decay, momentum=0):
    dim = int(dim)
    N = int(N)
    path = HOME / "Models" / str(func_name) / "dim_{}".format(str(dim)) / "N_{}".format(str(N)) / data_gen_name

    dir_name = get_nn_tag(depth, width, optimizer, lr, batch_size, weight_decay, momentum)
    path = path / dir_name
    Path(path).mkdir(parents=True, exist_ok=True)

    torch.save(net.state_dict(), path / "model_{0:.4f}.pt".format(loss_hist[-1]))
    with open(path / "loss_hist_{0:.4f}.json".format(loss_hist[-1]), "w") as f:
        json.dump([float(l) for l in loss_hist], f)

def fast_load_nn(func_name, dim, N, data_gen_name, depth, width, optimizer, lr, batch_size, weight_decay, momentum=0):
    dim = int(dim)
    N = int(N)
    path = HOME / "Models" / str(func_name) / "dim_{}".format(str(dim)) / "N_{}".format(str(N)) / data_gen_name


    dir_name = get_nn_tag(depth, width, optimizer, lr, batch_size, weight_decay, momentum)
    path = path / dir_name
    
    if not path.is_dir():
        return None

    model = SimpleNet(dim, 1, int(width), int(depth))
    for file_name in os.listdir(path):
        if "model" in file_name:
            model.load_state_dict(torch.load(path / file_name))
            model.eval()
            return model 

    

def get_all_net_tags(func_name, dim, N, data_gen_method):
    path = HOME / "Models" / func_name / "dim_{}".format(dim) / "N_{}".format(N) / data_gen_method
    if not os.path.isdir(path):
        return None
    return os.listdir(path)