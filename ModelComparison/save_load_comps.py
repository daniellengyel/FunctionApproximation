import torch
import json 

import pandas as pd

import os 
import sys
from pathlib import Path
HOME = Path(os.environ["PATH_INTP_FOLDER"])

from save_load_data import load_fast, get_all_data_configurations

from pathlib import Path

def save_comps(comps, N_test, seed_test, dim, N, func, data_gen_method):
    dim = int(dim)
    N = int(N)
    path = HOME / "ModelComps" / str(func) / "dim_{}".format(str(dim)) / "N_{}".format(str(N)) / data_gen_method

    
    for c in comps:
        if comps[c] is None:
            continue

        curr_path = path / c
        curr_path.mkdir(parents=True, exist_ok=True)

        if "train_mean_err" in comps[c]:
            with open(curr_path / "testLoss_{}_trainLoss_{}".format(comps[c]["test_mean_err"], comps[c]["train_mean_err"]), "w") as f:
                json.dump(comps[c], f)
        else:
            with open(curr_path / "testLoss_{}".format(comps[c]["test_mean_err"]), "w") as f:
                json.dump(comps[c], f)

def undo_model_tag(tag):
    tag_split = tag.split("_")
    res = {}
    i = 0
    if tag_split[0] == "nn":
        i = 1
        res["model"] = "nn"
    else:
        res["model"] = "rbf"
        res["kernel"] = tag_split[1]
        if tag_split[1] == "thin":
            i = 4
        else:
            i = 2

    while i < len(tag_split):
        if tag_split[i] == "opt" and tag_split[i + 1] == "SGD":
            res[tag_split[i]] = tag_split[i+1]
            res["momentum"] = tag_split[i+2]
            i += 3
        else:
            res[tag_split[i]] = tag_split[i+1]
            i += 2

    return res

def undo_loss_tag(tag):
    tag_split = tag.split("_")
    res = {}
    i = 0
    while i < len(tag_split):
        res[tag_split[i]] = tag_split[i + 1]
        i += 2

    return res 
    

def load_all_comps():

    path = HOME / "ModelComps"

    all_res = []

    all_data_confs = get_all_data_configurations()

    for data_conf in all_data_confs:
        func = data_conf["func_name"]
        dim = data_conf["dim"]
        dim_dir = "dim_{}".format(int(dim))
        N = data_conf["N"]
        N_dir = "N_{}".format(int(N))
        data_gen_method = data_conf["data_gen_method"]
        if not os.path.isdir(str(path / func / dim_dir / N_dir / data_gen_method)):
            continue
            
        for model_tag in os.listdir(path / func / dim_dir / N_dir / data_gen_method):
            res = {k: v for k, v in data_conf.items()} 
            model_tag_dir = undo_model_tag(model_tag)
            for k in model_tag_dir:
                res[k] = model_tag_dir[k]

            loss_tag = os.listdir(path / func / dim_dir / N_dir / data_gen_method / model_tag)[0]
            loss_tag_dir = undo_loss_tag(loss_tag)
            for k in loss_tag_dir:
                res[k] = float(loss_tag_dir[k])

            all_res.append(res)

    return pd.DataFrame(all_res).fillna(0)


                                    