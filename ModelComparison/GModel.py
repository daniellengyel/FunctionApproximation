import sys, os
import ujson as json
from pathlib import Path 

HOME = "/rds/general/user/dl2119/home/ICLR_Interp" # Path(os.environ["PATH_INTP_FOLDER"])
os.environ["PATH_INTP_FOLDER"] = HOME
HOME = Path(HOME)
sys.path.append(str(HOME))
sys.path.append(str(HOME / "DataGeneration"))
sys.path.append(str(HOME / "ModelGeneration"))
sys.path.append(str(HOME / "ModelComparison"))

from torch import Tensor
from scipy.interpolate import RBFInterpolator
import numpy as np

from save_load_data import load_fast, get_all_data_configurations
from save_load_model import fast_load_nn
from save_load_comps import undo_model_tag


class GModel():

    def __init__(self, func_name, dim, N, data_gen_method, tag):
        
        X_train, y_train = load_fast(func_name, dim, N, data_gen_method)
        
        self.func_name = func_name
        self.dim = dim
        self.N = N
        self.data_gen_method = data_gen_method
        self.tag_dict = undo_model_tag(tag)

        self._x_max, self._x_min = np.max(X_train), np.min(X_train)
        self._y_max, self._y_min = np.max(y_train), np.min(y_train)

        self._model = self._get_model(X_train, y_train)
        
        
    def f(self, X):
        X = (X - (self._x_max + self._x_min)/2.) * 5. / ((self._x_max - self._x_min)/2.)
        out = self._model(X)
        return out * (((self._y_max - self._y_min)/2.) / 5.) + (self._y_max + self._y_min)/2.
    
    def train_error(self):
        if self.tag_dict['model'] == "rbf":
            return 0.
        
        else:
            X_train, y_train = load_fast(self.func_name, self.dim, self.N, self.data_gen_method)
            out = self.f(X_train)
            return jnp.mean((y_train - out)**2) # * (5. / ((self._y_max - self._y_min)/2.))**2 uncomment to get loss as given during training. 
            
    
    def _get_model(self, X_train, y_train):
        tag_dict = self.tag_dict
        X_train = (X_train - (self._x_max + self._x_min)/2.) * 5. / ((self._x_max - self._x_min)/2.)
        y_train = (y_train - (self._y_max + self._y_min)/2.) * 5. / ((self._y_max - self._y_min)/2.)
        
        if tag_dict["model"] == "rbf":
            epsilon = 1.
            if "epsilon" in tag_dict:
                epsilon = tag_dict["epsilon"]

            kernel = tag_dict["kernel"]
            if kernel == "thin":
                kernel = "thin_plate_spline"
            rbf = RBFInterpolator(X_train, y_train, kernel=kernel, neighbors=int(float(tag_dict["neighbors"])), epsilon=epsilon)
            return rbf
        else:
            momentum = 0
            if "momentum" in tag_dict:
                momentum = tag_dict["momentum"]
                
            net = fast_load_nn(self.func_name, self.dim, self.N, self.data_gen_method, 
                               tag_dict["depth"], tag_dict["width"], tag_dict["opt"], tag_dict["lr"], 
                               tag_dict["bs"], tag_dict["l2p"], )
                               
            return lambda X: net(Tensor(X)).detach().numpy().ravel()
        