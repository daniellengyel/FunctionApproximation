import numpy as np
import jax.numpy as jnp 
from jax import jacfwd

def k_l(F, x, reparm_type="curv", use_norm=True):
    x = np.array(x).reshape(-1, 1)
    c = 1.
    if use_norm:
        c = np.linalg.norm(np.array([1, F.f1(x).reshape(-1)[0]]))

    if reparm_type == "curv":
        return np.abs(F.f2(x)/((1 + F.f1(x)**2)**(3/2.))) * c
    else:   
        return np.abs(F.f2(x)) * c

def curv_hess_parmetrization(F, N_high, reparm_type="curv", use_norm=True):
    res = []
    x_l, x_u = F.bounds[0]
    delta = (x_u - x_l)/float(N_high - 1)
    
    for i in range(N_high):
        curr = k_l(F, x_l + i*delta, reparm_type=reparm_type, use_norm=use_norm) * delta
        if len(res) == 0:
            res.append(curr)
        else:
            res.append(res[-1] + curr)
   
    return np.array(res).ravel()

def invert_param(F, curv_pts, N_high, N_low):
    x_l, x_u = F.bounds[0]
    x_low_curv_uniform = np.linspace(min(curv_pts), max(curv_pts), N_low)
    x_low_curv = []

    delta = (x_u - x_l)/float(N_high - 1)
    cur_idx = 0
    for x_c in x_low_curv_uniform:
        while curv_pts[cur_idx] < x_c:
            cur_idx += 1
        
        x_low_curv.append(x_l + delta*cur_idx)
    
    x_low_curv = np.array(x_low_curv)
    return x_low_curv.reshape(1, -1)

def get_curv_hess_reparam(F, N_high, N_low, reparm_type, use_norm):
    reparam_pts = curv_hess_parmetrization(F, N_high, reparm_type, use_norm)
    return np.array(invert_param(F, reparam_pts, N_high, N_low)).reshape(-1, 1)




