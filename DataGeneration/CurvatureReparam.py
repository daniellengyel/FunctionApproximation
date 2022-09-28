import numpy as np
import jax.numpy as jnp 
from jax import jacfwd

def k_l(F, x):
    x = np.array(x).reshape(-1, 1)
    return np.abs(F.f2(x)/((1 + F.f1(x)**2)**(3/2.))) * np.linalg.norm(np.array([1, F.f1(x).reshape(-1)[0]]))

def curvature_parmetrization(F, N_high):
    res = []
    x_l, x_u = F.bounds[0]
    delta = (x_u - x_l)/float(N_high - 1)
    
    for i in range(N_high):
        curr = k_l(F, x_l + i*delta) * delta
        if len(res) == 0:
            res.append(curr)
        else:
            res.append(res[-1] + curr)
   
    return np.array(res).ravel()

def invert_curvs(F, curv_pts, N_high, N_low):
    x_l, x_u = F.bounds[0]
    x_low_curv_uniform = np.linspace(min(curv_pts), max(curv_pts), N_high)
    x_low_curv = []

    delta = (x_u - x_l)/float(N_high - 1)
    cur_idx = 0
    for x_c in x_low_curv_uniform:
        while curv_pts[cur_idx] < x_c:
            cur_idx += 1
        
        x_low_curv.append(x_l + delta*cur_idx)
    
    x_low_curv = np.array(x_low_curv)
    return x_low_curv.reshape(1, -1)

def get_curvature_reparam(F, N_high, N_low):
    curv_pts = curvature_parmetrization(F, N_high)
    return invert_curvs(F, curv_pts, N_high, N_low)




