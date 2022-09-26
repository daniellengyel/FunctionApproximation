import numpy as np
import jax.numpy as jnp 
from jax import jacfwd

def k_l(F, x):
    x = np.array(x).reshape(-1, 1)
    return np.abs(F.f2(x)/((1 + F.f1(x)**2)**(3/2.))) * np.linalg.norm(np.array([1, F.f1(x).reshape(-1)[0]]))

def curvature_parmetrization(F, N, x_l, x_u):
    res = []
    
    delta = (x_u - x_l)/float(N - 1)
    
    for i in range(N):
        curr = k_l(F, x_l + i*delta) * delta
        if len(res) == 0:
            res.append(curr)
        else:
            res.append(res[-1] + curr)
            
    return np.array(res).ravel()

        
    # uniform curv 
N_curv = 200
curv = curvature_parmetrization(F, N_curv, x_l, x_u)


delta = (x_u - x_l)/float(N_curv -1 )



N_sample = 20
x_l, x_u = -2, 2
x_low_curv_uniform = np.linspace(min(curv), max(curv), N_sample)
x_low_curv = []

cur_idx = 0
for x_c in x_low_curv_uniform:
    while curv[cur_idx] < x_c:
        cur_idx += 1
        
    x_low_curv.append(x_l + delta*cur_idx)
    
x_low_curv = np.array(x_low_curv)
y_low_curv = F.f(x_low_curv.reshape(-1, 1)).ravel()

plt.plot(x_low_curv, y_low_curv)
plt.plot(x, y)


