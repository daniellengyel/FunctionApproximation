"""Here we define the barrier functions and at the same time the domain."""

import jax.numpy as jnp



class SquarePolytopeBarrier:

    def __init__(self, ws, bs, square_factor=10):
        """ws.shape = (N, d), bs.shape = (N)"""
        self.ws = jnp.array(ws)
        self.bs = jnp.array(bs)
        self.dim = len(ws[0])
        self.square_factor = square_factor

    # @partial(jit, static_argnums=(0,))
    def _get_dists(self, xs):
        """We consider the sum of log barrier (equivalent to considering each barrier to be a potential function).
        Distance to a hyperplane w.x = b is given by | w.x/|w| - b/|w| |. We consider the absolute value of this, which follows the assumption that if we are on the a side of the hyperplane we stay there. 
        However, the signs tell us whether we are on the side of the hyperplane which is closer to the origin. If the sign is negative, then we are closer."""
        """We assume that ws points outside of the region. So dists positive if outside."""
        xs_len_along_ws = xs.dot(self.ws.T)/jnp.linalg.norm(self.ws, axis=1)
        hyperplane_dist = self.bs/jnp.linalg.norm(self.ws, axis=1)
        dists = xs_len_along_ws - hyperplane_dist # dists.shape = (N_x, N_ws)
        signs = jnp.sign(dists)#2*(dists * jnp.sign(hyperplane_dist) > 0) - 1
        return jnp.abs(dists), signs
    
    # @partial(jit, static_argnums=(0,))
    def dir_dists(self, xs, dirs):
        # we get the distance of the direction to every boundary (if parallel we have infty). We have w.(x0 + td) = b. Hence, t = (b - w.x0)/(w.d). So t is the scale to apply to d to get to the hyperplane. 
        xs_len_along_ws = xs.dot(self.ws.T)/(dirs.dot(self.ws.T))
        hyperplane_dist = self.bs/(dirs.dot(self.ws.T))
        dists = xs_len_along_ws - hyperplane_dist # dists.shape = (N_x, N_ws)
        signs = jnp.sign(dists)#2*(dists * jnp.sign(hyperplane_dist) > 0) - 1
        return jnp.abs(dists), signs
    
    # @partial(jit, static_argnums=(0,))
    def f(self, xs):
        """x.shape = (N, d). Only square outside of region"""
        dists, signs = self._get_dists(xs) 
        print(signs)
        dists = jnp.where(signs < 0, 0, dists)
        print(dists)
        ret = self.square_factor*jnp.sum(jnp.square(dists), axis=1) # shape = (N_x)
        return ret

    # @partial(jit, static_argnums=(0,))
    def f1(self, xs):
        dists, signs = self._get_dists(xs)
        dists = jnp.where(signs < 0, 0, dists)
        grads = (2*dists).dot((self.ws / jnp.linalg.norm(self.ws, axis=1).T.reshape(-1, 1)))
        return grads
