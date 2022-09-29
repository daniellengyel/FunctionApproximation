from jax import jit, grad, jacfwd
import jax.numpy as jnp
from jax.lax import fori_loop

from functools import partial

from jax.config import config
config.update("jax_enable_x64", True)

"""f and f1 are reasonably well optimized. However, f2 will not be called at once on a lot of data,
hence we will used jacfwd to compute it for every datapoint given. """

def get_func(f):
    if f == "Ackley":
        return Ackley()

    elif f == "Schwefel":
        return Schwefel()
    
    elif f == "Michi":
        return Michi()

    elif f == "Rosenbrock":
        return Rosenbrock()

    elif f == "Dixon":
        return Dixon()
    
    elif f == "Rasti":
        return Rasti()

    raise Exception("Function {} does not exist.".format(f))


class Rasti():
    def __init__(self):
        self.freq = 4*jnp.pi
        self.bounds = jnp.array([[-2.5, 2.5]])
        self._f1 = jacfwd(lambda x: self.f(x.reshape(1, -1))[0])
        self._f2 = jacfwd(lambda x: self._f1(x.reshape(1, -1)))

    def f(self, X):
        """X.shape = (N, dim)"""
        # if len(X.shape) == 1:
        #     X = X.reshape(1, -1)
        # return jnp.sum(jnp.sin(X*self.freq), axis=1) + jnp.diag(X @ X.T)
        return  -25*jnp.exp(-jnp.linalg.norm(X, axis=1)**2 / 1.5) - 10*jnp.exp(-jnp.linalg.norm(X, axis=1)**2 / 0.5) + 10 * X.shape[1] + jnp.sum( - 10 * jnp.cos(2*jnp.pi*X), axis=1)

    # def f1(self, X):
    #     # if len(X.shape) == 1:
    #     #     X = X.reshape(1, -1)
    #     return jnp.cos(X*self.freq) * self.freq + 2 * X

    @partial(jit, static_argnames=['self'])
    def f1(self, X):
        assert len(X.shape) == 2
        grads = jnp.zeros(shape=X.shape)
        def body_fun(i, val):
            val = val.at[i].set(self._f1(X[i]))
            return val
        grads = fori_loop(0, X.shape[0], body_fun, grads)
        return grads

    @partial(jit, static_argnames=['self'])
    def f2(self, X): 
        assert len(X.shape) == 2
        hess = jnp.zeros(shape=(X.shape[0], X.shape[1], X.shape[1]))
        def body_fun(i, val):
            val = val.at[i].set(self._f2(X[i])[0])
            return val
        hess = fori_loop(0, X.shape[0], body_fun, hess)
        return hess

    # def f2_new(self, X):
    #     res = -jnp.sin(X*self.freq) * self.freq**2 + 2
    #     return jnp.array([jnp.diag(r) for r in res])


class Schwefel():
    def __init__(self):
        self.bounds = jnp.array([[-500, 500]])
        self._f1 = jacfwd(lambda x: self.f(x.reshape(1, -1))[0])
        self._f2 = jacfwd(lambda x: self._f1(x.reshape(1, -1)))
        self.c = 418.9829

    def f(self, X):
        d = X.shape[1]
        return self.c * d - jnp.sum(jnp.sin(jnp.sqrt(jnp.abs(X))) * X, axis=1)

    def f1(self, X):
        assert len(X.shape) == 2
        return -jnp.sin(jnp.sqrt(jnp.abs(X))) - 1/2.*(jnp.sqrt(jnp.abs(X))) * jnp.cos(jnp.sqrt(jnp.abs(X))) 

    @partial(jit, static_argnames=['self'])
    def f2(self, X): 
        assert len(X.shape) == 2
        hess = jnp.zeros(shape=(X.shape[0], X.shape[1], X.shape[1]))
        def body_fun(i, val):
            val = val.at[i].set(self._f2(X[i])[0])
            return val
        hess = fori_loop(0, X.shape[0], body_fun, hess)
        return hess

    def f2_new(self, X):
        assert len(X.shape) == 2
        first_term = -jnp.sign(X) * 0.5/(jnp.sqrt(jnp.abs(X)))*jnp.cos(jnp.sqrt(jnp.abs(X)))
        second_term_first_half = -1/4. * jnp.sign(X) * 1/(jnp.sqrt(jnp.abs(X))) * jnp.cos(jnp.sqrt(jnp.abs(X)))
        second_term_second_half = 1/4. * jnp.sin(jnp.sqrt(jnp.abs(X))) * jnp.sign(X)
        return first_term + second_term_first_half + second_term_second_half

class Ackley():
    def __init__(self):
        self.bounds = jnp.array([[-5, 5]])
        self._f1 = jacfwd(lambda x: self.f(x.reshape(1, -1))[0])
        self._f2 = jacfwd(lambda x: self._f1(x.reshape(1, -1)))

        self.a = 10
        self.b = 0.2
        self.c = jnp.pi

    def f(self, X):
        square_mean = jnp.mean(X**2, axis=1)
        cos_mean = jnp.mean(jnp.cos(self.c*X), axis=1)
        return -self.a * jnp.exp(- self.b * jnp.sqrt(square_mean)) - jnp.exp(cos_mean) + self.a + jnp.exp(1)

        

    @partial(jit, static_argnames=['self'])
    def f1(self, X):
        assert len(X.shape) == 2
        grads = jnp.zeros(shape=X.shape)
        def body_fun(i, val):
            val = val.at[i].set(self._f1(X[i]))
            return val
        grads = fori_loop(0, X.shape[0], body_fun, grads)
        return grads


    @partial(jit, static_argnames=['self'])
    def f2(self, X): 
        assert len(X.shape) == 2
        hess = jnp.zeros(shape=(X.shape[0], X.shape[1], X.shape[1]))
        def body_fun(i, val):
            val = val.at[i].set(self._f2(X[i])[0])
            return val
        hess = fori_loop(0, X.shape[0], body_fun, hess)
        return hess

class Michi():
    def __init__(self):
        self.bounds = jnp.array([[0, jnp.pi]])
        self._f1 = jacfwd(lambda x: self.f(x.reshape(1, -1))[0])
        self._f2 = jacfwd(lambda x: self._f1(x.reshape(1, -1)))
        self.m = 2

    def f(self, X):
        sins = jnp.sin(X)
        square_scale = jnp.array(range(1, X.shape[1] + 1)) * X**2 / jnp.pi
        return - jnp.sum(sins * jnp.sin(square_scale)**(2 * self.m), axis=1)

        

    @partial(jit, static_argnames=['self'])
    def f1(self, X):
        assert len(X.shape) == 2
        grads = jnp.zeros(shape=X.shape)
        def body_fun(i, val):
            val = val.at[i].set(self._f1(X[i]))
            return val
        grads = fori_loop(0, X.shape[0], body_fun, grads)
        return grads


    @partial(jit, static_argnames=['self'])
    def f2(self, X): 
        assert len(X.shape) == 2
        hess = jnp.zeros(shape=(X.shape[0], X.shape[1], X.shape[1]))
        def body_fun(i, val):
            val = val.at[i].set(self._f2(X[i])[0])
            return val
        hess = fori_loop(0, X.shape[0], body_fun, hess)
        return hess

           
class Rosenbrock():
    "Only for even dimensions"
    def __init__(self):
        self.bounds = jnp.array([[-5, 10]])
        self._f1 = jacfwd(lambda x: self.f(x.reshape(1, -1))[0])
        self._f2 = jacfwd(lambda x: self._f1(x.reshape(1, -1)))

    def f(self, X):
        diffs = X[:, 1:] - X[:, :-1]**2
        return jnp.sum(100 * diffs**2 + (X[:, :-1] - 1)**2, axis=1)
        

    @partial(jit, static_argnames=['self'])
    def f1(self, X):
        assert len(X.shape) == 2
        grads = jnp.zeros(shape=X.shape)
        def body_fun(i, val):
            val = val.at[i].set(self._f1(X[i]))
            return val
        grads = fori_loop(0, X.shape[0], body_fun, grads)
        return grads


    @partial(jit, static_argnames=['self'])
    def f2(self, X): 
        assert len(X.shape) == 2
        hess = jnp.zeros(shape=(X.shape[0], X.shape[1], X.shape[1]))
        def body_fun(i, val):
            val = val.at[i].set(self._f2(X[i])[0])
            return val
        hess = fori_loop(0, X.shape[0], body_fun, hess)
        return hess

class Dixon():
    "Only for even dimensions"
    def __init__(self):
        self.bounds = jnp.array([[-5, 5]])
        self._f1 = jacfwd(lambda x: self.f(x.reshape(1, -1))[0])
        self._f2 = jacfwd(lambda x: self._f1(x.reshape(1, -1)))

    def f(self, X):
        diffs = (2*X[:, 1:]**2 - X[:, :-1])**2
        return (X[:, 0] - 1)**2 + diffs @ jnp.array(list(range(2, X.shape[1] + 1)))
        

    @partial(jit, static_argnames=['self'])
    def f1(self, X):
        assert len(X.shape) == 2
        grads = jnp.zeros(shape=X.shape)
        def body_fun(i, val):
            val = val.at[i].set(self._f1(X[i]))
            return val
        grads = fori_loop(0, X.shape[0], body_fun, grads)
        return grads


    @partial(jit, static_argnames=['self'])
    def f2(self, X): 
        assert len(X.shape) == 2
        hess = jnp.zeros(shape=(X.shape[0], X.shape[1], X.shape[1]))
        def body_fun(i, val):
            val = val.at[i].set(self._f2(X[i])[0])
            return val
        hess = fori_loop(0, X.shape[0], body_fun, hess)
        return hess

           

           
