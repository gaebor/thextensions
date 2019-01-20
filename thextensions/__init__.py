import numpy
import theano
import theano.tensor as T
import theano.tensor.slinalg as linalg
from functools import reduce

class Optimizer(object):
    def __init__(self, objective, *vars, minimize=True, constraints={}, **kwargs):
        self.sym_vars = tuple(vars)
        self.grads = tuple(map(lambda var: T.grad(objective, var, **kwargs), vars))
        self.minimize = minimize
        self.objective = objective

        self.constraints = {var: var if var not in constraints else \
                                constraints[var] for var in vars}

    def updates(self):
        """needed for compiling a theano.function"""
        update_list = []
        grad_steps = self.grad_steps()
        sign = -1 if self.minimize else 1
        for i in range(len(self.sym_vars)):
            update_list.append([self.shr_vars[i],
                                self.constraints[self.sym_vars[i]] + sign*grad_steps[i]])
        return update_list
    
    def grad_steps(self):
        pass
        
    def init(self, *initvals, **kwargs):
        """assigns numeric values to the symbolic variables
        Can re-assign later, i.e. reset the parameters
        """
        if len(initvals) != len(self.sym_vars):
            raise ValueError("Length of symbolic variables ({0}) is not equal to "
                              "length of initializer list ({1})!".format(
                                len(initvals), len(self.sym_vars))
                            )
        if hasattr(self, "shr_vars"):
            if len(initvals) != len(self.shr_vars):
                raise ValueError("initializer and already initialized shared variables"
                        " differ in length!")
            for i in range(len(initvals)):
                self.shr_vars[i].set_value(initvals[i])
        else:
            self.shr_vars = tuple(map(theano.shared, initvals))

    def givens(self):
        """needed for compiling a theano.function"""
        return tuple(zip(self.sym_vars, self.shr_vars))
    
    def feed_dict(self):
        """needed for eval"""
        return {self.sym_vars[i]: self.shr_vars[i].get_value() for i in range(len(self.sym_vars))}
        
    def get_vars(self):
        """returns a getter function which returns a list of your optimization parameters"""
        updates = self.renormalize_updates()
        return theano.function([],
                            [update[1] for update in updates],
                            givens=self.givens(),
                            updates=updates)

    def renormalize_updates(self):
        """in case of constrained optimization, only renormalization updates, no gradient descent"""
        if not hasattr(self, "shr_vars"):
            raise AttributeError("object has no attribute 'shr_vars'! "
                        "Maybe you forgot to initialize the " +  type(self).__name__)
        new_vals = [self.constraints[var] for var in self.sym_vars]
        return list(zip(self.shr_vars, new_vals))

    @staticmethod
    def numpy_cast(x):
        return eval("numpy." + theano.config.floatX)(x)

class GradientDescentOptimizer(Optimizer):
    """
    x <-- x - eta*grad(f)(x)
    """
    def __init__(self, objective, *vars, minimize=True, eta=1.0, constraints={}, **kwargs):
        super().__init__(objective, *vars, minimize=minimize, constraints=constraints, **kwargs)
        self.eta = eta

    def grad_steps(self):
        return tuple(self.eta*x for x in self.grads)

class AdagradOptimizer(Optimizer):
    """
    x   <-- x - grad(f)(x)/sqrt(sqr)
    sqr <-- grad(f)(x)**2
    """

    def __init__(self, objective, *vars, minimize=True, eta=1.0, constraints={}, **kwargs):
        super().__init__(objective, *vars, minimize=minimize, constraints=constraints, **kwargs)
        self.eta1 = Optimizer.numpy_cast(1.0/eta**2 if eta > 0 else numpy.inf)

    def updates(self):
        update_list = super().updates()
        for i in range(len(self.sym_vars)):
            update_list.append([self.grad_sqs[i], self.grad_sqs[i] + self.grads[i]**2])
        return update_list

    def grad_steps(self):
        return tuple(self.grads[i]/T.sqrt(self.grad_sqs[i]) for i in range(len(self.sym_vars)))

    def init(self, *initvals, keep_sq=False, **kwargs):
        super().init(*initvals, **kwargs)
        if not hasattr(self, "grad_sqs") or not keep_sq:
            self.grad_sqs = tuple(map(lambda x: theano.shared(self.eta1*numpy.ones_like(x)), initvals))

class HessianOptimizer(Optimizer):
    """
    x <-- x - eta * Hessian \ grad(f)
    possibly with multipliers
    """
    def __init__(self, objective, *vars, eta=1.0, constraints=[], minimize=True, **kwargs):
        self.slack_vars = tuple(T.TensorType(c.dtype, c.broadcastable)() for c in constraints)
        objective = objective + sum([(self.slack_vars[i]*constraints[i]).sum() for i in range(len(constraints))])
        
        super().__init__(objective, *(vars+self.slack_vars), minimize=True, constraints={}, **kwargs)
        
        self.var = T.concatenate([var.reshape((-1,)) for var in self.sym_vars])
        self.hessian = Hessian(objective, *self.sym_vars)
        self.grad = Grad(objective, *self.sym_vars)
        self.eta = eta
        
    def init(self, *initvals, slack_variables=[], **kwargs):
    
        initvals = initvals + tuple(numpy.ones(dtype=theano.config.floatX, shape=s) for s in slack_variables)
        super().init(*initvals)
        
        self.sizes = [val.size for val in initvals]
        self.sizes = reduce(lambda x, y: x + [x[-1]+y], self.sizes, [0])
        
    def grad_steps(self):
        direction = linalg.solve_symmetric(self.hessian, self.grad)
        n = len(self.sym_vars)
        return tuple(self.eta*direction[self.sizes[i]:self.sizes[i+1]].reshape(self.sym_vars[i].shape) \
                     for i in range(len(self.sym_vars)))

def Grad(objective, *Vars, **kwargs):
    """gradients concatenated into a big vector"""
    return T.concatenate([
                        T.grad(objective, var, disconnected_inputs='ignore').reshape((-1,))
                           for var in Vars])
                                
def Hessian(objective, *Vars, **kwargs):
    """block structure matrix of Jacobian of gradients, symmetric"""
    return T.concatenate([
                T.concatenate([
                    T.jacobian(
                        T.grad(objective, var1, disconnected_inputs='ignore').reshape((-1,)),
                        var2, disconnected_inputs='ignore').reshape(
                            (var1.size, var2.size)
                            ) for var2 in Vars],
                axis=1) for var1 in Vars],
            axis=0)
