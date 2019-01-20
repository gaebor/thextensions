import numpy
import theano
import argparse

import thextensions.__init__

def main(args):
    x = theano.tensor.vector()

    f = x.sum()

    if args.optimizer == "Hessian":
        constraints = [x.dot(x)-1] if args.use_constraint else []
        slack_variables = [tuple()] if args.use_constraint else []
    else:
        constraints = {x: x/theano.tensor.sqrt(x.dot(x))}
        slack_variables = []
    
    optimizer = eval("thextensions.{}Optimizer".format(args.optimizer))(
                f, x, constraints=constraints, eta=args.eta, minimize=not args.max)
    
    optimizer.init(numpy.random.rand(2)-0.5, slack_variables=slack_variables)

    f_update = theano.function([], f,
                                givens=optimizer.givens(),
                                updates=optimizer.updates())
    getter = optimizer.get_vars()

    for _ in range(args.n_steps):
        print(f_update())
    print(getter())
    
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                description="Author: Gabor Borbely, contact: borbely@math.bme.hu",
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-e', '--eta', dest='eta', type=float,
                    default=1, help='learning rate')

    parser.add_argument('-m', '--max', '--maximize', dest='max', action="store_true")
                    
    parser.add_argument('-n', '--steps', dest='n_steps', type=int,
                    default=20, help='number of updates')
    
    parser.add_argument('-o', '--optimizer', dest='optimizer', type=str,
                    choices=["GradientDescent", "Adagrad", "Hessian"],
                    default="GradientDescent", help='type of the gradient descent')

    parser.add_argument('-c', '--constraint', '--constraints',
                    dest='use_constraint', action="store_true", default=True,
                    help='use the constraint "x^2+y^2==1"')
    
    parser.add_argument('-C', '--no-constraint', '--no-constraints',
                    dest='use_constraint', action="store_false", default=True)

    exit(main(parser.parse_args()))
