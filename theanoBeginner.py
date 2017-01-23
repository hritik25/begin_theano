import theano
from theano import tensor as T
import numpy as np
import timeit

# y = 2*x + 1
train_x = [3,4]
train_y = [7,9]
learningRate = 0.005

X = T.scalar()
Y = T.scalar()

def model(X,W,b):
	return X*W + b

W = theano.shared(np.asarray(0., dtype=theano.config.floatX))
b = theano.shared(np.asarray(0., dtype=theano.config.floatX))
y = model(X, W, b)

cost = T.mean(T.sqr(y-Y))
gradient = T.grad(cost = cost, wrt = [W,b])
updates = [[W, W - gradient[0] * learningRate], [b, b - gradient[1] * learningRate]]

train = theano.function(inputs = [X, Y], outputs=cost, updates = updates, allow_input_downcast=True)

startTime = timeit.default_timer()

for i in range(50):
	for x,y in zip(train_x, train_y):	
		train(x, y)
		print 'W = ', W.get_value(), 'b = ', b.get_value()

endTime = timeit.default_timer()
execTime = endTime - startTime

print 'Execution time for 50 loops with ', theano.config.device, '= ', execTime, 's'