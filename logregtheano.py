import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T
import os
import sys
import timeit

class logisticRegression(object):
    
	def __init__(self, input, n_in, n_out):
		self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
		self.b = theano.shared(value=np.zeros((n_out,),
						dtype=theano.config.floatX
					),
					name='b',
					borrow=True
				)
		self.pYGivenX = T.nnet.softmax(T.dot(input, self.W) + self.b)
		self.yPred = T.argmax(self.pYGivenX, axis = 1)
		self.params = [self.W, self.b]
		self.input = input

	def negativeLogLikelihood(self, y):
		return -T.mean(T.log(self.pYGivenX)[T.arange(y.shape[0]), y])

	def errors(self, y):
		if y.ndim != self.yPred.ndim:
			raise TypeError( 'y should have the same shape as self.y_pred',
					('y', y.type, 'yPred', self.yPred.type)
			)
		
		if y.dtype.startswith('int'):
			return T.mean(T.neq(self.yPred, y))
		else:
			raise NotImplementedError()

def loadData(dataset):
	
	with gzip.open(dataset, 'rb') as f:
		try:
			trainSet, validSet, testSet = cPickle.load(f, encoding = 'latin1')
		except:
			trainSet, validSet, testSet = cPickle.load(f)

	def sharedDataset(dataXY, borrow=True):
		dataX, dataY = dataXY
		sharedX = theano.shared(np.asarray(dataX, dtype = theano.config.floatX), borrow = borrow)
		sharedY = theano.shared(np.asarray(dataY, dtype = theano.config.floatX), borrow = borrow)
		return sharedX, T.cast(sharedY, 'int32')
	
	testSetX, testSetY = sharedDataset(testSet)
	validSetX, validSetY = sharedDataset(validSet)
	trainSetX, trainSetY = sharedDataset(trainSet)

	rVal = [(testSetX, testSetY), (validSetX, validSetY), (trainSetX, trainSetY)]
	return rVal	

def sgdOptimizationMnist(learningRate = 0.13, nEpochs=1000, dataset = 'data/mnist.pkl.gz', batchSize = 600):

	datasets = loadData(dataset)
	trainSetX, trainSetY = datasets[2]
	validSetX, validSetY = datasets[1]
	testSetX, testSetY = datasets[0]

	nTrainBatches = trainSetX.get_value(borrow=True).shape[0]/batchSize
	nValidBatches = validSetX.get_value(borrow=True).shape[0]/batchSize
	nTestBatches = testSetX.get_value(borrow=True).shape[0]/batchSize

	print('...building the model')

	index = T.lscalar() # index to a miniBatch
	x = T.matrix('x')
	y = T.ivector('y')

	classifier = logisticRegression(input=x, n_in = 28*28, n_out = 10)
	cost = classifier.negativeLogLikelihood(y)

	gW = T.grad(cost = cost, wrt = classifier.W)
	gb = T.grad(cost = cost, wrt = classifier.b)

	updates = [(classifier.W, classifier.W-learningRate*gW), (classifier.b, classifier.b-learningRate*gb)]

	trainModel = theano.function(inputs = [index], outputs = cost, updates = updates, givens ={
			x: trainSetX[index*batchSize : (index + 1)*batchSize],
			y: trainSetY[index*batchSize : (index + 1)*batchSize]
		}
	)

	testModel = theano.function(inputs = [index], outputs = classifier.errors(y), givens = {
			x: testSetX[index*batchSize : (index+1)*batchSize],
			y: testSetY[index*batchSize : (index+1)*batchSize]
		}
	)

	validateModel = theano.function(inputs = [index], outputs = classifier.errors(y), givens = {
			x: validSetX[index*batchSize : (index+1)*batchSize],
			y: validSetY[index*batchSize : (index+1)*batchSize]
		}
	)

	###############
	# TRAIN MODEL #
	###############


	print('...building the model')
	# early-stopping parameters

	patience = 5000 # look this many examples regardless
	patienceIncrease = 2 # wait this much longer when a new best is found

	improvementThreshold = 0.995
	validationFrequency = min(nTrainBatches, patience/2)

	bestValidationLoss = np.inf
	testScore = 0.
	startTime = timeit.default_timer()

	doneLooping = False
	epoch = 0
	while (epoch < nEpochs) and (not doneLooping):
		epoch+=1
		for miniBatchIndex in range(nTrainBatches):
			miniBatchAvgCost = trainModel(miniBatchIndex)
			iter = (epoch - 1)*nTrainBatches+miniBatchIndex

			if (iter + 1)%validationFrequency == 0:
				validationLosses = [validateModel(i) for i in range(nValidBatches)]
				thisValidationLoss = np.mean(validationLosses)

				print 'epoch {0}, minibatch {1}/{2}, validation error {3} %'.format(epoch, miniBatchIndex+1, nTrainBatches, thisValidationLoss*100.)
				if thisValidationLoss < bestValidationLoss:
					if thisValidationLoss < bestValidationLoss*improvementThreshold:
						patience = max(patience, iter*patienceIncrease)
					bestValidationLoss = thisValidationLoss
					testLosses = [testModel(i) for i in range(nTestBatches)]
					testScore = np.mean(testLosses)
					print 'epoch {0}, minibatch {1}/{2}, test error of best model {3} %'.format(epoch, miniBatchIndex + 1, nTrainBatches, testScore * 100.)
			
			if patience<=iter:
				doneLooping = True
				break

	endTime = timeit.default_timer()
	print '\nDevice used was ', theano.config.device
	print 'Optimization complete with best validation score of {0} %, with test performance {1} %'.format( bestValidationLoss * 100., testScore * 100.)
	print 'The code run for {0} epochs, with {1} epochs/sec'.format(epoch, 1. * epoch / (endTime - startTime))
	print 'The code for file ' + os.path.split(__file__)[1] + ' ran for {0}s'.format(endTime - startTime)

def predict():
	"""
	An example of how to load a trained model and use it
	to predict labels.
	"""

	# load the saved model
	classifier = cPickle.load(open('bestModel.pkl'))

	# compile a predictor function
	predict_model = theano.function(
		inputs=[classifier.input],
		outputs=classifier.yPred)

	# We can test it on some examples from test test
	dataset='mnist.pkl.gz'
	datasets = loadData(dataset)
	test_set_x, test_set_y = datasets[2]
	test_set_x = test_set_x.get_value()

	predicted_values = predict_model(test_set_x[:10])
	print "Predicted values for the first 10 examples in test set:"
	print predicted_values


if __name__ == '__main__':
	sgdOptimizationMnist()