import numpy as np
import pandas as pd
from process import get_data
from forward import forward
import matplotlib.pyplot as plt
from train import train, cost, classification_rate, oneHotEncoder


def main():
	X, Y = get_data('ecom.csv')
	K = int(Y.max())+1

	Xtrain = X[:-100]
	Ytrain = Y[:-100]

	Xtest = X[-100:]
	Ytest = Y[-100:]

	w, v = train(Xtrain, Ytrain)

	p, z = forward(Xtest, w, v)
	Ttest = oneHotEncoder(Ytest, K)
	c = cost(p, Ttest)
	r = classification_rate(Ytest, p.argmax(axis=1))

	print "Model's accuracy: ",r*100,"%, with cost of: ", c

if __name__ == '__main__':
	main()