import numpy as np
import pandas as pd
from process import get_data
from forward import forward
import matplotlib.pyplot as plt

def cost(p, y):
	c = -y*np.log(p)-(1-y)*np.log(1-p)
	return c.sum()

def classification_rate(y, p):
	return np.mean(y==p)

def derv_v(y, p, z):
	z2 = np.zeros((z.shape[0], z.shape[1]+1))
	z2[:, 1:] = z
	z2[:, 0] = 1
	ret = z2.T.dot(p - y)
	return ret
def derv_w(y, p, z, v, x):
	x2 = np.zeros((x.shape[0], x.shape[1]+1))
	x2[:, 1:] = x
	x2[:, 0] = 1
	dz = (p-y).dot(v.T)[:, 1:]*z*(1-z)
	ret = x2.T.dot(dz)
	return ret
def oneHotEncoder(t, k):
	T = np.zeros((t.shape[0], k))
	for i in range(t.shape[0]):
		T[i, int(t[i])] = 1
	return T

def main():
	X, Y = get_data('ecom.csv')
	K = int(Y.max())+1

	Xtrain = X[:-100]
	Ytrain = Y[:-100]
	Ttrain = oneHotEncoder(Ytrain, K)

	Xtest = X[-100:]
	Ytest = Y[-100:]
	Ttest = oneHotEncoder(Ytest, K)

	N, D = X.shape
	M = 3
	w = np.random.rand(D+1, M)
	v = np.random.rand(M+1, K)

	cs = []
	rs = []
	alpha = 0.000005
	for i in range(100000):
		p, z = forward(Xtrain, w, v)
		c = cost(p, Ttrain)
		cs.append(c)
		r = classification_rate(Ytrain, p.argmax(axis=1))
		rs.append(r)
		v -= alpha*derv_v(Ttrain, p, z)
		w -= alpha*derv_w(Ttrain, p, z, v, Xtrain)
		print "cost: ",c,", classification rate: ",r
	plt.plot(cs)
	plt.show()
	plt.plot(rs)
	plt.show()

	p, z = forward(Xtest, w, v)
	c = cost(p, Ttest)
	r = classification_rate(Ytest, p.argmax(axis=1))

	print "Model's accuracy: ",r*100,"%, with cost of: ", c

if __name__ == '__main__':
	main()