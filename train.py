import numpy as np
import pandas as pd
from forward import forward

def cost(p, y):
	c = -np.mean(y*np.log(p)+(1-y)*np.log(1-p))
	return c

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
def train(x, y):
	cs = []
	rs = []
	N, D = x.shape
	K = len(set(y))
	M = 3
	w = np.random.rand(D+1, M)
	v = np.random.rand(M+1, K)
	alpha = 0.000005
	T = oneHotEncoder(y, K)
	for i in range(100000):
		p, z = forward(x, w, v)
		c = cost(p, T)
		cs.append(c)
		r = classification_rate(y, p.argmax(axis=1))
		rs.append(r)
		v -= alpha*derv_v(T, p, z)
		w -= alpha*derv_w(T, p, z, v, x)
		print "cost: ",c,", classification rate: ",r
	plt.plot(cs)
	plt.show()
	plt.plot(rs)
	plt.show()
	return w, v