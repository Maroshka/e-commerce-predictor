import numpy as np
import pandas as pd
from process import get_data
from forward import forward
import matplotlib.pyplot as plt

def cost(p, y):
	c = -y*np.log(p)-(1-y)*np.log(1-p)
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
	x2[:, 1:] = z
	x2[:, 0] = 1
	dz = (p-y).dot(v.T)[:, 1:]*z*(1-z)
	ret = x2.T.dot(dz)
	return ret

def main():
	X, Y = get_data('ecom.csv')
	K = int(Y.max())+1
	N, D = X.shape
	M = 3
	w = np.random.rand(D+1, M)
	v = np.random.rand(M+1, K)

	T = np.zeros((N, K))
	for i in range(N):
		T[i, int(Y[i])] = 1
	cs = []
	rs = []
	alpha = 0.00005
	for i in range(1000):
		p, z = forward(X, w, v)
		c = cost(p, T)
		cs.append(c)
		r = classification_rate(Y, p.argmax(axis=1))
		rs.append(r)
		v -= alpha*derv_v(T, p, z)
		w -= alpha*derv_w(T, p, z, v, xs)
	plt.plot(cs)
	plt.show()
	plt.plot(rs)
	plt.show()

if __name__ == '__main__':
	main()