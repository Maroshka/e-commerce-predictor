import numpy as np
import pandas as pd

def softmax(z, v):
	expz = np.exp(z.dot(v))
	h = expz / expz.sum(axis=1, keepdims=True)
	return h
	
def sigmoid(a):
	return 1 / (1 + np.exp(-a))

def forward(x, w, v):
	x2 = np.zeros((x.shape[0], x.shape[1]+1))
	x2[:, 1:] = x
	x2[:, 0] = 1

	z = sigmoid(x2.dot(w)) 

	z2 = np.zeros((z.shape[0], z.shape[1]+1))
	z2[:, 1:] = z
	z2[:, 0] = 1

	h = softmax(z2, v)

	return h, z