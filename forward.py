import numpy as np
import pandas as pd

def forward(x, w, v):
	x2 = np.zeros((x.shape[0], x.shape[1]+1))
	x2[:, 1:] = x
	x2[:, 0] = 1

	a = x2.dot(w)
	z = 1 / (1 + np.exp(-a))

	z2 = np.zeros((z.shape[0], z.shape[1]+1))
	z2[:, 1:] = z
	z2[:, 0] = 1

	expz = np.exp(z2)
	h = expz / expz.sum(axis=1, keepdims=True)

	return h, z