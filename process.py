import numpy as np
import pandas as pd 

def get_data(fname):
	df = pd.read_csv(fname)
	data = df.as_matrix()	

	X = data[:, :-1]
	Y = data[:, -1]

	X[:, 1] = (X[:, 1]-np.mean(X[:, 1]))/np.std(X[:, 1])
	X[:, 2] = (X[:, 2]-np.mean(X[:, 2]))/np.std(X[:, 2])
	D = X.shape[1]
	N = X.shape[0]
	X2 = np.zeros((X.shape[0], X.shape[1]+3))
	X2[:, 0:D-1] = X[:, 0:D-1]
	for i in range(N):
		t = int(X[i, D-1])
		X2[i, (D-1)+t] = 1

	return X2, Y