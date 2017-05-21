import logging
import os
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--datafile", default="faces/ORL_32x32.mat", help="file with face data")
	parser.add_argument("--specfile", default="faces/3Train/3.mat", help="matrix specifying which images to use for training and testing")
	parser.add_argument("k", help="number of principal components to use", type=int, default=20)
	parser.add_argument("--use_sklearn", help="", action="store_true")
	args = parser.parse_args()

	data, labels = load_data(args.datafile)

	# Normalize data
	data = np.array(data/255.0)

	spec = load_spec(args.specfile)

	# Need to subtract 1 from the indices, because matrices
	# have originally been created in MATLAB and
	# MATLAB indexing starts at 1 as opposed to 0
	train_data = data[spec["trainIdx"].flatten()-1]
	train_labels = labels[spec["trainIdx"].flatten()-1].flatten()
	test_data = data[spec["testIdx"].flatten()-1]
	test_labels = labels[spec["testIdx"].flatten()-1].flatten()

	train_descriptors, eigenvectors, mean = pca(train_data, args.k, args.use_sklearn)
	test_descriptors = np.dot(test_data-mean, eigenvectors.T)

	eigenfaces = eigenvectors.reshape(args.k, 32, 32)

	nn = NearestNeighbors(n_neighbors=1).fit(train_descriptors)
	distances, indices = nn.kneighbors(test_descriptors)
	predicted_labels = train_labels[indices].flatten()
	score = accuracy_score(test_labels, predicted_labels)
	print(score)
	return mean, eigenfaces

def load_data(path):
	if not os.path.isfile(path):
		raise IOError("Argument is not a file")
	else:
		mat = loadmat(path)
		data = mat["fea"]
		labels = mat["gnd"]
		return data, labels

def load_spec(path):
	if not os.path.isfile(path):
		raise IOError("Argument is not a file")
	else:
		mat = loadmat(path)
		return mat

def pca(data, k, use_sklearn=False):
	mean = np.mean(data, axis=0)
	X = data - mean

	if use_sklearn:
		pca = PCA(n_components=k)
		pca.fit(X)
		eigenvectors = pca.components_
		descriptors = pca.transform(X)
	else:
		[n, d] = X.shape
		
		if n > d:
			C = np.dot(X.T, X)
			[eigenvalues, eigenvectors] = np.linalg.eigh(C)
		else:
			C = np.dot(X, X.T)
			[eigenvalues, eigenvectors] = np.linalg.eigh(C)
			eigenvectors = np.dot(X.T, eigenvectors).T
			for i in range(n):
				# normalize
				eigenvectors[i] = eigenvectors[i]/np.linalg.norm(eigenvectors[i])

		idx = np.argsort(-eigenvalues)
		eigenvalues = eigenvalues[idx]
		eigenvectors = eigenvectors[idx]

		# keep only k
		eigenvalues = eigenvalues[:k]
		eigenvectors = eigenvectors[:k]

		descriptors = np.dot(X, eigenvectors.T)

	return descriptors, eigenvectors, mean

def plot_eigenfaces(eigenfaces, rows, columns):
	fig = plt.figure()
	
	for i, face in enumerate(eigenfaces):
		ax = fig.add_subplot(rows, columns, i+1)
		ax.axis("off")
		ax.imshow(np.rot90(face, k=3), cmap="binary")

	fig.show()

if __name__ == "__main__":
	mean, eigenfaces = main()
	#plot_eigenfaces(eigenfaces, 10, 12)