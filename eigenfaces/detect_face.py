import logging
import os
import argparse
import datetime
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from progress.bar import Bar

def main():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--datafile", default="faces/ORL_32x32.mat", help="file with face data")
	parser.add_argument("--specfile", default="faces/3Train/3.mat", help="matrix specifying which images to use for training and testing")
	parser.add_argument("k", help="number of principal components to use", type=int, default=20)
	parser.add_argument("--use_sklearn", help="", action="store_true")
	args = parser.parse_args()

	data, labels = load_data(args.datafile)
	spec = load_spec(args.specfile)

	# Normalize data
	data = np.array(data/255.0)

	# Need to subtract 1 from the indices, because matrices
	# have originally been created in MATLAB and
	# MATLAB indexing starts at 1 as opposed to 0
	train_data = data[spec["trainIdx"].flatten()-1]
	train_labels = labels[spec["trainIdx"].flatten()-1].flatten()
	test_data = data[spec["testIdx"].flatten()-1]
	test_labels = labels[spec["testIdx"].flatten()-1].flatten()

	if args.use_sklearn:
		eigenfaces, train_descriptors, pca = find_eigenfaces(train_data, args.k)
		test_descriptors = fit_test_data(test_data, pca)
	else:
		mean = np.mean(train_data, axis=0)
		cov = calculate_covariance_matrix(train_data)
		#cov = np.cov(train_data.T)

		eigvals, eigvecs = linalg.eig(cov)
		idx = eigvals.argsort()[::-1]
		eigvals = eigvals[idx]
		eigvecs = eigvecs[:, idx]

		# keep only k
		eigvals = eigvals[:args.k]
		eigvecs = eigvecs[:args.k]

		# reshape eigenfaces
		eigenfaces = eigvecs.reshape(args.k, 32, 32)

		X = (train_data - mean).T
		train_descriptors = np.dot(eigvecs, X).T

		test_descriptors = np.dot(eigvecs, (test_data - mean).T).T

	nn = NearestNeighbors(n_neighbors=1).fit(train_descriptors)
	distances, indices = nn.kneighbors(test_descriptors)
	recognized = train_labels[indices].flatten()
	fraction = sum(np.array(test_labels == recognized, dtype=int))/float(test_labels.size)
	print(fraction)

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

def find_eigenfaces(data, k):
	pca = PCA(n_components=k)
	pca.fit(data)
	eigenfaces = pca.components_.reshape(k, 32, 32)
	descriptors = pca.transform(data)
	return eigenfaces, descriptors, pca

def fit_test_data(data, pca):
	return pca.transform(data)

def calculate_covariance_matrix(data):
	mean = np.mean(data, axis=0)
	X = (data-mean).T
	return (1.0/data.shape[0]) * np.dot(X, X.T)

if __name__ == "__main__":
	main()