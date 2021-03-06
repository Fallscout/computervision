import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score


def test(data, eigenvectors, mean, train_descriptors, train_labels, test_labels):
	"""
	Projects the given faces onto the subspace defined by eigenvectors and classifies them
	via nearest neighbor approach against the given descriptors.

	Args:
		data: the face data to classify.
		mean: the mean face of the training data.
		train_descriptors: the descriptors for the training set faces.
		train_labels: the labels for the training data.
		test_labels: the labels for the test data.

	Returns:
		Classification accuracy, describing how many test faces have been
		labeled as the right person.
	"""
	test_descriptors = np.dot(data - mean, eigenvectors.T)

	nn = NearestNeighbors(n_neighbors=1).fit(train_descriptors)
	distances, indices = nn.kneighbors(test_descriptors)
	predicted_labels = train_labels[indices].flatten()

	return accuracy_score(test_labels, predicted_labels)

def load_data(path):
	"""
	Loads the face data from the provided file.

	Args:
		path: the path to the .mat file.

	Returns:
		data: The image data.
		labels: The labels for the images.
	"""
	if not os.path.isfile(path):
		raise IOError("Argument is not a file")
	else:
		mat = loadmat(path)
		data = mat["fea"]
		labels = mat["gnd"]
		return data, labels

def load_spec(path):
	"""
	Loads the train/test specification from the provided file.
	This file indicates which images from the dataset are used
	for training and for testing.

	Args:
		path: the path to the .mat file.

	Returns:
		A dictionary containing the specification.
	"""
	if not os.path.isfile(path):
		raise IOError("Argument is not a file")
	else:
		mat = loadmat(path)
		return mat

def pca(data, k, use_sklearn=False):
	"""
	Applies principal component analysis to the given data, calculating the
	first k principal components.

	Args:
		data: the data to calculate the principal components for
		(in the format observation x feature vector).
		k: the number of principal components to calculate.
		use_sklearn: a flag to indicate whether sklearn should be used to do
		the PCA (for comparison with the own implementation)

	Returns:
		descriptors: the projection of the data onto the new subspace.
		eigenvectors: the eigenvectors corresponding to the k highest eigenvalues.
		mean: the mean of the data.
	"""
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

def reconstruct(mean, eigenvectors, descriptor):
	"""
	Reconstructs the image of a face from the mean face, the eigenvectors
	and the image's descriptors.

	Args:
		mean: the mean face.
		eigenvectors: the eigenvectors of the face subspace.
		descriptor: the target face's descriptor.

	Returns:
		The reconstructed face image as a row vector.
	"""
	return mean + np.dot(eigenvectors.T, descriptor)

def plot_reconstruction(mean, eigenvectors, descriptor, rows, columns):
	"""
	Plots the reconstruction of a face with varying numbers of eigenfaces.

	Args:
		mean: the mean face.
		eigenvectors: the eigenvectors of the face subspace.
		descriptor: the target face's descriptor.
		rows: number of rows in the plot.
		columns: number of columns in the plot.
	"""
	reconstructions = []
	titles = []

	for i in range(10, min(len(eigenvectors), 320), 20):
	    reconstructions.append(reconstruct(mean, eigenvectors[:i], descriptor[:i]).reshape(32, 32))
	    titles.append("{} eigenfaces".format(i))

	plot_faces(reconstructions, rows, columns, titles)


def plot_faces(faces, rows, columns, sptitle=None):
	"""
	Plots the given image data in rows x columns subplots.

	Args:
		faces: the face data.
		rows: number of rows in the plot.
		columns: number of columns in the plot.
		sptitle: titles for the subplots.
	"""
	fig = plt.figure()
	
	for i, face in enumerate(faces):
		ax = fig.add_subplot(rows, columns, i+1)
		ax.axis("off")
		ax.imshow(np.rot90(face, k=3), cmap=cm.gray)

		if sptitle != None:
			if len(sptitle) == len(faces):
				plt.title(sptitle[i], fontname="Tahoma", fontsize=10)
			else:
				plt.title("{} #{}".format(sptitle, i), fontname="Tahoma", fontsize=10)
	fig.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument("--datafile", default="faces/ORL_32x32.mat", help="file with face data")
	parser.add_argument("--specfile", default="faces/5Train/5.mat", help="matrix specifying which images to use for training and testing")
	parser.add_argument("--k", help="number of principal components to use", type=int, default=None)
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

	if args.k != None:
		if args.k > len(train_data):
			print("Warning: specified more principal components than there are images in the training set. " +
				"Using {} principal components instead.".format(len(train_data)))
			args.k = len(train_data)

		train_descriptors, eigenvectors, mean = pca(train_data, args.k, args.use_sklearn)
		eigenfaces = eigenvectors.reshape(args.k, 32, 32)
		
		score = test(test_data, eigenvectors, mean, train_descriptors, train_labels, test_labels)
		print("Accuracy: {}".format(score))
	else:
		# If no k is specified, calculate accuracies for all possible values of k and
		# return scores array to global scope
		scores = []
		for k in range(1, len(train_data)+1):
			train_descriptors, eigenvectors, mean = pca(train_data, k, args.use_sklearn)
			eigenfaces = eigenvectors.reshape(k, 32, 32)
			
			score = test(test_data, eigenvectors, mean, train_descriptors, train_labels, test_labels)
			scores.append(score)
