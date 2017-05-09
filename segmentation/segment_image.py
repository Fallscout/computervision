#!/usr/bin/env python3
import logging
import cv2
import argparse
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from progress.bar import Bar

def findpeak(data, idx, r, tree=None):
	"""
	Calculates the peak of the corresponding mode for the specified data point.

	Args:
		data: image data.
		idx: index of data point for which to calculate the peak.
		r: window size.
		tree: cKDTree generated from data.

	Returns:
		The peak of the mode for the specified data point.
	"""
	t = 0.01
	shift = 1

	# Get point of interest
	point = data[idx]
	point = point.reshape(1, point.size)

	while shift > t:		

		# Determine points in window
		if tree == None:
			distances = spatial.distance.cdist(point, data)
			close_points = np.where(distances <= r)[1]
		else:
			close_points = tree.query_ball_point(point, r)[0]

		neighbors = data[close_points]

		# Calculate new mean
		new_point = np.mean(close_points, axis=1)
		new_point = new_point.reshape(1, new_point.size)

		#Calculate shift distance
		shift = spatial.distance.euclidean(point, new_point)
		point = new_point

	return point

def meanshift(data, r, tree=None):
	"""
	Applies mean shift to every data point in the array.

	Args:
		data: image data.
		r: window size.
		tree: cKDTree generated from data.

	Returns:
		A list of labels for the data point, that indicates which
		peak each pixel of the image belongs to, and the list of peaks.
	"""
	peaks = []
	labels = np.zeros(data.shape[0]) - 1
	for i in range(data.shape[0]):
		peak = findpeak(data, i, r, tree)

		if len(peaks) == 0:
			peaks.append(peak)
			labels[i] = len(peaks) - 1
		else:
			distances = spatial.distance.cdist(peak, np.array(peaks).reshape(len(peaks), peak.size))[0]
			min_dist = np.min(distances)
			if min_dist < r/2.0:
				labels[i] = np.argmin(distances)
			else:
				peaks.append(peak)
				labels[i] = len(peaks) - 1

	return labels, np.array(peaks).reshape(len(peaks), peak.size)

def findpeak_opt(data, idx, r, c, tree=None):
	"""
	Calculates the peak of the corresponding mode for the specified data point.
	Also returns all points within a radius r/c of the search path.

	Args:
		data: image data.
		idx: index of data point for which to calculate the peak.
		r: window size.
		c: denominator of search path window.
		tree: cKDTree generated from data.

	Returns:
		The peak of the mode for the specified data point and a list
		of indices of points close to the search path.
	"""
	t = 0.01
	shift = 1
	cpts = set()

	# Get point of interest
	point = data[idx]
	point = point.reshape(1, point.size)

	while shift > t:		

		# Determine points in window
		if tree == None:
			distances = spatial.distance.cdist(point, data)
			close_points = np.where(distances <= r)[1]
			cpts = cpts.union(np.where(distances <= r/c)[1])
		else:
			close_points = tree.query_ball_point(point, r)[0]
			cpts = cpts.union(tree.query_ball_point(point, r/c)[0])

		
		neighbors = data[close_points]

		# Calculate new mean
		new_point = np.mean(neighbors, axis=0)
		new_point = new_point.reshape(1, new_point.size)

		#Calculate shift distance
		shift = spatial.distance.euclidean(point, new_point)
		point = new_point

	return point, list(cpts)

def meanshift_opt(data, r, c, tree=None):
	"""
	Applies mean shift to every data point in the array.
	Associates all points within the window r with the found peak.

	Args:
		data: image data.
		r: window size.
		tree: cKDTree generated from data.

	Returns:
		A list of labels for the data point, that indicates which
		peak each pixel of the image belongs to, and the list of peaks.
	"""
	skipped = 0
	peaks = []
	labels = np.zeros(data.shape[0], dtype=int) - 1
	bar = Bar("Processing", max=data.shape[0])
	for i in range(data.shape[0]):
		bar.next()
		if labels[i] != -1:
			skipped += 1
			continue

		peak, cpts = findpeak_opt(data, i, r, c, tree)

		if tree == None:
			distances = spatial.distance.cdist(peak, data)
			neighbors_in_range = np.where(distances <= r)[1]
		else:
			neighbors_in_range = tree.query_ball_point(peak, r)[0]

		if len(peaks) == 0:
			peaks.append(peak)
			labels[i] = len(peaks) - 1
		else:
			distances = spatial.distance.cdist(peak, np.array(peaks).reshape(len(peaks), peak.size))[0]
			min_dist = np.min(distances)
			if min_dist < r/2.0:
				labels[i] = np.argmin(distances)
			else:
				peaks.append(peak)
				labels[i] = len(peaks) - 1

		labels[neighbors_in_range] = labels[i]
		labels[cpts] = labels[i]

	bar.finish()
	logger.info("Skipped {:.2f}% of pixels".format(100*(skipped/float(data.shape[0]))))

	return labels, np.array(peaks).reshape(len(peaks), peak.size)

def plotclusters2D(data, labels, peaks):
	"""
	Plots the modes of the given image data in 2D by coloring each pixel
	according to its corresponding peak.

	Args:
		data: image data.
		labels: a list of labels, one for each pixel.
		peaks: a list of vectors, whose first three components can
		be interpreted as BGR values.
	"""
	fig = plt.figure()
	ax = fig.add_subplot(111)
	bgr_peaks = np.array(peaks[:, 0:3])
	rgb_peaks = bgr_peaks[...,::-1]
	rgb_peaks /= 255.0
	for idx, peak in enumerate(rgb_peaks):
		cluster = data[:, np.where(labels == idx)[0]]
		ax.scatter(cluster[0], cluster[1], c=[peak])
	fig.show()

def imSegment(im, r, c=4.0, use_spatial_features=False):
	"""
	Segments the given image with the mean shift algorithm.

	Args:
		im: the image to segment.
		r: the window size for the mean shift algorithm.
		c: denominator of search path window.
		use_spatial_features: if True, include spatial image information
		in the feature vectors. If False, only include LAB color values.

	Returns:
		The segmented image, where each pixel has been replaced with the color of
		its corresponding peak, the list of labels for the pixels and the list
		of peaks found by the mean shift algorithm.
	"""
	orig_img = np.array(im)

	# Blur image
	im = cv2.GaussianBlur(im, (5,5), 5.0)

	# Convert rgb to lab
	im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

	im = im.reshape(im.shape[0]*im.shape[1], im.shape[2])

	if use_spatial_features:
		[x, y] = np.meshgrid(range(1, orig_img.shape[1]+1), range(1, orig_img.shape[0]+1))
		x = np.array(x.T.reshape(x.shape[0]*x.shape[1]), dtype=float)
		y = np.array(y.T.reshape(y.shape[0]*y.shape[1]), dtype=float)
		L = np.array([y/np.max(y), x/np.max(x)]).transpose()
		data = np.concatenate((im, L), axis=1)
	else:
		data = np.array(im)

	# Initialize cKDTree
	tree = spatial.cKDTree(data)

	start = datetime.datetime.now()
	labels, peaks = meanshift_opt(data, r, c, tree)
	#labels, peaks = meanshift(data, r, tree)
	end = datetime.datetime.now()
	time_elapsed = (end-start).total_seconds()
	logger.info("Time elapsed: {} seconds".format(time_elapsed))

	peaks[:, 0:3] = cv2.cvtColor(np.array([peaks[:, 0:3]], dtype=np.uint8), cv2.COLOR_LAB2BGR)[0]
	peaks = np.array(peaks, dtype=np.uint8)

	im = peaks[:, 0:3][labels]
	im = im.reshape(orig_img.shape[0], orig_img.shape[1], orig_img.shape[2])

	return im, labels, peaks

if __name__ == "__main__":
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
		description="Applies mean shift to given image file. Runs predefined window sizes on the image if window size is not explicitly specified.")
	parser.add_argument("img_file", type=str, help="image to process")
	parser.add_argument("--r", type=int, default=None, help="size of the search window")
	parser.add_argument("--c", type=int, default=4, help="denominator for path window size. " +
		"Points within r/c of the search path will be associated with corresponding peak")
	parser.add_argument("--use_spatial_features", help="use x/y coordinates in addition to color space", action="store_true")
	parser.add_argument("--output_dir", type=str, default="output", help="directory to write output to")
	parser.add_argument("--logfile_name", type=str, default="log.txt", help="name of the log file")
	args = parser.parse_args()

	if not os.path.isdir(args.output_dir):
		os.makedirs(args.output_dir)

	logging.basicConfig(filename=os.path.join(args.output_dir, args.logfile_name), level=logging.INFO, filemode="w")
	logger = logging.getLogger(__name__)
	logger.addHandler(logging.StreamHandler())

	im = cv2.imread(args.img_file)
	#im = cv2.resize(im, (0,0), fx=0.2, fy=0.2)

	if args.r != None:
		# Results are not saved in files when using custom window sizes.
		# Run script while in python shell and use result variables afterwards for whatever you want.
		logger.info("Processing {} with r={}, c={}, using spatial features: {}".format(args.img_file, args.r, args.c, args.use_spatial_features))

		segIm, labels, peaks = imSegment(im=im, r=args.r, c=args.c, use_spatial_features=args.use_spatial_features)
	else:
		savepoint = 9
		counter = 0
		rs = [4, 8, 16, 32]
		cs = [4, 8, 16]
		flags = [False, True]

		for r in rs:
			for c in cs:
				for f in flags:
					counter += 1
					if counter <= savepoint:
						continue
					logger.info("Processing {} with r={}, c={}, using spatial features: {}".format(args.img_file, r, c, f))
					segIm, labels, peaks = imSegment(im=im, r=r, c=c, use_spatial_features=f)

					peaks_filename = os.path.join(args.output_dir, "peaks_r{}_c{}_{}.txt".format(r, c, "3D" if f == False else "5D"))
					labels_filename = os.path.join(args.output_dir, "labels_r{}_c{}_{}.txt".format(r, c, "3D" if f == False else "5D"))
					img_filename = os.path.join(args.output_dir, "img_r{}_c{}_{}.jpg".format(r, c, "3D" if f == False else "5D"))
					
					np.savetxt(peaks_filename, peaks)
					np.savetxt(labels_filename, labels)
					cv2.imwrite(img_filename, segIm)