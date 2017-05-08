import logging
import cv2
import sys
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from progress.bar import Bar

def findpeak(data, idx, r):

	t = 0.01
	shift = sys.maxint

	# Get point of interest
	point = data[:, idx]
	point = point.reshape(1, point.size)

	while shift > t:		

		# Compute distance to all points
		distances = cdist(point, data.transpose())

		# Determine points in window
		found = np.array(distances <= r, dtype=int)
		close_points = data[:, np.where(found == 1)[0]].T

		# Calculate new mean
		new_point = np.mean(close_points, axis=0)
		new_point = new_point.reshape(1, new_point.size)

		#Calculate shift distance
		shift = cdist(point, new_point.reshape(1, point.size)).item(0)
		point = new_point

	return point

def meanshift(data, r):
	peaks = []
	labels = np.zeros(data.shape[1]) - 1
	for i in range(data.shape[1]):
		print("Processing pixel {}".format(i))
		peak = findpeak(data, i, r)


		neighbor_dist = cdist(peak, data.transpose())[0]
		neighbors_in_range = np.where(neighbor_dist <= r)[0]

		if len(peaks) == 0:
			peaks.append(peak)
			labels[i] = len(peaks) - 1
		else:
			distances = cdist(peak, np.array(peaks).reshape(len(peaks), peak.size))[0]
			min_dist = np.min(distances)
			if min_dist < r/2.0:
				labels[i] = np.argmin(distances)
			else:
				peaks.append(peak)
				labels[i] = len(peaks) - 1

	return labels, np.array(peaks).reshape(len(peaks), peak.size)

def findpeak_opt(data, idx, r, c):

	t = 0.01
	shift = sys.maxint
	cpts = np.zeros(data.shape[1])

	# Get point of interest
	point = data[:, idx]
	point = point.reshape(1, point.size)

	while shift > t:		

		# Compute distance to all points
		distances = cdist(point, data.transpose())[0]

		# Determine points in window
		found = np.array(distances <= r, dtype=int)
		close_points = data[:, np.where(found == 1)[0]].T

		find_points = np.array(distances <= r/c, dtype=int)
		cpts = np.array((cpts + find_points) > 0, dtype=int)

		# Calculate new mean
		new_point = np.mean(close_points, axis=0)
		new_point = new_point.reshape(1, new_point.size)

		#Calculate shift distance
		shift = cdist(point, new_point.reshape(1, point.size)).item(0)
		point = new_point

	return point, np.where(cpts == 1)[0]

def meanshift_opt(data, r, c):
	#logger = logging.getLogger(__name__)
	skipped = 0
	peaks = []
	labels = np.zeros(data.shape[1], dtype=int) - 1
	bar = Bar("Processing", max=data.shape[1])
	for i in range(data.shape[1]):
		bar.next()
		if labels[i] != -1:
			skipped += 1
			continue

		peak, cpts = findpeak_opt(data, i, r, c)
		neighbor_dist = cdist(peak, data.transpose())[0]
		neighbors_in_range = np.where(neighbor_dist <= r)[0]

		if len(peaks) == 0:
			peaks.append(peak)
			labels[i] = len(peaks) - 1
		else:
			distances = cdist(peak, np.array(peaks).reshape(len(peaks), peak.size))[0]
			min_dist = np.min(distances)
			if min_dist < r/2.0:
				labels[i] = np.argmin(distances)
			else:
				peaks.append(peak)
				labels[i] = len(peaks) - 1

		labels[neighbors_in_range] = labels[i]
		labels[cpts] = labels[i]

	bar.finish()
	logger.info("Skipped {:.2f}% of pixels".format(100*(skipped/float(data.shape[1]))))

	return labels, np.array(peaks).reshape(len(peaks), peak.size)

def plotclusters(data, labels, means):
	pass

def imSegment(im, r, c=4.0, use_spatial_features=False):
	#logger = logging.getLogger(__name__)
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
		data = np.concatenate((im, L), axis=1).transpose()
	else:
		data = im.transpose()

	start = datetime.datetime.now()
	labels, peaks = meanshift_opt(data, r, c)
	#labels, peaks = meanshift(data, r)
	end = datetime.datetime.now()
	time_elapsed = (end-start).total_seconds()
	logger.info("Time elapsed: {} seconds".format(time_elapsed))

	converted_peaks = cv2.cvtColor(np.array([peaks[:, 0:3]], dtype=np.uint8), cv2.COLOR_LAB2BGR)[0]

	im = converted_peaks[labels]
	im = im.reshape(orig_img.shape[0], orig_img.shape[1], orig_img.shape[2])

	return im, labels, peaks

if __name__ == "__main__":
	logging.basicConfig(filename="logging.log", level=logging.INFO, filemode="w")
	logger = logging.getLogger(__name__)
	logger.addHandler(logging.StreamHandler())

	pics = ["pic1", "pic2", "pic3"]
	rs = [4, 8, 16, 32]
	cs = [4, 8, 16]
	flags = [False, True]
	for pic in pics:
		im = cv2.imread(pic + ".jpg")
		#im = cv2.resize(im, (0,0), fx=0.2, fy=0.2) # for debugging
		for r in rs:
			for c in cs:
				for f in flags:
					logger.info("Processing {} with r={}, c={}, using spatial features: {}".format(pic, r, c, f))
					segIm, labels, peaks = imSegment(im=im, r=r, c=c, use_spatial_features=f)

					peaks_file = "output_{}/peaks_r{}_c{}_{}.txt".format(pic, r, c, "3D" if f == False else "5D")
					labels_file = "output_{}/labels_r{}_c{}_{}.txt".format(pic, r, c, "3D" if f == False else "5D")
					img_file = "output_{}/img_r{}_c{}_{}.jpg".format(pic, r, c, "3D" if f == False else "5D")
					
					np.savetxt(peaks_file, peaks)
					np.savetxt(labels_file, labels)
					cv2.imwrite(img_file, segIm)