import cv2
import numpy as np
from scipy.spatial.distance import cdist
import sys
import matplotlib.pyplot as plt

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
		close_points = []

		for i, dist in enumerate(distances[0]):
			if dist <= r:
				close_points.append(data[:, i])

		close_points = np.array(close_points)

		# Calculate new mean
		new_point = np.mean(close_points, axis=0)
		new_point = new_point.reshape(1, new_point.size)

		#Calculate shift distance
		shift = cdist(point, new_point.reshape(1, 3)).item(0)
		point = new_point

	return point

def meanshift(data, r):
	peaks = []
	labels = np.zeros(data.shape[1]) - 1
	for i in range(data.shape[1]):
		peak = findpeak(data, i, r)

		merged = False

		if len(peaks) > 0:
			distances = cdist(peak, np.array(peaks).reshape(len(peaks), peak.size))
			for k, dist in enumerate(distances[0]):
				if dist < r/2.0:
					# Merge peaks
					merged = True
					labels[i] = k
					break

		if not merged:
			peaks.append(peak)
			labels[i] = len(peaks) - 1

	return labels, np.array(peaks).reshape(len(peaks), peak.size)

def findpeak_opt(data, idx, r):

	t = 0.01
	c = 4.0
	shift = sys.maxint
	cpts = []

	# Get point of interest
	point = data[:, idx]
	point = point.reshape(1, point.size)

	while shift > t:		

		# Compute distance to all points
		distances = cdist(point, data.transpose())

		# Determine points in window
		close_points = []

		for i, dist in enumerate(distances[0]):
			if dist <= r:
				close_points.append(data[:, i])
			if dist <= r/c:
				cpts.append(data[:, i])

		close_points = np.array(close_points)

		# Calculate new mean
		new_point = np.mean(close_points, axis=0)
		new_point = new_point.reshape(1, new_point.size)

		#Calculate shift distance
		shift = cdist(point, new_point.reshape(1, 3)).item(0)
		point = new_point

	return point, cpts

def meanshift_opt(data, r):
	peaks = []
	labels = np.zeros(data.shape[1]) - 1
	for i in range(data.shape[1]):
		if labels[i] != -1:
			continue

		peak, cpts = findpeak_opt(data, i, r)

		merged = False

		if len(peaks) > 0:
			distances = cdist(peak, np.array(peaks).reshape(len(peaks), peak.size))
			for k, dist in enumerate(distances[0]):
				if dist < r/2.0:
					# Merge peaks
					merged = True
					labels[i] = k
					break

		if not merged:
			peaks.append(peak)
			labels[i] = len(peaks) - 1

			# Basin of Attraction
			neighbor_dist = cdist(peak, data.transpose())
			for k, dist in enumerate(neighbor_dist[0]):
				if dist <= r:
					labels[k] = labels[i]

	return labels, np.array(peaks).reshape(len(peaks), peak.size)

def plotclusters(data, labels, means):
	pass

def imSegment(im, r):
	pass
	# Blur image
	#im = cv2.GaussianBlur(im, (5,5), 5.0)

	# Convert rgb to lab
	#im = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)

	#labels, peaks = meanshift_opt(data, r)

if __name__ == "__main__":
	pass