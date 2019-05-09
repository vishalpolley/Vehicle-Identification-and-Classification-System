import cv2
import numpy as np
from glob import glob
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from matplotlib import pyplot as plt

class ImageHelpers:
	def __init__(self):
		# self.orb_object = cv2.ORB_create()
		self.sift_object = cv2.xfeatures2d.SIFT_create()
		# self.surf_object = cv2.xfeatures2d.SURF_create()

	def gray(self, image):
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		return gray

	def harris_corners(self, image):
		'''
		Function : cv2.cornerHarris(image,blocksize,ksize,k)
		Parameters are as follows :
		1. image : the source image in which we wish to find the corners (grayscale)
		2. blocksize : size of the neighborhood in which we compare the gradient
		3. ksize : aperture parameter for the Sobel() Operator (used for finding Ix and Iy)
		4. k : Harris detector free parameter (used in the calculation of R)

		'''
		# Find Harris corners
		im = np.float32(image)
		dst = cv2.cornerHarris(im, 2, 3, 0.04)
		dst = cv2.dilate(dst, None)
		ret, dst = cv2.threshold(dst, 0.01 * dst.max(), 255, 0)
		dst = np.uint8(dst)

		# Find centroids
		ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)

		'''
		Define the criteria to stop and refine the corners. Criteria is such that,
		whenever 100 iterations of algorithm is ran, or an accuracy of epsilon = 0.001
		is reached, stop the algorithm and return the answer.

		'''
		criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
		corners = cv2.cornerSubPix(dst, np.float32(centroids), (5,5), (-1,-1), criteria)

		# Get the keypoints
		keypoints = [cv2.KeyPoint(crd[0], crd[1], 5) for crd in corners]
		return keypoints

	def shi_tomasi(self, image):
		'''
		Function: cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance[, corners[, mask[, blockSize[, useHarrisDetector[, k]]]]])
		Parameters are as follows :
		1. image – Input 8-bit or floating-point 32-bit, single-channel image.
		2. maxCorners – You can specify the maximum no. of corners to be detected. (Strongest ones are returned if detected more than max.)
		3. qualityLevel – Minimum accepted quality of image corners.
		4. minDistance – Minimum possible Euclidean distance between the returned corners.
		5. corners – Output vector of detected corners.
		6. mask – Optional region of interest.
		7. blockSize – Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
		8. useHarrisDetector – Set this to True if you want to use Harris Detector with this function.
		9. k – Free parameter of the Harris detector.

		'''
		im = np.float32(image)
		# Specifying maximum number of corners as 1000
		# 0.01 is the minimum quality level below which the corners are rejected
		# 10 is the minimum euclidean distance between two corners
		corners_img = cv2.goodFeaturesToTrack(im, 1000, 0.01, 10)
		corners_img = np.int0(corners_img)
		keypoints = []
		for corners in corners_img:
			x,y = corners.ravel()
			keypoints.append(cv2.KeyPoint(x, y, 5))
		return keypoints

	def fast_corners(self, image):
		fast = cv2.FastFeatureDetector_create()
		# Applying Gaussian Blurring
		blur = cv2.GaussianBlur(image, (5,5), 0)

		# Detect keypoints with non max suppression
		keypoints = fast.detect(blur, None)
		# Disable nonmaxSuppression
		# fast.setNonmaxSuppression(False)
		# Detect keypoints without non max suppression
		# keypoints_without_nonmax = fast.detect(gray, None)
		return keypoints

	def features(self, image, keypoints):
		# keypoints = self.orb_object.detect(image, None)
		# keypoints, descriptors = self.orb_object.compute(image, keypoints)
		# descriptors = [self.sift_object.compute(image, [kp]) for kp in keypoints]
		keypoints, descriptors = self.sift_object.compute(image, keypoints)
		# keypoints, descriptors = self.surf_object.detectAndCompute(image, None)
		return [keypoints, descriptors]


class BOVHelpers:
	def __init__(self, n_clusters = 20, max_iter = 600):
		self.n_clusters = n_clusters
		self.kmeans_obj = KMeans(n_clusters = n_clusters, max_iter = max_iter)
		self.kmeans_ret = None
		self.descriptor_vstack = None
		self.mega_histogram = None
		self.clf  = SVC(gamma = 'auto')

	def cluster(self):
		"""
		cluster using KMeans algorithm,

		"""
		print(self.kmeans_obj)
		self.kmeans_ret = self.kmeans_obj.fit_predict(self.descriptor_vstack)

	def developVocabulary(self, n_images, descriptor_list, kmeans_ret = None):

		"""
		Each cluster denotes a particular visual word
		Every image can be represeted as a combination of multiple
		visual words. The best method is to generate a sparse histogram
		that contains the frequency of occurence of each visual word

		Thus the vocabulary comprises of a set of histograms of encompassing
		all descriptions for all images

		"""

		self.mega_histogram = np.array([np.zeros(self.n_clusters) for i in range(n_images)])
		old_count = 0
		for i in range(n_images):
			l = len(descriptor_list[i])
			for j in range(l):
				if kmeans_ret is None:
					idx = self.kmeans_ret[old_count+j]
				else:
					idx = kmeans_ret[old_count+j]
				self.mega_histogram[i][idx] += 1
			old_count += l
		print("Vocabulary Histogram Generated")

	def standardize(self, std=None):
		"""
		standardize is required to normalize the distribution
		wrt sample size and features. If not normalized, the classifier may become
		biased due to steep variances.

		"""
		hist_data = np.float64(self.mega_histogram)
		if std is None:
			self.scale = StandardScaler().fit(hist_data)
			# self.scale = QuantileTransformer(output_distribution='normal').fit(hist_data)
			self.mega_histogram = self.scale.transform(hist_data)
		else:
			print("STD not none. External STD supplied")
			self.mega_histogram = std.transform(hist_data)

	def formatND(self, l):
		"""
		restructures list into vstack array of shape
		M samples x N features for sklearn

		"""
		vStack = np.array(l[0])
		for remaining in l[1:]:
			vStack = np.vstack((vStack, remaining))
		self.descriptor_vstack = vStack.copy()
		return vStack

	def train(self, train_labels):
		"""
		uses sklearn.svm.SVC classifier (SVM)

		"""
		print("Training SVM")
		print(self.clf)
		# print("Train labels", train_labels)
		self.clf.fit(self.mega_histogram, train_labels)
		print("Training completed")

	def predict(self, iplist):
		predictions = self.clf.predict(iplist)
		return predictions

	def plotHist(self, vocabulary = None):
		print("Plotting histogram")
		if vocabulary is None:
			vocabulary = self.mega_histogram

		x_scalar = np.arange(self.n_clusters)
		y_scalar = np.array([abs(np.sum(vocabulary[:,h], dtype=np.int32)) for h in range(self.n_clusters)])

		print(y_scalar)

		plt.bar(x_scalar, y_scalar)
		plt.xlabel("Visual Word Index")
		plt.ylabel("Frequency")
		plt.title("Complete Vocabulary Generated")
		plt.xticks(x_scalar + 0.4, x_scalar)
		plt.show()

class FileHelpers:

	def __init__(self):
		pass

	def resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
		"""
		Interpolation methods are cv.INTER_AREA for shrinking.
		cv.INTER_CUBIC(slow) & cv.INTER_LINEAR for zooming.
		cv.INTER_LINEAR for all resizing purposes.

		"""
		dim = None
		(h, w) = image.shape[:2]
		if width is None and height is None:
			return image
		if width is None:
			r = height / float(h)
			dim = (int(w * r), height)
		else:
			r = width / float(w)
			dim = (width, int(h * r))
		resized = cv2.resize(image, dim, interpolation = inter)
		return resized

	def getFiles(self, path):
		"""
		- returns  a dictionary of all files
		having key => value as  objectname => image path

		- returns total number of files.

		"""
		imlist = {}
		count = 0
		for each in glob(path + "*"):
			word = each.split("/")[-1]
			print(" #### Reading image category ", word, " ##### ")
			imlist[word] = []
			for imagefile in glob(path+word+"/*"):
				print("Reading file ", imagefile)
				im = cv2.imread(imagefile, 0)
				(h, w) = im.shape[:2]
				if h > 1024 or w > 1024:
					im = self.resize(im, width = 800)
				print(im.shape)
				imlist[word].append(np.uint8(im))
				count +=1

		return [imlist, count]
