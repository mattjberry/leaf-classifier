# Histogram of Curvature Scale (HoCS) Leaf Classifier
#
# This script:
#   - Implements and tests a Histogram of Curvature Scale (HoCS) descriptor.
#   - Extracts HoCS descriptors for training images in a leaf dataset.
#   - Prepares a K-Nearest-Neighbours (KNN) classifier from training descriptors.
#   - Extracts HoCS descriptors for test images in the leaf dataset.
#   - Classifies test image descriptors into one of three leaf shape classes.

import os
import skimage
import skimage.io as io
import skimage.morphology as morph
import skimage.measure as measure
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as neigh
import sklearn.metrics as metrics
from scipy import ndimage


# =============================================================================
# Histogram of Curvature Scale (HoCS) Function
#
# Implements a HoCS function that returns a feature vector for a given binary
# region. Curvature is computed using the normalized area integral invariant
# method. A histogram of curvature is computed at each scale from min_scale to
# max_scale (in steps of increment), each with num_bins bins. Histograms are
# normalized and concatenated into a single 1D feature vector.
# =============================================================================

def HoCS(B, min_scale, max_scale, increment, num_bins):
    '''
    Computes a histogram of curvature scale for the shape in the binary image B.
    Boundary fragments due to holes are ignored.
    :param B: A binary image consisting of a single foreground connected component.
    :param min_scale: smallest scale to consider
    :param max_scale: largest scale to consider
    :param increment:  increment on which to compute scales between min_scale and max_scale
    :param num_bins: number of bins for the histogram at each scale
    :return: 1D array of histograms concatenated together in order of increasing scale.

    Pre-conditons
     - min_scale > 1
     - min_scale < max_scale
    '''
    B = B.astype(bool)
    result = []

    labels = measure.label(B, connectivity=1)
    feats = measure.regionprops(labels)
    region_area = feats[0].area

    region_coords = feats[0].coords
    perimeter = feats[0].perimeter

    # get the outer boundary and ignore holes
    filled = ndimage.binary_fill_holes(B)
    boundary = filled & ~ndimage.binary_erosion(filled)

    # Get boundary pixel coordinates
    boundary_coords = np.argwhere(boundary)

    # compute curvatures at each scale
    for i in range(min_scale, max_scale + 1, increment):
        mask = morph.disk(i)
        mask_area = np.sum(mask)

        # find the intersection of mask region around each boundary pixel
        intersections = ndimage.convolve(B.astype(float), mask.astype(float), mode='constant', cval=0)

        # compute curvature kp for each point
        # kp = area(mask_area intersection region) / mask_area
        curvatures = []
        for coord in boundary_coords:
            y, x = coord
            area_intersection = intersections[y, x]
            kp = area_intersection / mask_area
            curvatures.append(kp)

        curvatures = np.array(curvatures)

        # compute histogram and normalize
        hist, stuff_i_dont_need = np.histogram(curvatures, bins=num_bins, range=(0.0, 1.0))
        norm_hist = hist / np.sum(hist) if np.sum(hist) > 0 else hist

        # append to result vector
        result.append(norm_hist)

    # collapse result into a feature vector
    feat_vec = np.concatenate(result)

    return feat_vec


# =============================================================================
# Smoke test for HoCS function
#
# Runs HoCS on image_0001.png from the leaftraining directory using
# min_scale=5, max_scale=25, increment=10, num_bins=10. Plots the resulting
# feature vector as a bar graph with y-axis limits [0.0, 1.0].
# =============================================================================

img = skimage.io.imread('leaftraining/leaftraining/image_0001.png')
hocs = HoCS(img, 5, 25, 10, 10)

plt.figure(figsize=(10, 8))
plt.bar(np.arange(len(hocs)), hocs)
plt.ylim([0.0, 1.0])
plt.show()

# =============================================================================
# Calculate training features
#
# Computes the HoCS feature vector for each training image listed in
# leaftraining.csv. Also generates training labels (1, 2, or 3) indicating
# which of the three leaf classes each sample belongs to.
# =============================================================================

# read in the images listed in leaftraining.csv and compute descriptors for them using HoCS() function.
training = pd.read_csv('leaftraining/leaftraining.csv', header=None)
training = training[0].tolist()

labels = (np.arange(30) // 10) + 1

X_training = []  # matrix where each row in an HCoS vector
y_training = []  # label array

for sample, label in zip(training, labels):
    image = io.imread('leaftraining/leaftraining/' + sample)
    sample_hcos = HoCS(image, 2, 25, 3, 12)

    X_training.append(sample_hcos)
    y_training.append(label)

X_training = np.array(X_training)
y_training = np.array(y_training)


# =============================================================================
# Train KNN classifier
#
# Trains a KNN classifier using the feature vectors from the training images.
# n_neighbors=1 was found to give the best classification rate. Using more
# neighbors produced increasingly worse results.
# Note: HoCS parameters and KNN parameters can be tuned
# independently without needing to recompute features each time.
# =============================================================================

# Train the KNN classifier
knn = neigh.KNeighborsClassifier(n_neighbors=1)
knn.fit(X_training, y_training)


# =============================================================================
# Calculate the testing features
#
# Computes HoCS features for all testing images listed in leaftesting.csv,
# using the same HoCS parameters. Also generates true class labels
# for the test images (49 from class 1, 29 from class 2, 52 from class 3).
# =============================================================================

# use the filenames in leaftesting.csv to load each image and process it
testing = pd.read_csv('leaftesting/leaftesting.csv', header=None)
testing = testing[0].tolist()

labels = np.concatenate([
    np.full(49, 1),
    np.full(29, 2),
    np.full(52, 3)
])

X_testing = []  # matrix where each row in an HCoS vector
y_testing = []  # label array

for sample, label in zip(testing, labels):
    image = io.imread('leaftesting/leaftesting/' + sample)
    sample_hcos = HoCS(image, 2, 25, 3, 12)

    X_testing.append(sample_hcos)
    y_testing.append(label)

X_testing = np.array(X_testing)
y_testing = np.array(y_testing)


# =============================================================================
# Classify the testing features
#
# Classifies test image descriptors using the trained KNN classifier. Computes
# the confusion matrix and overall classification rate. Prints filenames of
# any misclassified images. Target: >90% classification rate
# =============================================================================

y_predicted = knn.predict(X_testing)
accuracy = metrics.accuracy_score(y_testing, y_predicted)
confusion_matrix = metrics.confusion_matrix(y_testing, y_predicted)

misclassified_indices = np.where(y_testing != y_predicted)[0]
misclassified_files = [testing[i] for i in misclassified_indices]

# display results
print("Misclassified samples:")
for file in misclassified_files:
    print(file)

print("Confusion matrix")
print(confusion_matrix)

print(f"Classification Rate: {accuracy:.2%}")


# =============================================================================
# Reflections
#
# HoCS Parameters:
#   min_scale=2, max_scale=25, increment=3, num_bins=12 were chosen so the low
#   end captures sharp corners (class 1) and the high end captures overall
#   rounded shape (class 2) vs. straighter edges (classes 1 & 3). A step size of 3
#   keeps resolution fine without excessive iterations. An initially tighter
#   scale range was expanded after poor results with too-high a minimum and
#   too-small a maximum.
#
# KNN Parameters:
#   n_neighbors=1 gave the best classification rate. More neighbors produced
#   increasingly worse results, likely because many leaf images are similar in
#   size and shape, making the single-nearest-neighbor approach most precise.
#
# Misclassified Images:
#   image_0060.png, image_0066.png, image_0127.png, image_0140.png,
#   image_0173.png, image_0185.png. The first two are nearly identical and were
#   both misclassified. The latter four appear to have significant holes or edge
#   deformations that skew the HoCS descriptor. No strong class-level pattern
#   was observed among the misclassified images.
# =============================================================================
