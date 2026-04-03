# GLCM and LBP Texture Descriptor Classifier
#
# This script:
#   - Computes GLCM and LBP texture descriptors for a training dataset.
#   - Computes GLCM and LBP texture descriptors for a test dataset.
#   - Trains two KNN classifiers: one using GLCM features, one using LBP features.
#   - Classifies test image descriptors using both classifiers.
#   - Displays confusion matrices, misclassified filenames, and classification rates.

import numpy as np
import pandas as pd
import skimage.feature as feature
import skimage.io as io
import sklearn.neighbors as knn
import sklearn.metrics as metrics


# =============================================================================
# Compute texture descriptors for the training images
#
# For each training image, computes:
#   - GLCM features (contrast, dissimilarity, homogeneity, energy, correlation)
#     using distances [1, 3, 5] and angles [0, pi/4, pi/2, 3pi/4], stored as
#     a 120-row array.
#   - Rotationally invariant LBP features (P=8, R=1, method='uniform') as a
#     normalized 10-bin histogram.
#   - LBP variance features (method='var') as a 16-bin histogram (range 0-7000).
#   - LBP uniform and VAR histograms are concatenated into a 26-element vector,
#     stored as a 120-row array.
# =============================================================================

training = pd.read_csv('brodatztraining/brodatztraining.csv', header=None)
training = training[0].tolist()

lbp_features = []
glcm_features = []

for sample in training:
    img = io.imread('brodatztraining/brodatztraining/' + sample)
    img = img.astype(np.uint8)

    # get GLCMs with distances [1, 3, 5] and four angles (from scikit-image docs example)
    glcms = feature.graycomatrix(img, [1, 3, 5], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], normed=True)

    # extract GLCM properties
    contrast = feature.graycoprops(glcms, 'contrast').flatten()
    dissimilarity = feature.graycoprops(glcms, 'dissimilarity').flatten()
    homogeneity = feature.graycoprops(glcms, 'homogeneity').flatten()
    energy = feature.graycoprops(glcms, 'energy').flatten()
    correlation = feature.graycoprops(glcms, 'correlation').flatten()

    # combine into a single GLCM feature vector
    glcm_feature_vector = np.concatenate([contrast, dissimilarity, homogeneity, energy, correlation])
    glcm_features.append(glcm_feature_vector)

    # rotationally invariant LBP histogram (10 bins for P=8)
    lbp = feature.local_binary_pattern(img, 8, 1, method='uniform')
    lbp_hist, things_i_dont_need = np.histogram(lbp, bins=10, range=(0, 10))

    # LBP variance feature histogram (16 bins)
    lbp_var = feature.local_binary_pattern(img, 8, 1, method='var')
    var_hist, things_i_really_dont_need = np.histogram(lbp_var, bins=16, range=(0, 7000))

    # concatenate LBP uniform + VAR into a 26-element feature vector
    lbp_result = np.concatenate([lbp_hist, var_hist])
    lbp_features.append(lbp_result)

glcm_features = np.array(glcm_features)
lbp_features = np.array(lbp_features)


# =============================================================================
# Compute test image features
#
# Computes the exact same GLCM and LBP features as Step 1 for each test image.
# Results are stored in 320-row arrays (one row per test sample).
# Expected performance: GLCM ~65%+, LBP ~95%+.
# =============================================================================

testing = pd.read_csv('brodatztesting/brodatztesting.csv', header=None)
testing = testing[0].tolist()

lbp_features_testing = []
glcm_features_testing = []

for sample in testing:
    img = io.imread('brodatztesting/brodatztesting/' + sample)
    img = img.astype(np.uint8)

    # GLCM features (same as training)
    glcms = feature.graycomatrix(img, [1, 3, 5], [0, np.pi/4, np.pi/2, 3*np.pi/4], normed=True)
    contrast = feature.graycoprops(glcms, 'contrast').flatten()
    dissimilarity = feature.graycoprops(glcms, 'dissimilarity').flatten()
    homogeneity = feature.graycoprops(glcms, 'homogeneity').flatten()
    energy = feature.graycoprops(glcms, 'energy').flatten()
    correlation = feature.graycoprops(glcms, 'correlation').flatten()

    glcm_feature_vector = np.concatenate([contrast, dissimilarity, homogeneity,
                                          energy, correlation])
    glcm_features_testing.append(glcm_feature_vector)

    # LBP features (same as training)
    lbp = feature.local_binary_pattern(img, 8, 1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))

    lbp_var = feature.local_binary_pattern(img, 8, 1, method='var')
    var_hist, _ = np.histogram(lbp_var, bins=16, range=(0, 7000))

    lbp_result = np.concatenate([lbp_hist, var_hist])
    lbp_features_testing.append(lbp_result)

glcm_features_testing = np.array(glcm_features_testing)
lbp_features_testing = np.array(lbp_features_testing)


# =============================================================================
# Generate label arrays for training and testing data
#
# Labels are integers 1-8 representing texture classes. Training images are
# in blocks of 15 per class (120 total); test images are in blocks of 40 per
# class (320 total).
# =============================================================================

training_labels = (np.arange(120) // 15) + 1
testing_labels = (np.arange(320) // 40) + 1


# =============================================================================
# Train KNN classifiers
#
# Trains two separate KNN classifiers: one on GLCM features and one on LBP
# features. n_neighbors=1 was carried over from Q1 and left unchanged, as it
# produced good results.
# =============================================================================

# Train GLCM classifier
knn_glcm = knn.KNeighborsClassifier(n_neighbors=1)
knn_glcm.fit(glcm_features, training_labels)

# Train LBP classifier
knn_lbp = knn.KNeighborsClassifier(n_neighbors=1)
knn_lbp.fit(lbp_features, training_labels)


# =============================================================================
# Predict the classes of the test images
#
# Uses both trained KNN classifiers to predict class labels for all test images.
# =============================================================================

glcm_predict = knn_glcm.predict(glcm_features_testing)
lbp_predict = knn_lbp.predict(lbp_features_testing)


# =============================================================================
# Display results
#
# For each classifier, prints: filenames of misclassified images, the confusion
# matrix, and the overall classification rate.
# =============================================================================

# GLCM results
glcm_accuracy = metrics.accuracy_score(testing_labels, glcm_predict)
glcm_confusion_matrix = metrics.confusion_matrix(testing_labels, glcm_predict)

misclassified_indices = np.where(testing_labels != glcm_predict)[0]
misclassified_files = [testing[i] for i in misclassified_indices]

print("GLCM Misclassified samples:")
for file in misclassified_files:
    print(file)

print("GLCM Confusion matrix")
print(glcm_confusion_matrix)

print(f"GLCM Classification Rate: {glcm_accuracy:.2%}")

# LBP results
lbp_accuracy = metrics.accuracy_score(testing_labels, lbp_predict)
lbp_confusion_matrix = metrics.confusion_matrix(testing_labels, lbp_predict)

misclassified_indices = np.where(testing_labels != lbp_predict)[0]
misclassified_files = [testing[i] for i in misclassified_indices]

print("LBP Misclassified samples:")
for file in misclassified_files:
    print(file)

print("LBP Confusion matrix")
print(lbp_confusion_matrix)

print(f"LBP Classification Rate: {lbp_accuracy:.2%}")


# =============================================================================
# Reflections
#
# Performance comparison:
#   LBP achieved 98.75% accuracy with no parameter tuning. GLCM required more
#   experimentation and lagged behind. Since GLCM captures the repeated grey-level
#   co-occurrence patterns it suits images with clear directional structure, but
#   LBP's rotational invariance makes it more robust overall — consistent with
#   theory that LBP generally outperforms GLCM.
#
# Misclassified images:
#   - GLCM: Errors were widespread across classes. Classes 3 (rock) and 4
#     (fabric) performed worst. The fabric had strong lines at varying rotations,
#     which may confuse GLCM's directional features. The rocky texture had high
#     intra-class variation in angle and shadow.
#   - LBP: Only 4 misclassified images, in two pairs. One pair appeared to be
#     concrete images with a shadow in the corner; the other pair appeared to be
#     grass. Grass is difficult due to non-uniform texture patterns. The concrete
#     pair may have failed due to angle differences, though LBP should be
#     rotation-invariant — the exact cause is unclear.
# =============================================================================
