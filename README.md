# Object & Texture Classifiers

This repo contains two image classification pipelines built using Python and scikit-image. Both use a K-Nearest Neighbours (KNN) classifier to categorize images based on extracted feature descriptors.

---

## Leaf Shape Classification (HoCS)

Classifies leaf images into one of three shape classes using a **Histogram of Curvature Scale (HoCS)** descriptor.

The HoCS descriptor works by computing a normalized histogram of boundary curvature at multiple scales for a binary image. At each scale, a disk-shaped mask is convolved across the image and the fraction of foreground pixels within the disk at each boundary point gives a curvature value. These per-scale histograms are concatenated into a single feature vector.

**Pipeline:**
1. Implements the `HoCS()` function using the normalized area integral invariant method
2. Tests the function on a sample training image and plots the resulting feature vector
3. Extracts HoCS feature vectors for all training images (`leaftraining/`)
4. Trains a KNN classifier (`n_neighbors=1`) on the training features
5. Extracts HoCS feature vectors for all test images (`leaftesting/`)
6. Classifies test images and reports the confusion matrix, misclassified filenames, and classification rate

**Results:** Achieves ~95% classification rate on the test set.

**Dependencies:** `numpy`, `scipy`, `scikit-image`, `scikit-learn`, `matplotlib`, `pandas`

---

## Texture Classification (GLCM and LBP)

Classifies texture images into one of eight classes using two different texture descriptors — **Grey-Level Co-occurrence Matrix (GLCM)** and **Local Binary Patterns (LBP)** — and compares their performance.

**GLCM** captures texture by measuring how often pairs of pixel intensities appear at a given distance and angle. Five properties (contrast, dissimilarity, homogeneity, energy, correlation) are extracted across three distances and four angles to form a feature vector per image.

**LBP** captures texture by comparing each pixel to its neighbours and encoding the result as a binary pattern. A rotationally invariant uniform LBP histogram (10 bins) is combined with an LBP variance histogram (16 bins) to form a 26-element feature vector per image.

**Pipeline:**
1. Computes GLCM and LBP feature vectors for all training images (`brodatztraining/`)
2. Computes the same features for all test images (`brodatztesting/`)
3. Generates class labels for training (blocks of 15) and testing (blocks of 40)
4. Trains two separate KNN classifiers (`n_neighbors=1`), one for each feature type
5. Predicts class labels for all test images using both classifiers
6. Reports confusion matrices, misclassified filenames, and classification rates for each

**Results:** LBP achieves ~99% classification rate; GLCM achieves ~79%.

**Dependencies:** `numpy`, `pandas`, `scikit-image`, `scikit-learn`

---

## Requirements

Install all dependencies with:

```bash
pip3 install numpy pandas scipy scikit-image scikit-learn matplotlib
```
