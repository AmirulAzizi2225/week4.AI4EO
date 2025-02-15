# Week 4 AI4EO Assignment
## Echo Classification: Leads vs. Sea Ice and standard deviation of both classes

# Overview
This Github project focuses on classifying echoes into two distinct categories: Leads and Sea Ice, using unsupervised learning methods. The classification is performed on Sentinel-2 optical data and Sentinel-3 altimetry data, providing insights into the effectiveness of different cluster techniques. The results are further validated by comparing our computed model against the ESA official classification using a confusion matrix. 

# Key working steps
1. Data Preprocessing
   - Load and clean the satellite data.
   - Extract echo features relevant for classification.
   
2. Unsupervised Classification (K-means Clustering) and Gaussian Mixture Models
   - Apply K-means and GMM clustering to distinguish between Leads and Sea Ice.
   - Visualise and analyse cluster separation.

3. Computation of Echo Shapes
   - Calculate average echo shape and standard deviation for each class.
   
4. Validation with ESA Classification
   - Compare our results with ESA’s official classification using a confusion matrix.
   - Compute performance matrix (accuracy, precision, recall, F1-score).

# Dataset
Two different datasets will be used  to leverage the ability of the different clusters in classifying sea-ice and leads:

Satellite Data: Sentinel-2 optical data & Sentinel-3 altimetry data.

# Data preprocessing 
We will be using several data preprocessing functions to extract relevant features for clustering and classification.

1. peakiness:
This function will calculates the peakiness of a waveform, which is useful for classifying altimetry echoes.

2. unpack_gpod(variable):
This function extracts and processes satellite altimetry data from a netCDF file

3. calculate_SSD(RIP):
This function calculates Surface Spread Deviation (SSD), a measure of the spread of a waveform.

For this assignement, we will be utilising two widely-used clustering types, K-means clustering and Gausssian Mixture Models (GMM) clustering. Here are some brief explanation on both clustering methods:

# Cluster Models
## K-means clustering 
K-means clustering is a type of unsupervised learning algorithm capable of splitting complex datasets into several k groups, where k represents the number of groups pre-specified by the analyst. It classifies the data points based on the similarity of the features of the data (macqueen1967some). The basic idea is to define k centroids, one for each cluster, and then assign each data point to the nearest centroid, while keeping the centroids as small as possible.

The process begins by selecting the number of clusters (K) and randomly initializing K centroids. Each data point is then assigned to the nearest centroid, after which the centroids are updated based on the assigned points. This iterative process continues until the centroids no longer change, indicating convergence. While K-Means is efficient for large datasets, it is sensitive to centroid initialization and assumes that clusters are spherical and well-separated. The optimal number of clusters (K) can be determined using techniques like the Elbow Method or Silhouette Score. Despite its limitations, K-Means is widely used in customer segmentation, image segmentation, anomaly detection, and text mining, making it a powerful tool for various real-world applications.

## Gaussian Mixture Models (GMM)
Gaussian Mixture Models (GMM) are a probabilistic model used for representing normally distributed subpopulations within a larger dataset. Unlike hard clustering methods such as K-Means, which assign each data point to a single cluster, GMM is a soft clustering technique, meaning that each point has a probability of belonging to multiple clusters rather than being strictly assigned to just one. The model assumes that the data is generated from a mixture of several Gaussian (normal) distributions, each characterized by its own mean (μ), variance (σ²), and weight (π), which determines its contribution to the overall distribution (Reynolds, 2009; McLachlan & Peel, 2004).

GMMs are particularly powerful because they can model complex, multi-modal distributions by combining multiple simpler Gaussian distributions. The parameters of these Gaussian components are typically estimated using the Expectation-Maximization (EM) algorithm, which iteratively refines the model by estimating the probabilities of cluster assignments (E-step) and optimizing the parameters of the Gaussian distributions (M-step) until convergence is reached. This flexibility allows GMMs to capture elliptical or overlapping clusters, making them more versatile than K-Means, which assumes spherical clusters of equal variance.

Due to their ability to perform both clustering and density estimation, GMMs are widely used in speech recognition, anomaly detection, image segmentation, and financial modeling. They are particularly useful in scenarios where data points may belong to multiple categories with varying degrees of certainty, making them well-suited for probabilistic classification and unsupervised learning tasks.

# Results

## K-Means clustering on Sentinel-2 data
![kmeans_clustering](https://github.com/user-attachments/assets/f02e7df9-52d4-4015-b17b-f7189fb96632)

## GMM clustering on Sentinel-2 data
![GMM_clustering](https://github.com/user-attachments/assets/8cdac3bf-31b6-4c7a-9524-864898852693)

We can observe that in this case, GMM cluster does a much better job than k-means cluster at classfying sea-ice and leads for Sentinel-2 optical data.

## Mean and standard deviation of sea-ice and lead using K-means clustering
![mean_std_kmeans](https://github.com/user-attachments/assets/d74573bf-edb5-4de1-8597-4628440dd1a7)

## Mean and standard deviation of sea-ice and lead using GMM clustering
![mean_std_GMM](https://github.com/user-attachments/assets/a1e6de40-86b5-457e-8ce3-ae3c95757808)


## Waveform alignment using cross-correlation on K-means clustering
![10equally_spaced_kmeans](https://github.com/user-attachments/assets/44ffc600-62be-4efc-be6b-075d8715c01d)


## Waveform alignment using cross-correlation on GMM clustering
![10equally_spaced_GMM](https://github.com/user-attachments/assets/0419e4ef-59cc-4b72-9304-531a0fd3c3c7)

## How number of intervals affect our waveform alignment
For the sake of observing the trend when interval increases, we can align waveforms with 100 equally spaced fucntion where cluster_gmm = 0
![100equally_spaced_GMM](https://github.com/user-attachments/assets/6984fa22-c6cc-4b27-97e0-7a26a7522cfb)


## Quantification of echoes against ESA classification using confusion matrix on K-means clustering
Confusion Matrix:
[[   0 4885   10 3983]
 [1527    0 1755   35]
 [   0    0    0    0]
 [   0    0    0    0]]

Classification Report:
              precision    recall  f1-score   support

         0.0       0.00      0.00      0.00    8878.0
         1.0       0.00      0.00      0.00    3317.0
         2.0       0.00      0.00      0.00       0.0
         3.0       0.00      0.00      0.00       0.0

    accuracy                           0.00   12195.0
   macro avg       0.00      0.00      0.00   12195.0
weighted avg       0.00      0.00      0.00   12195.0

## Quantification of echoes against ESA classification using confusion matrix on GMM clustering

Confusion Matrix:
[[8856   22]
 [  24 3293]]

Classification Report:
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00      8878
         1.0       0.99      0.99      0.99      3317

    accuracy                           1.00     12195
   macro avg       1.00      1.00      1.00     12195
weighted avg       1.00      1.00      1.00     12195


Author: Muhammad Amirul Haziq Bin Azizi (https://github.com/AmirulAzizi2225)


