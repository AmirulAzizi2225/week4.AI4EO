# Week 4 AI4EO Assignment
## Echo Classification: Leads vs. Sea Ice and standard deviation of both classes

# Overview
This Github project focuses on classifying echoes into two distinct categories: Leads and Sea Ice, using unsupervised learning methods. The classification is performed on Sentinel-2 optical data and Sentinel-3 altimetry data, providing insights into the effectiveness of different cluster techniques. The results are further validated by comparing our computed model against the ESA official classification using a confusion matrix. 

# Key working steps
1. Data Preprocessing
   - Load and clean the satellite data.
   - Extract echo features relevant for classification.
   
2. Unsupervised Classification (K-means Clustering)
   - Apply K-means clustering to distinguish between Leads and Sea Ice.
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

## Gaussian Mixture Models (GMM)
Gaussian Mixture Models (GMM) are a probabilistic model for representing normally distributed subpopulations within an overall population. The model assumes that the data is generated from a mixture of several Gaussian distributions, each with its own mean and variance (reynolds2009gaussian; mclachlan2004finite). GMMs are widely used for clustering and density estimation, as they provide a method for representing complex distributions through the combination of simpler ones.


# Results

## K-Means clustering on Sentinel-2 data
![kmeans_clustering](https://github.com/user-attachments/assets/f02e7df9-52d4-4015-b17b-f7189fb96632)

## GMM clustering on Sentinel-2 data
![GMM_clustering](https://github.com/user-attachments/assets/8cdac3bf-31b6-4c7a-9524-864898852693)

We can observe that in this case, GMM cluster does a much better job than k-means cluster at classfying sea-ice and leads for Sentinel-2 optical data.

## Mean and standard deviation of sea-ice and lead
![mean_std](https://github.com/user-attachments/assets/a850c347-4ebb-44a2-b09a-0f335d8f3e6f)
This plot describe the mean and standard deviation of sea-ice and lead

## Quantification of echoes against ESA official classification using confusion matrix



Author: Muhammad Amirul Haziq Bin Azizi (https://github.com/AmirulAzizi2225)


