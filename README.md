## Echo Classification: Leads vs. Sea Ice

## Overview
This project aims to classify radar echoes into two categories: Leads and Sea Ice, using unsupervised learning methods. The classification is performed on Sentinel-2 optical data and Sentinel-3 altimetry data, and the results are compared against the ESA official classification using a confusion matrix.

Objectives
- Classify echoes into Leads and Sea Ice using clustering techniques.
- Compute average echo shapes and standard deviations for each class.
- Validate classification accuracy by comparing with ESA’s official classification using a confusion matrix.

Dataset
- Satellite Data: Sentinel-2 optical data & Sentinel-3 altimetry data.
- Preprocessing: Data is preprocessed to extract relevant features for clustering and classification.

Methodology
1. Data Preprocessing
   - Load and clean the satellite data.
   - Extract echo features relevant for classification.
   
2. Unsupervised Classification (K-means Clustering)
   - Apply K-means clustering to distinguish between Leads and Sea Ice.
   - Visualize and analyze cluster separation.

3. Computation of Echo Shapes
   - Calculate average echo shape and standard deviation for each class.
   
4. Validation with ESA Classification
   - Compare our results with ESA’s official classification using a confusion matrix.
   - Compute performance metrics (accuracy, precision, recall, F1-score).

Results
- Confusion matrix showing classification performance.
- Plots of average echo shapes for Leads and Sea Ice.
- Standard deviation analysis** to evaluate variations in echoes.

Contributors
- Muhammad Amirul Haziq Bin Azizi (https://github.com/AmirulAzizi2225)


