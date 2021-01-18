# Implementation of Clustering Algorithms to offer Business Insights.
## Motive- Customer Segmentation

### ABSTRACT
The project aims to **implement clustering algorithms to offer business insights**.
Clustering is the task of dividing the population or data points into a number of
groups such that data points in the same groups are more similar to other data points
in the same group than those in other groups. In simple words, the aim is to segregate
groups with similar traits and assign them into clusters. Today many businesses
generate a lot of data. Clustering can help businesses to manage their data better –
Customer segmentation, grouping web pages, market segmentation and information
retrieval are four examples. For retail businesses, data clustering helps with customer
shopping behavior, sales campaigns and customer retention. In the insurance
industry, clustering is regularly employed in fraud detection, risk factor
identification and customer retention efforts. In banking, clustering is used for
customer segmentation, credit scoring and analyzing customer profitability.

### In this project three different clustering algorithms are used.
1. DBSCAN Clustering
1. Hierarchical Clustering
1. K-means Clustering

### DBSCAN
Density Based Spacial Clustering of Applications with noise.
We are going to use the DBSCAN for algorithm for the purpose of clustering. 
It is an unsupervised machine learning algorithm.
It is used for clusters of high density. It automatically predicts the outliers and removes it. 
It is better than hierarchical and k-means clustering algorithm. 
It makes the clusters based on the parameters like epsilon,min points and noise.
It separately predicts the core points, border points and outliers efficiently.

### Advantages
* Is great at separating clusters of high density versus clusters of low density within a given dataset.
* Is great with handling outliers within the dataset.

## Disadvantages
* While DBSCAN is great at separating high density clusters from low density clusters, DBSCAN struggles with clusters of similar density.
* Struggles with high dimensionality data. I know, this entire article I have stated how DBSCAN is great at contorting the data into different dimensions and shapes. However, DBSCAN can only go so far, if given data with too many dimensions, DBSCAN suffers

### Hierarchical Clustering
Agglomerative Hierarchical clustering -This algorithm  works by  grouping  the data one by one on the basis of the  nearest distance measure of all the pairwise distance between the data point. Again distance between the data point is recalculated but which distance to consider when the groups has been formed? For this there are many available methods. Some of them are:
1. single-nearest distance or single linkage.
1. complete-farthest distance or complete linkage.
1. average-average distance or average linkage.
1. centroid distance.
1. ward's method - sum of squared euclidean distance is minimized.
This way we go on grouping the data until one cluster is formed. Now on the basis of dendogram graph we can calculate how many number of clusters should be actually present.

### Advantages
* First, we do not need to specify the number of clusters required for the algorithm.
* Second, hierarchical clustering is easy to implement.
* And third, the dendrogram produced is very useful in understanding the data.


### Disadvantages
*First, the algorithm can never undo any previous steps. So for example, the algorithm clusters 2 points, and later on we see that the connection was not a good one, the program cannot undo that step.
*Second, the time complexity for the clustering can result in very long computation times, in comparison with efficient algorithms, such k-Means.
*Finally, if we have a large dataset, it can become difficult to determine the correct number of clusters by the dendrogram.


### K-means Clustering
K-Means clustering algorithm is defined as a unsupervised learning methods having an iterative process
in which the dataset are grouped into k number of predefined non-overlapping clusters or subgroups making
the inner points of the cluster as similar as possible while trying to keep the clusters at distinct space
it allocates the data points to a cluster so that the sum of the squared distance between the clusters centroid
and the data point is at a minimum, at this position the centroid of the cluster is the arithmetic mean 
of the data points that are in the clusters.
This algorithm is an iterative algorithm that partitions the dataset according to their features into K number of predefined non- overlapping distinct clusters or subgroups. 
It makes the data points of inter clusters as similar as possible and also tries to keep the clusters as far as possible. 
It allocates the data points to a cluster if the sum of the squared distance between the cluster’s centroid and the data points is at a minimum where the cluster’s centroid is the arithmetic mean of the data points that are in the cluster.
A less variation in the cluster results in similar or homogeneous data points within the cluster.

### Advantages
* It is fast,Robust,Flexible,Easy to understand & Comparatively efficient
* If data sets are distinct then gives the best results
* Produce tighter clusters
* When centroids are recomputed the cluster changes.
* Easy to interpret
* Better computational cost
* Enhances Accuracy
* Works better with spherical clusters

### Disadvantages
* Needs prior specification for the number of cluster centers
* If there are two highly overlapping data then it cannot be distinguished and cannot tell that there are two clusters
* With the different representation of the data, the results achieved are also different
* Euclidean distance can unequally weight the factors
* It gives the local optima of the squared error function
* Sometimes choosing the centroids randomly cannot give fruitful results

## Steps to run the code
1. Download and install Anaconda Navigator.
1. Clone and extract contents of this repository.
1. Open Jupyter Notebook
1. Navigate to the directory where you have extracted the contents of this repository.
1. Open **CustomerSegmentation.ipynb** file.
1. Analyse the data.Feel free to make any changes in the code as per your preference.
GOOD LUCK!!
