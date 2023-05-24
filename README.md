# Image-Compression-Project

Name : Pranav Subhash Tikone
Project no. : 3
Project Name : Image Compression using ML
Topic : Unsupervised Learning
Date : 24 / 05 / 2023

This project Compresses the quality of image using the K-Means Clustering Algorithm.
K-Means clustering is an algorithm used in unsupervised learning models. Here, the machine groups/clusters the unlabelled data on its own by finding the nearest centroids. We initialize the position vectors of centroids with some initial values and then update their values by computing the closer data points and calculating the average.

The number of clusters is denoted by K, which can be initialized by us based on what output we have to compute. Here, each K value refers to a particular colour, which means that each colour will have its own centroid position. We work on these positions and group different colours into different clusters by running the K-Means algorithm. Here, I have used K = 50.

After each iteration, the distance between each centroid and data point is computed and finally to change the position of the centroid in order to get the perfect colours in particular clusters, we update the positions of each of the 50 centroids and reduce the quality of the colours in order to reduce the quality of the image; compress the image. Here, I have used 25 iterations.

The libraries used are - 
1) Numpy : To perform mathematical calculations like Norm and reshaping the arrays.
2) Matplotlib : To view the original and compressed image.

Finally, after the running the code we are able to see the original and compressed image and are alearly able to identify the differences between the 2 images.

NOTE : To convert each of the values in the range 0 to 1, we divide the original img file by 255 as the file is in JPG format. No need to do for PNG format.
