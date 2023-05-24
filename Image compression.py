import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

original_img = plt.imread(r"C:\Users\ASUS\Desktop\Pranav\Machine Learning\Datasets\ronaldo.jpg")
#plt.imshow(original_img)

#print(f"Shape of original img is {original_img.shape}")

original_img = original_img/255

x_img = np.reshape(original_img, (original_img.shape[0]*original_img.shape[1],3))

#print(f"Shape of reshaped img is {x_img.shape}")

K = 50
max_iterations = 25

def find_closest_centroids(X, centroids):
    
     K = centroids.shape[0]
     idx = np.zeros(X.shape[0],dtype=(int))
    
     for i in range (X.shape[0]):
        d = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i]-centroids[j])
            d.append(norm_ij)
            
        idx[i] = np.argmin(d)
        
     return idx

def compute_centroids(X, idx, K):
    
    m,n = X.shape
    centroids = np.zeros((K,n))
    
    for i in range(K):
        
        points = X[idx==i]
        centroids[i] = np.mean(points, axis=0)
        
    return centroids
    
def k_means_init_centroids(x,K):
    
    randidx = np.random.permutation(x.shape[0])
    centroids = x[randidx[:K]]
    
    return centroids

initial_centroids = k_means_init_centroids(x_img, K)

def run_kmeans(X, initial_centroids, max_iters, plot_progress = False):
    
    m,n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)
    
    for i in range(max_iters):
        print(f"K-Means iteration {i}/{max_iters-1}")
        idx = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
        
    return centroids, idx

centroids, idx = run_kmeans(x_img, initial_centroids, max_iterations)

#print(f"Shape of idx : {idx.shape}")

idx = find_closest_centroids(x_img, centroids)

x_recovered = centroids[idx,:]
x_recovered = np.reshape(x_recovered, original_img.shape)

fig, ax = plt.subplots(1,2,figsize=(50,50))
plt.axis('Off')

ax[0].imshow(original_img)
ax[0].set_title('Original Image')
ax[0].set_axis_off()

ax[1].imshow(x_recovered)
ax[1].set_title('Compressed Image')
ax[1].set_axis_off()

