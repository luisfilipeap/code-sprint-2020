import h5py
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time
from skimage.measure import compare_ssim, compare_psnr
from imageio import imread


#f1 = h5py.File('/home/luis/Desktop/Datasets/LoDoPaB-CT/ground_truth_train_261.hdf5','r+')
#image = f1['data'][25,:,:]

image = imread('/home/luis/Desktop/Datasets/Apple CT/ground_truths_1_resized/31105_350.tif')
print(np.unique(image))
gt = np.array(image, dtype=np.float64)

print(np.max(gt))

w, h = original_shape = tuple(gt.shape)

image_array = np.reshape(gt, (w * h,1))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=5, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# Get labels for all points
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))


codebook_random = shuffle(image_array, random_state=0)[:6]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

# Display all results, alongside original image
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(gt)

result = recreate_image(kmeans.cluster_centers_, labels, w, h)
plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (5 colors, K-Means)')
plt.imshow(result)

plt.show()

result = (result-np.mean(result))/np.std(result)
gt = (result-np.mean(gt))/np.std(gt)

result = 255*(result-np.min(result))/(np.max(result)-np.min(result))
gt = 255*(result-np.min(gt))/(np.max(gt)-np.min(gt))

gt = np.array(gt, dtype=np.uint8)
result = np.array(result, dtype=np.uint8)




print(compare_psnr(gt, result))
print(compare_ssim(gt, result))
