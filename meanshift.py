import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2

image = Image.open('bsd5.jpg')
gr = Image.open('gr5.jpg')
image = np.array(image)

original_shape = image.shape
print(original_shape)
X = np.reshape(image, [-1, 3])

bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)
print(bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
print(ms)

labels = ms.labels_
print(labels.shape)
print(labels.size)

cluster_centers = ms.cluster_centers_

cluster_centers = np.uint8(cluster_centers)
print(cluster_centers.shape)

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

segmented_image =cluster_centers[np.reshape(labels, original_shape[:2])]
segmented_image2 =np.reshape(labels, original_shape[:2])
print(segmented_image.shape)

plt.figure(1)
ax1=plt.subplot(2, 2, 1)
ax1.set_title('original image')
plt.imshow(image)
plt.axis('off')
ax2=plt.subplot(2, 2, 2)
ax2.set_title('segmented image')
plt.imshow(segmented_image)
plt.axis('off')
ax3=plt.subplot(2, 2, 3)
ax3.set_title(' colors indicating segmentation ')
plt.imshow(segmented_image2)
plt.axis('off')
ax4=plt.subplot(2, 2, 4)
ax4.set_title('ground truth ')
plt.imshow(gr,cmap='gray')
plt.show()


