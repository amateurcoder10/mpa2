import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

#np.set_printoptions(threshold=np.inf)
img = cv2.imread('bsd5.jpg')
gr = cv2.imread('gr5.jpg')
Z1 = img.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z1)

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image

print(center)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))


center=np.array([[0,0,255],[0,255,0],[255,0,0],[0,255,255],[255,255,255],[0,0,0],[255,255,0],[127,127,127]],np.uint8)
res = center[label.flatten()]
resnew = res.reshape((img.shape))


plt.figure(1)
ax1=plt.subplot(2, 2, 1)
ax1.set_title('original image')
plt.imshow(img)
plt.axis('off')
ax2=plt.subplot(2, 2, 2)
ax2.set_title('segmented image')
plt.imshow(res2)
plt.axis('off')
ax3=plt.subplot(2, 2, 3)
ax3.set_title(' colors indicating segmentation ')
plt.imshow(resnew)
plt.axis('off')
ax4=plt.subplot(2, 2, 4)
ax4.set_title('ground truth ')
plt.imshow(gr,cmap='gray')
plt.show()

