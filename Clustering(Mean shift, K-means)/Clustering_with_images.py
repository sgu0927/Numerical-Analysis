import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

img_path = 'test3'
files = [ f for f in listdir(img_path) if isfile(join(img_path,f)) ]
for i in range(len(files)):
    img = cv2.imread(join(img_path,files[i]),1)
    dst = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    print(np.shape(dst))
    flat_image = np.reshape(dst, [-1, 3])
    print(np.shape(flat_image))

    ######## Mean Shift Clustering ############

    # The following bandwidth can be automatically detected using
    bandwidth = estimate_bandwidth(flat_image, quantile=0.03, n_samples=1000)
    ms = MeanShift(bandwidth = bandwidth,bin_seeding=True)
    ms.fit(flat_image)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    center = np.uint8(cluster_centers)
    res = center[labels.flatten()]
    res2 = res.reshape((img.shape))
    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    print("number of estimated clusters : %d" % n_clusters_)

    res2 = cv2.cvtColor(res2,cv2.COLOR_LAB2BGR)
    cv2.imshow('res1',res2)
    cv2.waitKey(0) 

    ######## K- means Clustering ############

    flat_image = np.reshape(dst, [-1, 3])
    flat_image = np.float32(flat_image)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n_clusters_
    ret,label,center=cv2.kmeans(flat_image,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    res2 = cv2.cvtColor(res2,cv2.COLOR_LAB2BGR)
    cv2.imshow('res2',res2)
    cv2.waitKey(0)


