import math
import cv2
import numpy as np
from numpy.linalg import norm
from random import seed
from random import randint

filename = '1.jpg'

# --- Part 1.1
img = cv2.imread(filename)
imgSize = img.shape
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 0, 0.01, 10)
corners = np.int0(corners)
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img, (x, y), 3, 255, -1)

cv2.imshow('All Corners', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()

# --- Part 1.2
img = cv2.imread(filename)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

corners = cv2.goodFeaturesToTrack(gray, 300, 0.01, 10)
corners = np.int0(corners)
dataPoints = []
for corner in corners:
    x, y = corner.ravel()
    dataPoints.append([x, y])
    cv2.circle(img, (x, y), 3, 255, -1)

cv2.imshow('Top 300 Corners', img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()


# --- part2:


class k_means:

    def __init__(self, K, iterationLimit=30):
        self.K = K
        self.iterationLimit = iterationLimit
        self.labels = None
        self.cnts = None
        self.error = None

    def cntInit(self, X):
        np.random.RandomState(1)
        cnts = []
        for i in range(0, self.K):
            index = randint(0, len(X) - 1)
            # print(X[index])
            cnts.append(X[index])
        return cnts

    def compute_cnts(self, X, labels):
        cnts = np.zeros([self.K, 2], dtype=int)

        for k in range(self.K):
            sx = 0
            sy = 0
            count = 0
            for i in range(len(X)):
                if labels[i] == k:
                    sx += X[i][0]
                    sy += X[i][1]
                    count += 1
            if count != 0:
                cnts[k][0] = sx / count
                cnts[k][1] = sy / count
        return cnts

    def compute_distance(self, X, cnts):
        # print(len(X), self.K)
        distance = np.zeros([len(X), self.K], dtype=int)
        for k in range(self.K):
            for i in range(len(X)):
                distance[i][k] = math.sqrt((X[i][0] - cnts[k][0]) ** 2 + (X[i][1] - cnts[k][1]) ** 2)
                # print((X[i][0] - cnts[k][0]), (X[i][1] - cnts[k][1]), distance[i][k])

        return distance

    def find_closest_cluster(self, distance):
        return np.argmin(distance, axis=1)

    def fit(self, X):
        self.cnts = self.cntInit(X)
        for i in range(self.iterationLimit):
            old_cnts = self.cnts
            distance = self.compute_distance(X, old_cnts)
            self.labels = self.find_closest_cluster(distance)
            self.cnts = self.compute_cnts(X, self.labels)
            if np.all(old_cnts == self.cnts):
                break
        self.error = self.compute_sse(X, self.labels, self.cnts)

    def predict(self, X, old_cnts=None):
        distance = self.compute_distance(X, old_cnts)
        return self.find_closest_cluster(distance)

    def compute_sse(self, X, labels, cnts):
        s = 0
        for k in range(self.K):
            for i in range(len(X)):
                if labels[i] == k:
                    s += (X[i][0] - cnts[k][0]) ** 2 + (X[i][1] - cnts[k][1]) ** 2
        return s


# --- use k_means class
for itr in range(1, 8):
    km = k_means(itr, 50)
    km.fit(dataPoints)
    centroids = km.cnts
    print("Number of cluster= ", itr, ", SSE = ", km.error)
    color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0],
             [255, 255, 255],
             [100, 100, 100], [150, 150, 150]]
    for k in range(km.K):

        for i in range(len(dataPoints)):
            if km.labels[i] == k:
                cv2.circle(img, (dataPoints[i][0], dataPoints[i][1]), 3, color[k], -1)
    title = '#clusters=' + str(itr) + ", SSE= " + str(km.error)
    cv2.imshow(title, img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()

# --- part 3

km = k_means(3, 50)
km.fit(dataPoints)
centroids = km.cnts
print("Number of cluster= ", 3, ", SSE = ", km.error)
color = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [0, 0, 0], [255, 255, 255],
         [100, 100, 100], [150, 150, 150]]
for k in range(km.K):
    x1 = y1 = 100000
    x2 = y2 = 0

    for i in range(len(dataPoints)):
        if km.labels[i] == k:
            cv2.circle(img, (dataPoints[i][0], dataPoints[i][1]), 3, color[k], -1)
            x1 = min(x1, dataPoints[i][0])
            y1 = min(y1, dataPoints[i][1])
            x2 = max(x2, dataPoints[i][0])
            y2 = max(y2, dataPoints[i][1])

    cv2.rectangle(img, (x1, y1), (x2, y2), color[k], 3)
title = '#clusters=' + str(3) + ", SSE= " + str(km.error)
cv2.imshow(title, img)
if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
