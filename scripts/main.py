import math

import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# ATTRIBUTE = "Calibrated Distance"
from scripts.steps.load_data import load_image

ATTRIBUTE = "Confidence map"
# ATTRIBUTE = "Amplitude"

# DETECTOR = "ORB"
DETECTOR = "SIFT"
# DETECTOR = "SURF"
TOP_K_MATCHES = 300
MAX_HOMOGRAPHY_ITERATIONS = 250

list_t = []
list_R = []

for i in range(1,10):
    print(i)

    img1, depth_img1 = load_image("../data/RV_Data/Pitch/d1_-40/", "d1_{0:04d}.dat".format(i), ATTRIBUTE)
    # img2, depth_img2 = load_image("../data/RV_Data/Pitch/d1_-40/", "d1_{0:04d}.dat".format(i), ATTRIBUTE)
    img2, depth_img2 = load_image("../data/RV_Data/Pitch/d3_-34/", "d3_{0:04d}.dat".format(i), ATTRIBUTE)

    if DETECTOR == "ORB":
        detector = cv2.ORB_create()
    elif DETECTOR == "SIFT":
        detector = cv2.xfeatures2d.SIFT_create()
    elif DETECTOR == "SURF":
        detector = cv2.xfeatures2d.SURF_create()
    else:
        print("Detector ({}) unknown".format(DETECTOR))
        exit(-1)

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING if DETECTOR == "ORB" else cv2.NORM_L1, crossCheck=False)

    # Match descriptors.
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
    # plt.imshow(img3)
    # plt.show()
    #
    # plt.plot([x.distance for x in matches])
    # plt.title("Distances for matched descriptors (low to high)")
    # plt.show()

    # print("Total matches", len(matches))
    top_matches = matches[:TOP_K_MATCHES]  # ???

    points1 = np.array([list(kp2[m.trainIdx].pt) for m in top_matches])
    points2 = np.array([list(kp1[m.queryIdx].pt) for m in top_matches])

    # Figure out transformation matrix
    height, width = img1.shape
    h, _ = cv2.findHomography(points1, points2, cv2.RANSAC, maxIters=MAX_HOMOGRAPHY_ITERATIONS)
    im1Reg = cv2.warpPerspective(img2, h, (width, height))
    h, _ = cv2.findHomography(points2, points1, cv2.RANSAC, maxIters=MAX_HOMOGRAPHY_ITERATIONS)
    im2Reg = cv2.warpPerspective(img1, h, (width, height))
    # plt.imshow(img1)
    # plt.show()
    # plt.imshow(img2)
    # plt.show()
    # plt.imshow(im1Reg)
    # plt.show()
    # plt.imshow(im2Reg)
    # plt.show()

    # Figure out pose change
    cameraMatrix = np.float64([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    E, _ = cv2.findEssentialMat(points1, points2, cameraMatrix, cv2.RANSAC, 0.999, 1.0, mask=None)
    # print(E)
    newE, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix)
    # print("R", R)
    # print("t", t)

    # print(E, newE)

    list_t.append(t)
    list_R.append(R)


print(list_t)
plt.plot(sorted(np.array(list_t)[:,0,0]))
plt.plot(sorted(np.array(list_t)[:,1,0]))
plt.plot(sorted(np.array(list_t)[:,2,0]))
plt.legend(["x", "y", "z"])
plt.title("predicted transformation")
plt.show()

print(list_R)
plt.plot(sorted(np.array(list_R)[:,0,0]))
plt.plot(sorted(np.array(list_R)[:,0,1]))
plt.plot(sorted(np.array(list_R)[:,0,2]))
plt.plot(sorted(np.array(list_R)[:,1,0]))
plt.plot(sorted(np.array(list_R)[:,1,1]))
plt.plot(sorted(np.array(list_R)[:,2,2]))
plt.plot(sorted(np.array(list_R)[:,2,0]))
plt.plot(sorted(np.array(list_R)[:,1,1]))
plt.plot(sorted(np.array(list_R)[:,2,2]))
plt.legend([
    "0,0", "0,1", "0,2",
    "1,0", "1,1", "1,2",
    "2,0", "2,1", "2,2",
])
plt.title("predicted R")
plt.show()

# plt.plot(R)
