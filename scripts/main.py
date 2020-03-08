import math

import numpy as np
import cv2
import matplotlib.pyplot as plt

# ATTRIBUTE = "Calibrated Distance"
ATTRIBUTE = "Confidence map"
# ATTRIBUTE = "Amplitude"

DETECTOR = "ORB"
# DETECTOR = "SIFT"
# DETECTOR = "SURF"
TOP_K_MATCHES = 300
MAX_HOMOGRAPHY_ITERATIONS = 250

img1 = cv2.imread('../data/output/RV_Data/Pitch/' + ATTRIBUTE + '/0001/-31.png', 0)  # queryImage
img2 = cv2.imread('../data/output/RV_Data/Pitch/' + ATTRIBUTE + '/0001/-34.png', 0)  # trainImage
# img2 = cv2.imread('../data/output/RV_Data/Pitch/' + ATTRIBUTE + '/0001/-37.png', 0)  # trainImage
# img2 = cv2.imread('../data/output/RV_Data/Pitch/' + ATTRIBUTE + '/0001/-40.png', 0)  # trainImage
# img2 = cv2.imread('../data/output/RV_Data/Pitch/' + ATTRIBUTE + '/0002/-31.png', 0)  # trainImage


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
bf = cv2.BFMatcher(cv2.NORM_HAMMING if DETECTOR == "ORB" else cv2.NORM_L1, crossCheck=(DETECTOR == "ORB"))

# Match descriptors.
matches = bf.match(des1, des2)
# Sort them in the order of their distance.
print(matches)
matches = sorted(matches, key=lambda x: x.distance)
img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, None, flags=2)
plt.imshow(img3)
plt.show()


plt.plot([x.distance for x in matches])
plt.show()

print(kp1)
print(len(kp1))
print(len(kp2))
print("====")

top_matches = matches[:TOP_K_MATCHES]  # ???


points1 = np.array([list(kp2[m.trainIdx].pt) for m in top_matches])
points2 = np.array([list(kp1[m.queryIdx].pt) for m in top_matches])

print("points2", points2)

cameraMatrix = np.float64([[1,0,0],
                           [0,1,0],
                           [0,0,1]])

E, _ = cv2.findEssentialMat(points1, points2, cameraMatrix=cameraMatrix, method=cv2.RANSAC)

_, R, t, _ = cv2.recoverPose(E, points1, points2, cameraMatrix)
print("R", R)
print("t", t)


# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    sin_x = math.sqrt(R[2, 0] * R[2, 0] + R[2, 1] * R[2, 1])
    if not singular:
        z1    = math.atan2(R[2,0], R[2,1])     # around z1-axis
        x      = math.atan2(sin_x,  R[2,2])     # around x-axis
        z2    = math.atan2(R[0,2], -R[1,2])    # around z2-axis
    else: # gimbal lock
        z1    = 0                                         # around z1-axis
        x      = math.atan2(sin_x,  R[2,2])     # around x-axis
        z2    = 0                                         # around z2-axis

    return [z1, x, z2]

    # if not singular:
    #     x = math.atan2(R[2, 1], R[2, 2])
    #     y = math.atan2(-R[2, 0], sy)
    #     z = math.atan2(R[1, 0], R[0, 0])
    # else:
    #     x = math.atan2(-R[1, 2], R[1, 1])
    #     y = math.atan2(-R[2, 0], sy)
    #     z = 0
    #
    # return np.array([x, y, z])


print(rotationMatrixToEulerAngles(R))

h, _ = cv2.findHomography(points1, points2, cv2.RANSAC, maxIters=MAX_HOMOGRAPHY_ITERATIONS)
height, width = img1.shape
im1Reg = cv2.warpPerspective(img2, h, (width, height))

plt.imshow(img1)
plt.show()

plt.imshow(img2)
plt.show()

plt.imshow(im1Reg)
plt.show()

# print("h", mask_h.shape)
# print("E", mask_E.shape)
#
# print("h", h)
# print("E", E)


# E, mask_E = cv2.findEssentialMat(points1, points2, cameraMatrix=cameraMatrix, method=cv2.RANSAC, prob=0.9, threshold=1.0)

# plt.plot(h, 'r-')
# ax = plt.twinx()
# plt.plot(E, 'k:')
# plt.show()
#
# plt.plot(mask_h, 'r-')
# ax = plt.twinx()
# plt.plot(mask_E, 'k:')
# plt.show()

