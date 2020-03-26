import numpy as np
import cv2
import matplotlib.pyplot as plt

cv = cv2

from scripts.estimate_R_and_t import estimate_R_and_t
from scripts.steps.load_data import load_image

# ATTRIBUTE = "Confidence map"
ATTRIBUTE = "Amplitude"

# DETECTOR = "ORB"
# DETECTOR = "SIFT"
DETECTOR = "SURF"
TOP_K_MATCHES = 4

list_t = []
list_R = []

for i in range(1, 100):
    print(i)

    # img1, depth_img1 = load_image("../data/RV_Data/Pitch/d1_-40/", "d1_{0:04d}.dat".format(i), ATTRIBUTE)
    # img2, depth_img2 = load_image("../data/RV_Data/Pitch/d1_-40/", "d1_{0:04d}.dat".format(i), ATTRIBUTE)
    # img2, depth_img2 = load_image("../data/RV_Data/Pitch/d2_-37/", "d2_{0:04d}.dat".format(i), ATTRIBUTE)
    # img2, depth_img2 = load_image("../data/RV_Data/Pitch/d3_-34/", "d3_{0:04d}.dat".format(i), ATTRIBUTE)
    # img2, depth_img2 = load_image("../data/RV_Data/Pitch/d4_-31/", "d4_{0:04d}.dat".format(i), ATTRIBUTE)

    img1, depth_img1 = load_image("../data/RV_Data/Translation/Y1/", "frm_{0:04d}.dat".format(i), ATTRIBUTE)
    # img2, depth_img2 = load_image("../data/RV_Data/Translation/Y1/", "frm_{0:04d}.dat".format(i), ATTRIBUTE)
    img2, depth_img2 = load_image("../data/RV_Data/Translation/Y2/", "frm_{0:04d}.dat".format(i), ATTRIBUTE)
    # img2, depth_img2 = load_image("../data/RV_Data/Translation/Y3/", "frm_{0:04d}.dat".format(i), ATTRIBUTE)
    # img2, depth_img2 = load_image("../data/RV_Data/Translation/Y4/", "frm_{0:04d}.dat".format(i), ATTRIBUTE)

    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)

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

    bf = cv2.BFMatcher(cv2.NORM_HAMMING if DETECTOR == "ORB" else cv2.NORM_L1, crossCheck=False)
    matches = bf.match(des1, des2)
    # matches = sorted(matches, key=lambda x: x.distance)
    top_matches = matches[:TOP_K_MATCHES]  # ???
    # img3 = cv2.drawMatches(img1, kp1, img2, kp2, top_matches, None, flags=2)
    # plt.imshow(img3)
    # plt.show()

    points1 = np.array([list(kp1[m.queryIdx].pt) for m in top_matches])
    points2 = np.array([list(kp2[m.trainIdx].pt) for m in top_matches])

    point1_3d = np.array([list(depth_img2[int(kp1[m.queryIdx].pt[1])][int(kp1[m.queryIdx].pt[0])]) for m in top_matches]).T
    point2_3d = np.array([list(depth_img1[int(kp2[m.trainIdx].pt[1])][int(kp2[m.trainIdx].pt[0])]) for m in top_matches]).T

    p = point1_3d
    p_prime = point2_3d

    R, t = estimate_R_and_t(p, p_prime)


    def plot_3d(q, q_prime, title):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(q[0, :], q[1, :], q[2, :], c='k')
        ax.scatter(q_prime[0, :], q_prime[1, :], q_prime[2, :], c='r', marker='x')

        for i in range(q.shape[1]):
            plt.plot([q[0, i], q_prime[0, i]], [q[1, i], q_prime[1, i]], [q[2, i], q_prime[2, i]], 'k--')

        plt.title(title if title else "None")
        plt.show()


    # plot_3d(p, p_prime, "p and p_prime")
    # plot_3d(p_prime, R.dot(p) + t, "p_prime and Rp + T")

    list_R.append(R)
    list_t.append(t)

XYZ_mean = np.array(list_t).mean(axis=0)
XYZ_std = np.array(list_t).std(axis=0)

print("X: ({},{})".format(XYZ_mean[0][0], XYZ_std[0][0]))
print("Y: ({},{})".format(XYZ_mean[2][0], XYZ_std[2][0]))
print("Z: ({},{})".format(XYZ_mean[1][0], XYZ_std[1][0]))
