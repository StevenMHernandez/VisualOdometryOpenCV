import time

import numpy as np
import cv2
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler

from scripts.steps.ransac import ransac

cv = cv2

from scripts.steps.estimate_R_and_t import estimate_R_and_t
from scripts.steps.load_data import load_image

ATTRIBUTE = "Amplitude"


def evaluate(base_data_path, movement_data_path, settings):
    list_R = []
    list_t = []

    print(base_data_path)
    print(movement_data_path)

    for image_i in range(1, 100):
        print(image_i)

        #
        # Load Amplitude image and 3D point cloud ✓
        #
        img1, depth_img1 = load_image(base_data_path.format(image_i), ATTRIBUTE, settings["MEDIAN_BLUR"])
        img2, depth_img2 = load_image(movement_data_path.format(image_i), ATTRIBUTE, settings["MEDIAN_BLUR"])

        #
        # Apply Image Blurring ✓
        #
        if settings["GAUSSIAN_BLUR"]:
            img1 = cv2.GaussianBlur(img1, (5, 5), 0)
            img2 = cv2.GaussianBlur(img2, (5, 5), 0)
            depth_img1 = cv2.GaussianBlur(depth_img1, (5, 5), 0)
            depth_img2 = cv2.GaussianBlur(depth_img2, (5, 5), 0)

        #
        # Feature Selection ✓
        #
        detector = None
        if settings['DETECTOR'] == "ORB":
            detector = cv2.ORB_create()
        elif settings['DETECTOR'] == "SIFT":
            detector = cv2.xfeatures2d.SIFT_create()
        elif settings['DETECTOR'] == "SURF":
            detector = cv2.xfeatures2d.SURF_create()
        else:
            print("Detector ({}) unknown".format(settings['DETECTOR']))
            exit(-1)

        kp1, des1 = detector.detectAndCompute(img1, None)
        kp2, des2 = detector.detectAndCompute(img2, None)

        #
        # Feature Matching ✓
        #
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        top_matches = []
        for i, (m, n) in enumerate(matches):
            if not settings["KNN_MATCHING_RATIO"] or m.distance < settings["KNN_MATCHING_RATIO"] * n.distance:
                top_matches.append(m)

        #
        # Convert matches to 3d point lists ✓
        #
        p = np.array([list(depth_img1[int(kp1[m.queryIdx].pt[1])][int(kp1[m.queryIdx].pt[0])]) for m in top_matches]).T
        p_prime = np.array(
            [list(depth_img2[int(kp2[m.trainIdx].pt[1])][int(kp2[m.trainIdx].pt[0])]) for m in top_matches]).T

        #
        # Select R and t with RANSAC (or not RANSAC) ✓
        #
        if settings['RANSAC_THRESHOLD'] > 0:
            R, t, inliers = ransac(p, p_prime, settings['RANSAC_THRESHOLD'])
        else:
            inliers = list(range(p.shape[1]))  # all matches are considered inliers
            R, t = estimate_R_and_t(p, p_prime)

        #
        # Add results for statistics ✓
        #
        list_R.append(np.degrees(mat2euler(R)))
        list_t.append(t)

        #
        # Plots
        #
        # plot_matches(img1, kp1, img2, kp2, top_matches, inliers, base_data_path, image_i)
        # plot_3d(p_prime, R.dot(p) + t, "Rp + t", base_data_path, image_i)

    RPY_mean = np.array(list_R).mean(axis=0)
    RPY_std = np.array(list_R).std(axis=0)

    XYZ_scaling = 1000 / 3

    XYZ_mean = (np.array(list_t) * XYZ_scaling).mean(axis=0)
    XYZ_std = (np.array(list_t) * XYZ_scaling).std(axis=0)

    plot_cdf((np.array(list_R)), "Rotation")
    plot_cdf((np.array([x[:, 0] for x in list_t]) * XYZ_scaling), "Translation")

    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plot_means(np.array(list_R), RPY_mean, RPY_std)
    plt.subplot(1, 2, 2)
    plot_means(np.array([x[:, 0] for x in list_t]) * XYZ_scaling, RPY_mean, RPY_std)
    plt.suptitle("{}\n{}".format(base_data_path, movement_data_path))
    plt.show()

    return [
        XYZ_mean[0][0], XYZ_std[0][0],  # X
        XYZ_mean[2][0], XYZ_std[2][0],  # Y
        XYZ_mean[1][0], XYZ_std[1][0],  # Z
        RPY_mean[2], RPY_std[2],  # φ
        RPY_mean[0], RPY_std[0],  # θ
        RPY_mean[1], RPY_std[1],  # ψ
    ]


def plot_means(_list, _mean, _std):
    plt.plot(sorted(_list[:, 0]), '.-')
    plt.plot(sorted(_list[:, 1]), '.-')
    plt.plot(sorted(_list[:, 2]), '.-')
    plt.legend(["x", "y", "z"])
    plt.title("\n\nRotation\n{}\n{}".format(_mean, _std))


def plot_cdf(X, title):
    for i in range(3):
        _X = X[:, i]
        _step_size = 0.1
        _range = np.arange(_X.min() - _step_size, _X.max() + _step_size, _step_size)
        _cdf_sum = np.zeros(_range.shape)
        for i, v in enumerate(_range):
            _cdf_sum[i] = (_X < v).sum()

        plt.plot(_range, _cdf_sum / len(_X))
    plt.legend(["X", "Y", "Z"])
    plt.xlabel("Prediction")
    plt.ylabel("CDF")
    plt.ylim([0, 1])
    plt.title(title)
    plt.show()


def plot_matches(img1, kp1, img2, kp2, top_matches, inliers, base_data_path, image_i):
    plt.figure(figsize=(10, 5))
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, np.array(top_matches)[inliers].tolist(), None, flags=2)
    plt.imshow(img3)
    ts = time.time()
    plt.savefig("../output/feature_match/" + base_data_path.split("/")[3] + "." + str(image_i) + ".png")
    plt.close()
    # plt.show()


def plot_3d(q, q_prime, title, base_data_path, image_i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(q[0, :], q[2, :], q[1, :], c='k', label='p\'')
    ax.scatter(q_prime[0, :], q_prime[2, :], q_prime[1, :], c='r', marker='x', label='Rp + t')

    for i in range(q.shape[1]):
        plt.plot([q[0, i], q_prime[0, i]], [q[2, i], q_prime[2, i]], [q[1, i], q_prime[1, i]], 'k--')

    ax.legend()

    plt.title(title if title else "None")
    # plt.show()
    plt.savefig("../output/3d_points/" + base_data_path.split("/")[3] + "." + str(image_i) + ".png")
    plt.close()
