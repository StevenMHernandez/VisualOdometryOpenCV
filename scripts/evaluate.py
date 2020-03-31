import numpy as np
import cv2
import matplotlib.pyplot as plt

from scripts.steps.rotation_to_euler import rotation_to_euler

cv = cv2

from scripts.steps.estimate_R_and_t import estimate_R_and_t
from scripts.steps.load_data import load_image

ATTRIBUTE = "Amplitude"


def evaluate(base_data_path, movement_data_path, settings):
    list_R = []
    list_t = []

    for i in range(1, 100):
        print(i)
        img1, depth_img1 = load_image(base_data_path.format(i), ATTRIBUTE)
        img2, depth_img2 = load_image(movement_data_path.format(i), ATTRIBUTE)

        if settings["MEDIAN_BLUR"]:
            img1 = cv2.medianBlur(img1, 5)
            img2 = cv2.medianBlur(img2, 5)
        if settings["GAUSSIAN_BLUR"]:
            img1 = cv2.GaussianBlur(img1, (5, 5), 0)
            img2 = cv2.GaussianBlur(img2, (5, 5), 0)

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

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        top_matches = []
        for i, (m, n) in enumerate(matches):
            if not settings["KNN_MATCHING"] or m.distance < settings["FEATURE_FILTER_RATIO"] * n.distance:
                top_matches.append(m)

        top_matches = top_matches[:3]


        # img3 = cv2.drawMatches(img1, kp1, img2, kp2, top_matches, None, flags=2)
        # plt.figure(figsize=(10,5))
        # plt.imshow(img3)
        # plt.savefig("feature_match_" + settings["DETECTOR"] + "." + str(settings["KNN_MATCHING"]) + ".png")
        # exit()

        point1_3d = np.array(
            [list(depth_img2[int(kp1[m.queryIdx].pt[1])][int(kp1[m.queryIdx].pt[0])]) for m in top_matches]).T
        point2_3d = np.array(
            [list(depth_img1[int(kp2[m.trainIdx].pt[1])][int(kp2[m.trainIdx].pt[0])]) for m in top_matches]).T

        p = point1_3d
        p_prime = point2_3d

        R, t = estimate_R_and_t(p, p_prime)

        def plot_3d(q, q_prime, title):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(q[0, :], q[2, :], q[1, :], c='k', label='p\'')
            ax.scatter(q_prime[0, :], q_prime[2, :], q_prime[1, :], c='r', marker='x', label='Rp + t')

            for i in range(q.shape[1]):
                plt.plot([q[0, i], q_prime[0, i]], [q[2, i], q_prime[2, i]], [q[1, i], q_prime[1, i]], 'k--')

            ax.legend()

            # plt.title(title if title else "None")
            # plt.show()
            plt.savefig("3d_points_" + title + ".png")

        # plot_3d(p, p_prime, "original")
        # plot_3d(p_prime, R.dot(p) + t, "minimized")


        list_R.append(rotation_to_euler(R))
        list_t.append(t)

    RPY_mean = np.array(list_R).mean(axis=0)
    RPY_std = np.array(list_R).std(axis=0)

    XYZ_mean = np.array(list_t).mean(axis=0)
    XYZ_std = np.array(list_t).std(axis=0)

    return [
        XYZ_mean[0][0], XYZ_std[0][0],  # X
        XYZ_mean[2][0], XYZ_std[2][0],  # Y
        XYZ_mean[1][0], XYZ_std[1][0],  # Z
        RPY_mean[0], RPY_std[0],  # φ
        RPY_mean[2], RPY_std[2],  # θ
        RPY_mean[1], RPY_std[1],  # ψ
    ]
