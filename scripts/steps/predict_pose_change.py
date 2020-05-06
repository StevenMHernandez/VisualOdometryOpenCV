import cv2
from cachetools import cached
from cachetools.keys import hashkey
from pandas import np
from transforms3d.euler import mat2euler
from transforms3d.quaternions import mat2quat
import matplotlib.pyplot as plt

from scripts.steps.estimate_R_and_t import estimate_R_and_t
from scripts.steps.load_data import load_image
from scripts.steps.ransac import ransac

ATTRIBUTE = "Amplitude"

detector = cv2.xfeatures2d.SIFT_create()

@cached(cache={}, key=lambda image_name, img_x: hashkey(image_name))
def cachedDetectAndCompute(image_name, img_x):
    return detector.detectAndCompute(img_x, None)

def predict_pose_change(data_1, data_2, settings, real_change, CALCULATE_ERROR, print_image=False):
    #
    # Load Amplitude image and 3D point cloud ✓
    #
    img1, depth_img1 = load_image(data_1, ATTRIBUTE, settings["MEDIAN_BLUR"])
    img2, depth_img2 = load_image(data_2, ATTRIBUTE, settings["MEDIAN_BLUR"])

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
    kp1, des1 = cachedDetectAndCompute(data_1, img1)
    kp2, des2 = cachedDetectAndCompute(data_2, img2)

    #
    # Feature Matching ✓
    #
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    top_matches = []
    _knn_ratio = settings["KNN_MATCHING_RATIO"]
    while (len(top_matches) < 3):
        for i, (m, n) in enumerate(matches):
            if not _knn_ratio or m.distance < _knn_ratio * n.distance:
                top_matches.append(m)
        print("_knn_ratio", _knn_ratio, "#", len(matches), "#", len(top_matches))
        _knn_ratio += 0.1

    #
    # Convert matches to 3d point lists ✓
    #
    p = np.array([list(depth_img1[int(kp1[m.queryIdx].pt[1])][int(kp1[m.queryIdx].pt[0])]) for m in top_matches]).T
    p_prime = np.array([list(depth_img2[int(kp2[m.trainIdx].pt[1])][int(kp2[m.trainIdx].pt[0])]) for m in top_matches]).T

    #
    # Select R and t with RANSAC (or not RANSAC) ✓
    #
    R, t, inliers, informationMatrix = ransac(p, p_prime, settings['RANSAC_THRESHOLD'])

    if len([x for x in inliers if x]) < 2:
        print("top_matches", len(top_matches), len(matches))
        raise Exception("Not enough inliers !")

    #
    # Add results for statistics ✓
    #
    XYZ_scaling = 1000  # meters to millimeters
    final_r = np.degrees(mat2euler(R, "ryxz"))
    final_r_q = np.degrees(mat2quat(R))
    final_t = [x[0] * XYZ_scaling for x in t]


    if CALCULATE_ERROR:
        print('pitch', final_r[1], real_change['pitch'], final_r[1] - real_change['pitch'])
        final_t[0] += real_change['x']
        final_t[1] += real_change['y']
        final_t[2] += real_change['z']
        final_r[0] += real_change['roll']
        final_r[1] += real_change['pitch']
        final_r[2] += real_change['yaw']

    #
    # Plots
    #
    if print_image:
        plot_matches(img1, kp1, img2, kp2, top_matches, inliers, "./", data_1.split("/")[-1], data_2.split("/")[-1])
        # plot_3    d(p, p_prime, "initial", "./", 123, 'p', 'p\'')
        # plot_3d(p_prime, R.dot(p) + t, "Rp + t", "./", 123, 'p\'', 'Rp + t')

    return final_r_q, final_t, inliers, informationMatrix


def plot_matches(img1, kp1, img2, kp2, top_matches, inliers, base_data_path, image_i, image_j):
    plt.figure(figsize=(10, 5))
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, np.array(top_matches)[inliers].tolist(), None, flags=2)
    plt.imshow(img3)
    plt.title(" # inliers: " + str(len([x for x in inliers if x])))
    # plt.savefig("../output/feature_match/" + base_data_path.split("/")[3] + "." + str(image_i) + ".png")
    plt.savefig("../output/feature_match/path_output_1/" + str(image_i) + "." + str(image_j) + ".png")
    plt.close()


def plot_3d(q, q_prime, title, base_data_path, image_i, label_1, label_2):
    fig = plt.figure(figsize=[6, 6])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(q[0, :], q[1, :], q[2, :], c='k', label=label_1)
    ax.scatter(q_prime[0, :], q_prime[1, :], q_prime[2, :], c='r', marker='x', label=label_2)

    for i in range(q.shape[1]):
        plt.plot([q[0, i], q_prime[0, i]], [q[1, i], q_prime[1, i]], [q[2, i], q_prime[2, i]], 'k--')

    ax.legend()

    # plt.savefig("../output/3d_points/" + base_data_path.split("/")[3] + "." + str(image_i) + title + ".png")
    plt.savefig("../output/3d_points/." + str(image_i) + title + ".png")
    plt.close()
