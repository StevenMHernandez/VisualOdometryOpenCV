import cv2
from pandas import np
from transforms3d.euler import mat2euler

from scripts.steps.estimate_R_and_t import estimate_R_and_t
from scripts.steps.load_data import load_image
from scripts.steps.ransac import ransac

ATTRIBUTE = "Amplitude"


def predict_pose_change(data_1, data_2, settings, real_change, CALCULATE_ERROR):
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
    p_prime = np.array([list(depth_img2[int(kp2[m.trainIdx].pt[1])][int(kp2[m.trainIdx].pt[0])]) for m in top_matches]).T

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
    XYZ_scaling = 1000  # meters to millimeters
    final_r = np.degrees(mat2euler(R, "ryxz"))
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
    # plot_matches(img1, kp1, img2, kp2, top_matches, inliers, base_data_path, image_i)
    # plot_3d(p, p_prime, "initial", base_data_path, image_i, 'p', 'p\'')
    # plot_3d(p_prime, R.dot(p) + t, "Rp + t", base_data_path, image_i, 'p\'', 'Rp + t')
    # exit()

    return final_r, final_t, inliers
