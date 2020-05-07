from copy import deepcopy
from random import shuffle
from numpy.linalg import norm
import numpy as np

from scripts.steps.estimate_R_and_t import estimate_R_and_t


def ransac(p, p_prime, settings):
    th = settings['RANSAC_THRESHOLD']

    indices = list(range(p.shape[1]))

    max_S_k = 0
    inliers = []

    predictions_ignored = 0

    for k in range(settings["RANSAC_ITERATIONS"]):
        shuffle(indices)
        p_m = p[:,indices[:4]]
        p_prime_m = p_prime[:,indices[:4]]

        for i in range(4):
            for j in range(4):
                if norm(p_m[:, i] - p_prime_m[:, j]) < settings['RANSAC_MIN_OVERLAP_DISTANCE']:
                    # If any of the selected pairs in {p_m and p_prime_m} are too close (i.e. overlapping)
                    # ignore this prediction
                    predictions_ignored += 1
                    continue


        R,t = estimate_R_and_t(p_m, p_prime_m)

        p_projected = R.dot(p) + t
        error = norm(p_prime - p_projected, axis=0)

        if (error < th).sum() > max_S_k:
            max_S_k = (error < th).sum()
            inliers = deepcopy(list(error < th))

    if predictions_ignored > 0:
        print("predictions_ignored:", predictions_ignored)

    num_inliers = len([x for x in inliers if x])

    # Based on: (https://github.com/rising-turtle/visual_odometry/blob/master/src/VRO/camera_node.cpp#L78)
    informationMatrix = np.identity(6) * (num_inliers / (sum(error[inliers])**2))

    R,t = estimate_R_and_t(p[:,inliers], p_prime[:,inliers])
    return R,t,inliers,informationMatrix
