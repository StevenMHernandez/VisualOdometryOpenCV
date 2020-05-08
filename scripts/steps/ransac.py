from copy import deepcopy
from random import shuffle
from numpy.linalg import norm
import numpy as np

from scripts.steps.estimate_R_and_t import estimate_R_and_t


def ransac(p, p_prime, settings):
    threshold = settings['RANSAC_THRESHOLD']

    indices = list(range(p.shape[1]))

    max_S_k = 0
    inliers = []

    predictions_ignored = 0

    for k in range(settings["RANSAC_ITERATIONS"]):
        #
        # Select 4 pairs randomly
        #
        shuffle(indices)
        p_m = p[:,indices[:4]]
        p_prime_m = p_prime[:,indices[:4]]

        #
        # Ignore selected pairs in {p_m and p_prime_m} which are too close or overlapping.
        # Such overlapping pairs often result in bad predictions
        #
        for i in range(4):
            for j in range(4):
                if norm(p_m[:, i] - p_prime_m[:, j]) < settings['RANSAC_MIN_OVERLAP_DISTANCE']:
                    predictions_ignored += 1
                    continue


        #
        # Make estimation based on 4 randomly selected pairs
        #
        R,t = estimate_R_and_t(p_m, p_prime_m)

        #
        # Using predicted R and t to calculate maximum support (max_S_k)
        #
        p_projected = R.dot(p) + t
        error = norm(p_prime - p_projected, axis=0)
        num_supporting_this_hypothesis = (error < threshold).sum()
        if num_supporting_this_hypothesis > max_S_k:
            max_S_k = num_supporting_this_hypothesis
            inliers = deepcopy(list(error < threshold))

    if predictions_ignored > 0:
        print("predictions_ignored:", predictions_ignored)

    #
    # Make final prediction using all inliers
    #
    R,t = estimate_R_and_t(p[:,inliers], p_prime[:,inliers])

    #
    # Determine final inliers
    #
    p_projected = R.dot(p) + t
    error = norm(p_prime - p_projected, axis=0)
    inliers = deepcopy(list(error < threshold))

    #
    # Calculate information matrix
    # Based on: (https://github.com/rising-turtle/visual_odometry/blob/master/src/VRO/camera_node.cpp#L78)
    #
    num_inliers = len([x for x in inliers if x])
    informationMatrix = np.identity(6) * (num_inliers / (sum(error[inliers])**2))

    return R,t,inliers,informationMatrix
