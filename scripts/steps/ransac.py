from copy import deepcopy
from random import shuffle
from numpy.linalg import norm
import numpy as np

from scripts.steps.estimate_R_and_t import estimate_R_and_t


def ransac(p, p_prime, th):
    indices = list(range(p.shape[1]))

    max_S_k = 0
    inliers = []

    for k in range(25):
        shuffle(indices)
        p_m = p[:,indices[:4]]
        p_prime_m = p_prime[:,indices[:4]]

        R,t = estimate_R_and_t(p_m, p_prime_m)

        p_projected = R.dot(p) + t
        error = norm(p_prime - p_projected, axis=0)

        if (error < th).sum() > max_S_k:
            max_S_k = (error < th).sum()
            inliers = deepcopy(list(error < th))

    num_inliers = len([x for x in inliers if x])

    # Based on: (https://github.com/rising-turtle/visual_odometry/blob/master/src/VRO/camera_node.cpp#L78)
    informationMatrix = np.identity(6) * (num_inliers / (sum(error[inliers])**2))

    R,t = estimate_R_and_t(p[:,inliers], p_prime[:,inliers])
    return R,t,inliers,informationMatrix
