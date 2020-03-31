from copy import deepcopy
from random import shuffle
from numpy.linalg import norm

from scripts.steps.estimate_R_and_t import estimate_R_and_t


def ransac(p, p_prime):
    indices = list(range(p.shape[1]))
    th = 0.1

    max_S_k = 0
    inliers = None

    for k in range(100):
        shuffle(indices)
        p_m = p[:,indices[:4]]
        p_prime_m = p_prime[:,indices[:4]]

        R,t = estimate_R_and_t(p_m, p_prime_m)

        p_projected = R.dot(p) + t
        error = norm(p_prime - p_projected, axis=0)

        if (error < th).sum() > max_S_k:
            max_S_k = (error < th).sum()
            inliers = deepcopy(list(error < th))

    R,t = estimate_R_and_t(p[:,inliers], p_prime[:,inliers])
    return R,t,inliers
