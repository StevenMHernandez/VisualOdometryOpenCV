import numpy as np
from numpy.linalg import det
from scipy.linalg import svd
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler


def plot_3d(q, q_prime, title=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(q[0, :], q[1, :], q[2, :], c='k')
    ax.scatter(q_prime[0, :], q_prime[1, :], q_prime[2, :], c='r', marker='x')

    for i in range(len(q) + 1):
        plt.plot([q[0, i], q_prime[0, i]], [q[1, i], q_prime[1, i]], [q[2, i], q_prime[2, i]], 'k--')
    plt.title(title)
    plt.show()


#
# REFERENCE: http://nghiaho.com/?page_id=671 was helpful in further validating
#
def estimate_R_and_t(A, B):
    #
    # Get Centroid ✓
    # Test: |v| = 3
    # Test:
    #
    centroid_A = np.atleast_2d(A.mean(axis=1)).T
    centroid_B = np.atleast_2d(B.mean(axis=1)).T

    #
    # Get H
    # Test: |H| = (3,3)
    #
    H = (A - centroid_A).dot((B - centroid_B).T)

    #
    # SVD
    # Test: H == U.dot(np.diag(S)).dot(V.T)
    #
    U, S, V_h = svd(H)
    V = V_h.T

    #
    # Get Rotation
    #
    R = V.dot(U.T)

    if det(R) < 0:
        V[:, 2] *= -1  # Multiply final column of V
        R = V.dot(U.T)

    #
    # Get Translation
    #
    t = centroid_B - (R.dot(centroid_A))

    # plot_3d(A, B)
    # plot_3d(B, R.dot(A) + t)

    return R, t


if __name__ == "__main__":
    A = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    # B = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]])
    B = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 123]])
    # B = (np.array([[0, 0, -1, -1], [0, 1, 1, 1], [0, 0, 0, 1]]).T).T
    # B = (np.array([[0, 0, -1, -1], [0, 1, 1, 1], [0, 0, 0, 1]]).T + [-10, 10.5, 0]).T

    R, t = estimate_R_and_t(A, B)
    for s in "sr":
        for x in "xyz":
            for y in "xyz":
                if x != y:
                    for z in "xyz":
                        if x != z and y != z:
                            option = s + x + y + z
                            print(option, np.degrees(mat2euler(R, option)).round())

    # print(A)
    # A_prime = A.copy()
    # tmp = A_prime[1,:].copy()
    # A_prime[1,:] = A_prime[2,:].copy()
    # A_prime[2,:] = tmp
    # B_prime = B.copy()
    # tmp = B_prime[1,:].copy()
    # B_prime[1,:] = B_prime[2,:].copy()
    # B_prime[2,:] = tmp
    #
    # print(A_prime)
    #
    # R, t = estimate_R_and_t(A, B)
    # xyz = np.degrees(mat2euler(R))
    # print(R)
    # print(t)
    # print(xyz)
    #
    # R, t = estimate_R_and_t(A_prime, B_prime)
    # xyz = np.degrees(mat2euler(R))
    # print(R)
    # print(t)
    # print(xyz)
