# Checks if a matrix is a valid rotation matrix.
import math

import numpy as np

#
# SOURCE: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
#

def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotation_to_euler(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    def clamp(x, _min, _max):
        return max(_min, min(_max, x))

    if not singular:
        x = math.atan2(-R[1,2], R[2, 2])
        y = math.asin(clamp(R[0, 2], -1, 1))
        z = math.atan2(-R[0,1], R[0, 0])
    else:
        x = math.atan2(R[2,1], R[1, 1])
        y = math.asin(clamp(R[0, 2], -1, 1))
        z = 0


    return [np.degrees(x), np.degrees(y), np.degrees(z)]
