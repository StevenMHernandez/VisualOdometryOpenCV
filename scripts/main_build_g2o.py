from copy import deepcopy
from time import time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from pyquaternion import Quaternion

from steps.predict_pose_change import predict_pose_change

if __name__ == "__main__":
    print("Start")
    start = time()

    to_evaluate = {
        "path": "../data/RV_Data2/d1_{0:04d}.dat",
        "files_start": 1,
        "num_files": 584,
        "span": 1000,
    }

    settings = {
        "RANSAC_ITERATIONS": 25,  # number
        "RANSAC_THRESHOLD": 0.01,  # error (m)
        "RANSAC_MIN_OVERLAP_DISTANCE": 0.125,  # cm
        "MAX_PAIR_MOVEMENT_DISTANCE": 25,  # cm
        "MAX_ALLOWED_POSE_CHANGE_XYZ": 75,  # cm
        "MIN_NUMBER_OF_INLIERS": 8,  # number

        "MEDIAN_BLUR": False,
        "GAUSSIAN_BLUR": False,
    }

    f = open("../output/output.g2o", "w")

    base_data_path = to_evaluate["path"]
    vertex_output = "VERTEX_SE3:QUAT 0 0 0 0 1 0 0 0 \n"
    edge_output = ""

    num_inliers_matrix = np.zeros(
        [to_evaluate["num_files"] - to_evaluate["files_start"], to_evaluate["num_files"] - to_evaluate["files_start"]])

    global_xyz_sum = np.array([0.0, 0.0, 0.0])
    global_quaternions_sum = Quaternion(1, 0, 0, 0)

    previous_global_xyz_sum = np.array([0.0, 0.0, 0.0])
    previous_global_quaternions_sum = Quaternion(1, 0, 0, 0)
    for image_i in range(to_evaluate["files_start"], to_evaluate["num_files"]):
        previous_global_xyz_sum = global_xyz_sum
        previous_global_quaternions_sum = global_quaternions_sum

        for image_j in range(image_i + 1, min(image_i + 1 + to_evaluate["span"], to_evaluate["num_files"])):
        # for image_j in range(image_i + 1, to_evaluate["num_files"]):
            if image_i != image_j:
                print()
                print(image_i, image_j)

                #
                # Predict Pose Change
                #
                if image_i == image_j - 1:
                    #
                    # We need to guarantee that concurrent image frames produce some pose change prediction.
                    # To accomplish this,
                    #
                    _settings = deepcopy(settings)
                    while True:
                        print("=====================")
                        print("RANSAC_THRESHOLD", _settings["RANSAC_THRESHOLD"])
                        print("MAX_PAIR_MOVEMENT_DISTANCE", _settings["MAX_PAIR_MOVEMENT_DISTANCE"])
                        print("MAX_ALLOWED_POSE_CHANGE_XYZ", _settings["MAX_ALLOWED_POSE_CHANGE_XYZ"])
                        print("MIN_NUMBER_OF_INLIERS", _settings["MIN_NUMBER_OF_INLIERS"])
                        print("=====================")
                        try:
                            final_r_q, _, final_t, inliers, informationMatrix = predict_pose_change(
                                base_data_path.format(image_i),
                                base_data_path.format(image_j), _settings,
                                real_change=None, CALCULATE_ERROR=False, print_image=False)

                            print("inlier:", len([x for x in inliers if x]))
                            break
                        except Exception as e:
                            if ("Not enough inliers" in str(e)):
                                _settings["MIN_NUMBER_OF_INLIERS"] = max(3, _settings["MIN_NUMBER_OF_INLIERS"] - 1)
                            print(e)
                            _settings["RANSAC_THRESHOLD"] *= 1.25
                            _settings["MAX_PAIR_MOVEMENT_DISTANCE"] *= 1.125
                            _settings["MAX_ALLOWED_POSE_CHANGE_XYZ"] *= 1.0125
                else:
                    try:
                        final_r_q, _, final_t, inliers, informationMatrix = predict_pose_change(
                            base_data_path.format(image_i),
                            base_data_path.format(image_j), settings,
                            real_change=None, CALCULATE_ERROR=False, print_image=False)
                    except:
                        print("error when performing `predict_pose_change()`")
                        continue

                num_inliers = len([x for x in inliers if x])
                num_inliers_matrix[image_i - to_evaluate["files_start"], image_j - to_evaluate["files_start"]] = num_inliers

                #
                # Compute: local-change and global-change: (sum of local changes).
                #
                if image_i == image_j - 1:
                    if image_i == 1:
                        global_xyz_sum = previous_global_xyz_sum + np.array(final_t)
                    else:
                        global_xyz_sum = previous_global_xyz_sum + (previous_global_quaternions_sum).rotate(np.array(final_t))
                    global_quaternions_sum *= Quaternion(final_r_q)
                    # global_quaternions_sum *= Quaternion(final_r_q[3], final_r_q[0], final_r_q[1], final_r_q[2])
                    output_string = " ".join([str(x) for x in global_xyz_sum]) + " " + " ".join([str(x) for x in global_quaternions_sum])
                    vertex_output += "VERTEX_SE3:QUAT " + str(image_i) + " " + output_string + " \n"

                output_string = " ".join([str(x) for x in final_t]) + " " + " ".join([str(x) for x in final_r_q])
                edge_output += "EDGE_SE3:QUAT " + str(image_i - 1) + " " + str(image_j - 1) + " " + output_string + " "
                for i in range(0, 6):
                    for j in range(i, 6):
                        edge_output += str(informationMatrix[i, j]) + " "
                edge_output += "\n"

    f.write(vertex_output)
    f.write(edge_output)

    pd.DataFrame(num_inliers_matrix).to_csv("inlier_matrix.csv")

    plt.imshow(num_inliers_matrix)
    plt.colorbar()
    plt.show()

    plt.imshow(num_inliers_matrix > 0)
    plt.show()

    end = time()

    print("Time taken (s) :", (end - start))
    print("done")
