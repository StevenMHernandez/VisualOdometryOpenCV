import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.defchararray import lower

from scripts.steps.predict_pose_change import predict_pose_change

cv = cv2


def evaluate(base_data_path, movement_data_path, settings, real_change, CALCULATE_ERROR):
    list_index = []
    list_R = []
    list_t = []
    list_inliers = []

    movement_type = lower(base_data_path.split("/")[3])

    print(base_data_path)
    print(movement_data_path)

    for image_i in range(1, 100):
        print(image_i)

        #
        # Predict Pose Change
        #
        try:
            _, final_r, final_t, inliers, _ = predict_pose_change(base_data_path.format(image_i), movement_data_path.format(image_i), settings, real_change, CALCULATE_ERROR=False, print_image=True)

            list_index.append(image_i)
            list_R.append(final_r)
            list_t.append(final_t)
            list_inliers.append(inliers)
        except Exception as e:
            print("some exception occured. skipping.")

    R_bar = np.array(list_R)
    t_bar = np.array(list_t)

    XYZ_mean = t_bar.mean(axis=0)
    XYZ_std = t_bar.std(axis=0)
    RPY_mean = R_bar.mean(axis=0)
    RPY_std = R_bar.std(axis=0)

    #
    # Plot CDF
    #
    # if movement_type == 'translation':
    plt.figure(figsize=[7, 7])
    plt.subplot(2,1,1)
    plot_cdf((t_bar), "Translation", movement_type, real_change, xyz=True)
    # else:
    plt.subplot(2,1,2)
    plot_cdf((R_bar), "Rotation", movement_type, real_change, rpy=True)
    plt.show()

    #
    # Plot all predictions per Rotation and Translation
    #
    plt.figure(figsize=(12, 8))
    plt.subplot(1, 2, 1)
    plot_means(R_bar, RPY_mean, RPY_std, "Rotation")
    plt.xlabel("image-pair index")
    plt.ylabel("prediction")
    plt.subplot(1, 2, 2)
    plot_means(t_bar, XYZ_mean, XYZ_std, "Translation")
    plt.ylabel("prediction")
    plt.xlabel("image-pair index")
    plt.suptitle("{}\n{}".format(base_data_path, movement_data_path))
    plt.show()

    plt.plot((t_bar))
    for i in range(len(list_index)):
        print()
        plt.text(i, (t_bar)[i][0], "#" + str(list_index[i]))
    plt.legend(["X", "Y", "Z"])
    plt.ylabel("prediction")
    plt.show()

    plt.plot((R_bar))
    for i in range(len(list_index)):
        print()
        plt.text(i, (R_bar)[i][0], "#" + str(list_index[i]))
    plt.legend(["φ", "θ", "ψ", ])
    plt.ylabel("prediction")
    plt.show()

    #
    # Plot number of inliers used.
    #
    print()
    print()
    X = [sum(x) for x in list_inliers]
    X_indices = np.argsort(X)
    X = np.array(X)[X_indices]
    M = np.array(list_index)[X_indices]
    plt.plot(range(len(M)), X, 'ro')
    plt.plot(range(len(M)), X, 'k.')
    for i in range(len(M)):
        print("list_index", M)
        print(i)
        plt.text(i, X[i] + 0.25, str(M[i]))
    plt.xlabel("image-pair index")
    plt.ylabel("Number of inliers")
    plt.title("number of inliers (sorted)")
    plt.show()
    print("X_indices", X_indices)
    print("list_index", M)

    #
    # Plot number of inliers used.
    #
    print()
    print()
    X = [sum(x) for x in list_inliers]
    X_indices = np.argsort(X)
    X = np.array(X)[X_indices]
    M = np.array(list_index)[X_indices]
    plt.plot(range(len(M)), X, 'ro')
    plt.plot(range(len(M)), X, 'k.')
    for i in range(len(M)):
        print("list_index", M)
        print(i)
        plt.text(i, X[i] + 0.25, str(M[i]))
    plt.xlabel("image-pair index")
    plt.ylabel("Number of inliers")
    plt.title("number of inliers (sorted)")
    plt.show()
    print("X_indices", X_indices)
    print("list_index", M)

    return [
        XYZ_mean[0], XYZ_std[0],  # X
        XYZ_mean[1], XYZ_std[1],  # Y
        XYZ_mean[2], XYZ_std[2],  # Z
        RPY_mean[0], RPY_std[0],  # φ
        RPY_mean[1], RPY_std[1],  # θ
        RPY_mean[2], RPY_std[2],  # ψ
    ]


def plot_means(_list, _mean, _std, name):
    plt.plot(sorted(_list[:, 0]), '.-')
    plt.plot(sorted(_list[:, 1]), '.-')
    plt.plot(sorted(_list[:, 2]), '.-')
    plt.legend(["x", "y", "z"])
    plt.title("\n\n{}\n{}\n{}".format(name, _mean, _std))


def plot_cdf(X, title, movement_type, real_change, xyz=False, rpy=False):
    # plt.figure(figsize=[7, 4])
    for i in range(3):
        _X = X[:, i]
        _step_size = 0.1
        _range = np.arange(_X.min() - _step_size, _X.max() + _step_size, _step_size)
        _cdf_sum = np.zeros(_range.shape)
        for i, v in enumerate(_range):
            _cdf_sum[i] = (_X < v).sum()

        plt.plot(_range, _cdf_sum / len(_X))
    plt.legend(["X", "Y", "Z"] if xyz else ["φ", "θ", "ψ", ])
    plt.xlabel("Prediction ({})".format("mm" if xyz else "degrees"))
    plt.ylabel("CDF")
    plt.xlim([X.min(), X.max()])
    plt.ylim([0, 1])

    # plt.show()

    # plt.savefig("../output/cdf_{}_{}_{}.png".format("t" if xyz else "r", movement_type, max(real_change.values())))
    # plt.close()
