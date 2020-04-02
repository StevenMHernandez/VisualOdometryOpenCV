import numpy as np
import cv2
import matplotlib.pyplot as plt
from numpy.core.defchararray import lower

from scripts.steps.predict_pose_change import predict_pose_change

cv = cv2


def evaluate(base_data_path, movement_data_path, settings, real_change, CALCULATE_ERROR):
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
        final_r, final_t, inliers = predict_pose_change(base_data_path.format(image_i), movement_data_path.format(image_i), settings, real_change, CALCULATE_ERROR)

        list_R.append(final_r)
        list_t.append(final_t)
        list_inliers.append(inliers)

    R_bar = np.array(list_R)
    t_bar = np.array(list_t)

    XYZ_mean = t_bar.mean(axis=0)
    XYZ_std = t_bar.std(axis=0)
    RPY_mean = R_bar.mean(axis=0)
    RPY_std = R_bar.std(axis=0)

    # #
    # # Plot CDF
    # #
    # if movement_type == 'translation':
    #     plot_cdf(t_bar, "Translation", movement_type, real_change, xyz=True)
    # else:
    #     plot_cdf(R_bar, "Rotation", movement_type, real_change, rpy=True)

    # #
    # # Plot all predictions per Rotation and Translation
    # #
    # plt.figure(figsize=(12, 8))
    # plt.subplot(1, 2, 1)
    # plot_means(R_bar, RPY_mean, RPY_std, "Rotation")
    # plt.subplot(1, 2, 2)
    # plot_means(t_bar, XYZ_mean, XYZ_std, "Translation")
    # plt.suptitle("{}\n{}".format(base_data_path, movement_data_path))
    # plt.show()

    # #
    # # Plot number of inliers used.
    # #
    # X = sorted([sum(x) for x in list_inliers])
    # plt.plot(X)
    # plt.show()

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
    plt.figure(figsize=[7, 4])
    for i in range(3):
        _X = X[:, i]
        _step_size = 0.1
        _range = np.arange(_X.min() - _step_size, _X.max() + _step_size, _step_size)
        _cdf_sum = np.zeros(_range.shape)
        for i, v in enumerate(_range):
            _cdf_sum[i] = (_X < v).sum()

        plt.plot(_range, _cdf_sum / len(_X))
    plt.legend(["X", "Y", "Z"] if xyz else ["φ", "θ", "ψ", ])
    plt.xlabel("Error ({})".format("mm" if xyz else "degrees"))
    plt.ylabel("CDF")
    plt.xlim([0, X.max()])
    plt.ylim([0, 1])

    plt.savefig("../output/cdf_{}_{}_{}.png".format("t" if xyz else "r", movement_type, max(real_change.values())))
    plt.close()


def plot_matches(img1, kp1, img2, kp2, top_matches, inliers, base_data_path, image_i):
    plt.figure(figsize=(10, 5))
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, np.array(top_matches)[inliers].tolist(), None, flags=2)
    plt.imshow(img3)
    plt.savefig("../output/feature_match/" + base_data_path.split("/")[3] + "." + str(image_i) + ".png")
    plt.close()


def plot_3d(q, q_prime, title, base_data_path, image_i, label_1, label_2):
    fig = plt.figure(figsize=[6, 6])
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(q[0, :], q[1, :], q[2, :], c='k', label=label_1)
    ax.scatter(q_prime[0, :], q_prime[1, :], q_prime[2, :], c='r', marker='x', label=label_2)

    for i in range(q.shape[1]):
        plt.plot([q[0, i], q_prime[0, i]], [q[1, i], q_prime[1, i]], [q[2, i], q_prime[2, i]], 'k--')

    ax.legend()

    plt.savefig("../output/3d_points/" + base_data_path.split("/")[3] + "." + str(image_i) + title + ".png")
    plt.close()
