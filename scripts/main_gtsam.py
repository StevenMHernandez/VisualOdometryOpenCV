"""
 * @file Pose3SLAMExample_initializePose3.cpp
 * @brief A 3D Pose SLAM example that reads input from g2o, and initializes the
 *  Pose3 using InitializePose3
 * @date Jan 17, 2019
 * @author Vikrant Shah based on CPP example by Luca Carlone
"""
# pylint: disable=invalid-name, E1101

from __future__ import print_function
import argparse
from time import sleep

import numpy as np
import matplotlib.pyplot as plt

import gtsam
from gtsam.utils import plot

optimizer = None


def vector6(x, y, z, a, b, c):
    """Create 6d double numpy array."""
    return np.array([x, y, z, a, b, c], dtype=np.float)

def plot_it(result):
    resultPoses = gtsam.allPose3s(result)
    xyz = [resultPoses.atPose3(i).translation() for i in range(resultPoses.size())]
    x_ = [pose.x() for pose in xyz]
    y_ = [pose.y() for pose in xyz]
    z_ = [pose.z() for pose in xyz]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(resultPoses.size()):
        plot.plot_pose3(1, resultPoses.atPose3(i))
        ax.text(x_[i], y_[i], z_[i], str(i))
    plt.plot(x_, y_, z_, 'k-')
    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="A 3D Pose SLAM example that reads input from g2o, and "
                    "initializes Pose3")
    parser.add_argument('-i', '--input', help='input file g2o format')
    parser.add_argument('-o', '--output',
                        help="the path to the output file with optimized graph")
    parser.add_argument("-p", "--plot", action="store_true",
                        help="Flag to plot results")
    args = parser.parse_args()

    g2oFile = "../output/output.g2o"

    is3D = True
    graph, initial = gtsam.readG2o(g2oFile, is3D)

    # Add Prior on the first key
    priorModel = gtsam.noiseModel_Diagonal.Variances(vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4))

    print("Adding prior to g2o file ")
    graphWithPrior = graph
    firstKey = initial.keys().at(0)
    graphWithPrior.add(gtsam.PriorFactorPose3(firstKey, gtsam.Pose3(), priorModel))

    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(1000)
    optimizer = gtsam.LevenbergMarquardtOptimizer(graphWithPrior, initial, params)
    result = optimizer.optimize()
    optimizer.iterate()
    optimizer.iterate()

    print("Optimization complete")

    plot_it(initial)
    plot_it(result)

    print("initial error = ", graphWithPrior.error(initial))
    print("final error =   ", graphWithPrior.error(result))