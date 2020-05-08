import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # f = open("../output/output__1.g2o")
    # f = open("../output/output__10.g2o")
    # f = open("../output/output__1000.g2o")
    f = open("../output/output.g2o")
    SIZE = 583

    adjacencyMatrix = np.zeros([SIZE, SIZE])
    globalXYZ = np.zeros([SIZE, 3])
    globalQuaternion = np.zeros([SIZE, 4])
    localXYZ = np.zeros([SIZE, 3])
    localQuaternion = np.zeros([SIZE, 4])

    for l in f.readlines():
        if "VERTEX_SE3:QUAT" in l:
            print(l)
            ll = l.split(" ")
            image_i = int(ll[1])
            if image_i < SIZE:
                globalXYZ[image_i, 0] = float(ll[2])
                globalXYZ[image_i, 1] = float(ll[3])
                globalXYZ[image_i, 2] = float(ll[4])
                globalQuaternion[image_i, 0] = float(ll[5])
                globalQuaternion[image_i, 1] = float(ll[6])
                globalQuaternion[image_i, 2] = float(ll[7])
                globalQuaternion[image_i, 3] = float(ll[8])
        if "EDGE_SE3:QUAT" in l:
            print(l)
            ll = l.split(" ")
            image_i = int(ll[1])
            image_j = int(ll[2])
            if image_j < SIZE:
                adjacencyMatrix[image_i, image_j] = 1
            if image_i < SIZE:
                if image_i + 1 == image_j:
                    localXYZ[image_i, 0] = float(ll[3])
                    localXYZ[image_i, 1] = float(ll[4])
                    localXYZ[image_i, 2] = float(ll[5])
                    localQuaternion[image_i, 0] = float(ll[6])
                    localQuaternion[image_i, 1] = float(ll[7])
                    localQuaternion[image_i, 2] = float(ll[8])
                    localQuaternion[image_i, 3] = float(ll[9])

    plt.imshow(adjacencyMatrix)
    plt.xlabel("image_i index")
    plt.ylabel("image_j index")
    plt.show()

    Z = adjacencyMatrix.sum(axis=1)
    Z.sort()
    plt.plot(Z)
    plt.show()


    plt.subplot(2,1,1)
    plt.plot(globalXYZ[:,:])
    plt.legend(["x","y","z"])
    plt.xlabel("image_i index")
    plt.ylabel("Estimated Pose (mm)")
    plt.subplot(2,1,2)
    plt.plot((globalQuaternion[:,:]))
    plt.legend(["qw", "qx","qy","qz"])
    plt.ylabel("Estimated Pose (Quaternions)")
    plt.xlabel("image_i index")
    plt.show()

    plt.subplot(2,1,1)
    plt.plot(localXYZ[:,:])
    plt.legend(["x","y","z"])
    plt.xlabel("image_i index")
    plt.ylabel("Estimated Change (mm)")
    plt.subplot(2,1,2)
    plt.plot((localQuaternion[:,:]))
    plt.legend(["qw", "qx","qy","qz"])
    plt.xlabel("image_i index")
    plt.ylabel("Estimated Change (Quaternions)")
    plt.show()

    print("Number of edges:", sum(sum(adjacencyMatrix > 0)))
