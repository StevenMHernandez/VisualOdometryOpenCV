from scripts.evaluate import evaluate

if __name__ == "__main__":
    print("Start")

    to_evaluate = {
        "../data/RV_Data/Pitch/d1_-40/d1_{0:04d}.dat": [
            ({"pitch": 3}, "../data/RV_Data/Pitch/d2_-37/d2_{0:04d}.dat"), # tuple: (value changed, directory for files)
            ({"pitch": 6}, "../data/RV_Data/Pitch/d3_-34/d3_{0:04d}.dat"),
            ({"pitch": 9}, "../data/RV_Data/Pitch/d4_-31/d4_{0:04d}.dat"),
        ],
        "../data/RV_Data/Yaw/d1_44/d1_{0:04d}.dat": [
            ({"yaw": 3}, "../data/RV_Data/Yaw/d2_41/d2_{0:04d}.dat"),
            ({"yaw": 6}, "../data/RV_Data/Yaw/d3_38/d3_{0:04d}.dat"),
            ({"yaw": 9}, "../data/RV_Data/Yaw/d4_35/d4_{0:04d}.dat"),
        ],
        "../data/RV_Data/Translation/Y1/frm_{0:04d}.dat": [
            ({"y": 100}, "../data/RV_Data/Translation/Y2/frm_{0:04d}.dat"),
            ({"y": 200}, "../data/RV_Data/Translation/Y3/frm_{0:04d}.dat"),
            ({"y": 300}, "../data/RV_Data/Translation/Y4/frm_{0:04d}.dat"),
        ]
    }

    settings = {
        "KNN_MATCHING_RATIO": 0.75,
        # "KNN_MATCHING_RATIO": 0,

        "RANSAC_THRESHOLD": 0.1,
        # "RANSAC_THRESHOLD": 0,

        "DETECTOR": "SIFT",
        # "DETECTOR": "SURF",

        "MEDIAN_BLUR": False,
        "GAUSSIAN_BLUR": False,
    }

    f = open("../output/output.csv", "w")

    f.write("change, X.mean,X.std,Y.mean,Y.std,Z.mean,Z.std,φ.mean,φ.std,θ.mean,θ.std,ψ.mean,ψ.std\n")

    for k in to_evaluate.keys():
        print(k)
        base_data_path = k
        for change, movement_data_path in to_evaluate[k]:
            real_change = {
                "x": 0,
                "y": 0,
                "z": 0,
                "roll": 0,
                "pitch": 0,
                "yaw": 0,
            }
            for k in change.keys():
                real_change[k] = change[k]
            left_side = "x y z: ({} {} {}) φ θ ψ: ({} {} {}),".format(
                real_change['x'],
                real_change['y'],
                real_change['z'],
                real_change['roll'],
                real_change['pitch'],
                real_change['yaw']
            )

            results = evaluate(base_data_path, movement_data_path, settings, real_change, CALCULATE_ERROR=True)
            f.write(left_side + ",".join(["{:.3f}".format(x) for x in results]) + "\n")
            f.flush()

    print("done")
