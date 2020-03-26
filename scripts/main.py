from scripts.evaluate import evaluate

if __name__ == "__main__":
    print("Start")

    to_evaluate = {
        "../data/RV_Data/Pitch/d1_-40/d1_{0:04d}.dat": [
            "../data/RV_Data/Pitch/d2_-37/d2_{0:04d}.dat",
            "../data/RV_Data/Pitch/d3_-34/d3_{0:04d}.dat",
            "../data/RV_Data/Pitch/d4_-31/d4_{0:04d}.dat",
        ],
        "../data/RV_Data/Yaw/d1_44/d1_{0:04d}.dat": [
            "../data/RV_Data/Yaw/d2_41/d2_{0:04d}.dat",
            "../data/RV_Data/Yaw/d3_38/d3_{0:04d}.dat",
            "../data/RV_Data/Yaw/d4_35/d4_{0:04d}.dat",
        ],
        "../data/RV_Data/Translation/Y1/frm_{0:04d}.dat": [
            "../data/RV_Data/Translation/Y2/frm_{0:04d}.dat",
            "../data/RV_Data/Translation/Y3/frm_{0:04d}.dat",
            "../data/RV_Data/Translation/Y4/frm_{0:04d}.dat",
        ]
    }

    settings = {
        # "DETECTOR": "ORB",
        # "DETECTOR": "SIFT",
        "DETECTOR": "SURF",
        "FEATURE_FILTER_RATIO": 0.5,
        "MEDIAN_BLUR": False,
        "GAUSSIAN_BLUR": False,
    }

    f = open("output.csv", "w")

    f.write("X.mean,X.std,Y.mean,Y.std,Z.mean,Z.std,φ.mean,φ.std,θ.mean,θ.std,ψ.mean,ψ.std\n")

    for k in to_evaluate.keys():
        print(k)
        base_data_path = k
        for movement_data_path in to_evaluate[k]:
            results = evaluate(base_data_path, movement_data_path, settings)
            f.write(",".join([str(x) for x in results]) + "\n")
