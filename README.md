# Visual Odometry and Graph SLAM with OpenCV and GTSAM (Python)

## Visual Odometry

```
python scripts/main.py
```

Code for predicting the pose change for a pair of samples can be found in `./scripts/steps/predict_pose_change.py`. This is the most important file to consider for Visual Odomotry.

## Graph SLAM

First, build the .g2o file by running the following script
This script performs visual odometry measurements on all image pairs 
file is saved to `./output/output.g2o`

```
python scripts/main_build_g2o.py
```

Next, perform GTSAM optimization on the `.g2o` file.

```
python scripts/main_gtsam.py
```
