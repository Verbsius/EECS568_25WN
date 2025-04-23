# Deep Learning Approach to Global Bird's Eye View Semantic Mapping

# Setup
1. Clone the repo
2. Download the AI-IMU dataset and store in root directory.
3. Download the local BEV data 
4. Store `MotionNet_Prediction` and `MotionNet_Prediction_Test` folders from the data in root directory.
5. Create `temp` and `results` folder under root directory.
6. Run `python3 kitti_mapping_opt.py`
7. If you want to run `main_KITTI.py`, please call it under root directory, and run `python3 ai_imu_dr/src/main_kitti.py`

# Results

<div style="display: flex; justify-content: space-between;">
    <img src="media/scene_07_gt_mean.png" alt="GT Mean" width="24%">
    <img src="media/scene_07_test_mean.png" alt="Test Mean" width="24%">
    <img src="media/scene_07_test_variance.png" alt="Test Variance" width="24%">
    <img src="media/scene_07_trajectory_heading_rotated.png" alt="Trajectory Heading" width="24%">
</div>

<div style="display: flex; justify-content: space-between;">
    <img src="media/scene_08_gt_mean.png" alt="GT Mean" width="24%">
    <img src="media/scene_08_test_mean.png" alt="Test Mean" width="24%">
    <img src="media/scene_08_test_variance.png" alt="Test Variance" width="24%">
    <img src="media/scene_08_trajectory_heading_rotated.png" alt="Trajectory Heading" width="24%">
</div>

<div style="display: flex; justify-content: space-between;">
    <img src="media/scene_09_gt_mean.png" alt="GT Mean" width="24%">
    <img src="media/scene_09_test_mean.png" alt="Test Mean" width="24%">
    <img src="media/scene_09_test_variance.png" alt="Test Variance" width="24%">
    <img src="media/scene_09_trajectory_heading_rotated.png" alt="Trajectory Heading" width="24%">
</div>

## Poster

The poster can be found at `media/poster.pdf`.

## Presentation Video
