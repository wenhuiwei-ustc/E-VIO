# E-VIO
This repository is the official implementation of the papers **E-VIO**: A Continual Evolving Visual-Inertial Odometry for Drones in Flight, which has been submitted to ISPRS Journal of Photogrammetry and Remote Sensing.

# Setup
## Installation
- Create conda environment
- Install g2opy, torch==1.10.0, torchvision==0.11.1

## Data preparation
To re-train or run the experiments from our paper, please download and pre-process the respective datasets.

### Cityscapes
Download the following files 
- `leftImg8bit_sequence_trainvaltest.zip`
- `timestamp_sequence.zip`
- `vehicle_sequence.zip`

### Oxford RobotCar
Download the following files 
- `22015-10-29-12-18-17_stereo_centre.tar`, `2015-10-29-12-18-17_gps.tar`
- `2015-02-03-08-45-10_stereo_centre.tar`, `2015-02-03-08-45-10_gps.tar`
- `2015-08-21-10-40-24_stereo_centre.tar`, `2015-08-21-10-40-24_gps.tar`

Undistort the center images:
```python
python datasets/robotcar.py <IMG_PATH> <MODELS_PATH>
```

### KITTI
Download the KITTI Odometry dataset
- `odometry data set`
- `odometry ground truth poses`

Extract the raw data matching the odometry dataset.
| 04        | 2011_09_30_drive_0016 |
| 09        | 2011_09_30_drive_0033 |
| 10        | 2011_09_30_drive_0034 |

### EuRoc
Download the EuRoc MAV dataset
- `MH_03, MH_05, V2_02`

#  Running the Code

## Pre-training
We pre-trained CoVIO on the Cityscapes Dataset.
```python
python main_pretrain.py
```

## Continual Learning with E-VIO
For continual learning, we used the KITTI Odometry Dataset, the Oxford RobotCar Dataset and the EuRoc MAC Dataset.
Then run:
```python
python main_adapt.py
```

## License

For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license.
For any commercial purpose, please contact the authors.


## Reference
[1] Continual SLAM: Beyond Lifelong Simultaneous Localization and Mapping Through Continual Learning
[2] Lite-Mono: A Lightweight CNN and Transformer Architecture for Self-Supervised Monocular Depth Estimation