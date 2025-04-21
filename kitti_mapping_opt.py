"""
File: kitti_mapping_opt.py
Last Modified: 4/9/2025
Execution Time: 3 min
Description: This script processes KITTI dataset sequences to generate global 
             maps using BEV (Bird's Eye View) mapping. It includes functionality 
             for trajectory estimation, global map generation, and visualization 
             of results such as mean, variance, and trajectory heading.
"""

# %%
import sys
sys.path.append('ai_imu_dr/src')

import os
import shutil
import numpy as np
from collections import namedtuple
import glob
import time
import datetime
import pickle
import torch
import matplotlib.pyplot as plt
from termcolor import cprint
from navpy import lla2ned
from collections import OrderedDict
from ai_imu_dr.src.dataset import BaseDataset
from ai_imu_dr.src.utils_torch_filter import TORCHIEKF
from ai_imu_dr.src.utils_numpy_filter import NUMPYIEKF as IEKF
from ai_imu_dr.src.utils import prepare_data, umeyama_alignment
from scipy.spatial.transform import Rotation
import pdb
# %%

LABEL_COLORS = np.array([
    (255, 255, 255),  # unlabeled
    (245, 150, 100),  # car
    (245, 230, 100),  # bike
    (30, 30, 255),    # person
    (255, 0, 255),    # road
    (255, 150, 255),  # parking
    (75, 0, 75),      # sidewalk
    (0, 200, 255),    # building
    (50, 120, 255),   # fence
    (0, 175, 0),      # vegetation
    (80, 240, 150),   # terrain
    (150, 240, 255),  # pole
    (90, 30, 150),    # traffic-sign
    (255, 0, 0),      # moving-car
    (0, 0, 255),      # moving-person
]).astype(np.uint8)

# Download data from https://www.dropbox.com/s/ey41xsvfqca30vv/data.zip
# Have it in top level, in folder called data

# Kitti classes from ai_imu_dr/src/main_kitti.py
class KITTIParameters(IEKF.Parameters):
    # gravity vector
    g = np.array([0, 0, -9.80655])

    cov_omega = 2e-4
    cov_acc = 1e-3
    cov_b_omega = 1e-8
    cov_b_acc = 1e-6
    cov_Rot_c_i = 1e-8
    cov_t_c_i = 1e-8
    cov_Rot0 = 1e-6
    cov_v0 = 1e-1
    cov_b_omega0 = 1e-8
    cov_b_acc0 = 1e-3
    cov_Rot_c_i0 = 1e-5
    cov_t_c_i0 = 1e-2
    cov_lat = 1
    cov_up = 10

    def __init__(self, **kwargs):
        super(KITTIParameters, self).__init__(**kwargs)
        self.set_param_attr()

    def set_param_attr(self):
        attr_list = [a for a in dir(KITTIParameters) if
                     not a.startswith('__') and not callable(getattr(KITTIParameters, a))]
        for attr in attr_list:
            setattr(self, attr, getattr(KITTIParameters, attr))


class KITTIDataset(BaseDataset):
    OxtsPacket = namedtuple('OxtsPacket',
                            'lat, lon, alt, ' + 'roll, pitch, yaw, ' + 'vn, ve, vf, vl, vu, '
                                                                       '' + 'ax, ay, az, af, al, '
                                                                            'au, ' + 'wx, wy, wz, '
                                                                                     'wf, wl, wu, '
                                                                                     '' +
                            'pos_accuracy, vel_accuracy, ' + 'navstat, numsats, ' + 'posmode, '
                                                                                  'velmode, '
                                                                                  'orimode')

    # Bundle into an easy-to-access structure
    OxtsData = namedtuple('OxtsData', 'packet, T_w_imu')
    min_seq_dim = 25 * 100  # 60 s
    datasets_fake = ['2011_09_26_drive_0093_extract', '2011_09_28_drive_0039_extract',
                     '2011_09_28_drive_0002_extract']
    """
    '2011_09_30_drive_0028_extract' has trouble at N = [6000, 14000] -> test data
    '2011_10_03_drive_0027_extract' has trouble at N = 29481
    '2011_10_03_drive_0034_extract' has trouble at N = [33500, 34000]
    """

    # training set to the raw data of the KITTI dataset.
    # The following dict lists the name and end frame of each sequence that
    # has been used to extract the visual odometry / SLAM training set
    odometry_benchmark = OrderedDict()
    odometry_benchmark["2011_10_03_drive_0027_extract"] = [0, 45692]
    odometry_benchmark["2011_10_03_drive_0042_extract"] = [0, 12180]
    odometry_benchmark["2011_10_03_drive_0034_extract"] = [0, 47935]
    odometry_benchmark["2011_09_26_drive_0067_extract"] = [0, 8000]
    odometry_benchmark["2011_09_30_drive_0016_extract"] = [0, 2950]
    odometry_benchmark["2011_09_30_drive_0018_extract"] = [0, 28659]
    odometry_benchmark["2011_09_30_drive_0020_extract"] = [0, 11347]
    odometry_benchmark["2011_09_30_drive_0027_extract"] = [0, 11545]
    odometry_benchmark["2011_09_30_drive_0028_extract"] = [11231, 53650]
    odometry_benchmark["2011_09_30_drive_0033_extract"] = [0, 16589]
    odometry_benchmark["2011_09_30_drive_0034_extract"] = [0, 12744]

    odometry_benchmark_img = OrderedDict()
    odometry_benchmark_img["2011_10_03_drive_0027_extract"] = [0, 45400]
    odometry_benchmark_img["2011_10_03_drive_0042_extract"] = [0, 11000]
    odometry_benchmark_img["2011_10_03_drive_0034_extract"] = [0, 46600]
    odometry_benchmark_img["2011_09_26_drive_0067_extract"] = [0, 8000]
    odometry_benchmark_img["2011_09_30_drive_0016_extract"] = [0, 2700]
    odometry_benchmark_img["2011_09_30_drive_0018_extract"] = [0, 27600]
    odometry_benchmark_img["2011_09_30_drive_0020_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0027_extract"] = [0, 11000]
    odometry_benchmark_img["2011_09_30_drive_0028_extract"] = [11000, 51700]
    odometry_benchmark_img["2011_09_30_drive_0033_extract"] = [0, 15900]
    odometry_benchmark_img["2011_09_30_drive_0034_extract"] = [0, 12000]

    def __init__(self, args):
        super(KITTIDataset, self).__init__(args)

        self.datasets_validatation_filter['2011_09_30_drive_0028_extract'] = [11231, 53650]
        self.datasets_train_filter["2011_10_03_drive_0042_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0018_extract"] = [0, 15000]
        self.datasets_train_filter["2011_09_30_drive_0020_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0027_extract"] = [0, None]
        self.datasets_train_filter["2011_09_30_drive_0033_extract"] = [0, None]
        self.datasets_train_filter["2011_10_03_drive_0027_extract"] = [0, 18000]
        self.datasets_train_filter["2011_10_03_drive_0034_extract"] = [0, 31000]
        self.datasets_train_filter["2011_09_30_drive_0034_extract"] = [0, None]

        for dataset_fake in KITTIDataset.datasets_fake:
            if dataset_fake in self.datasets:
                self.datasets.remove(dataset_fake)
            if dataset_fake in self.datasets_train:
                self.datasets_train.remove(dataset_fake)

    @staticmethod
    def read_data(args):
        """
        Read the data from the KITTI dataset

        :param args:
        :return:
        """

        print("Start read_data")
        t_tot = 0  # sum of times for the all dataset
        date_dirs = os.listdir(args.path_data_base)
        for n_iter, date_dir in enumerate(date_dirs):
            # get access to each sequence
            path1 = os.path.join(args.path_data_base, date_dir)
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue
                # read data
                oxts_files = sorted(glob.glob(os.path.join(path2, 'oxts', 'data', '*.txt')))
                oxts = KITTIDataset.load_oxts_packets_and_poses(oxts_files)

                """ Note on difference between ground truth and oxts solution:
                    - orientation is the same
                    - north and east axis are inverted
                    - position are closed to but different
                    => oxts solution is not loaded
                """

                print("\n Sequence name : " + date_dir2)
                if len(oxts) < KITTIDataset.min_seq_dim:  #  sequence shorter than 30 s are rejected
                    cprint("Dataset is too short ({:.2f} s)".format(len(oxts) / 100), 'yellow')
                    continue
                lat_oxts = np.zeros(len(oxts))
                lon_oxts = np.zeros(len(oxts))
                alt_oxts = np.zeros(len(oxts))
                roll_oxts = np.zeros(len(oxts))
                pitch_oxts = np.zeros(len(oxts))
                yaw_oxts = np.zeros(len(oxts))
                roll_gt = np.zeros(len(oxts))
                pitch_gt = np.zeros(len(oxts))
                yaw_gt = np.zeros(len(oxts))
                t = KITTIDataset.load_timestamps(path2)
                acc = np.zeros((len(oxts), 3))
                acc_bis = np.zeros((len(oxts), 3))
                gyro = np.zeros((len(oxts), 3))
                gyro_bis = np.zeros((len(oxts), 3))
                p_gt = np.zeros((len(oxts), 3))
                v_gt = np.zeros((len(oxts), 3))
                v_rob_gt = np.zeros((len(oxts), 3))

                k_max = len(oxts)
                for k in range(k_max):
                    oxts_k = oxts[k]
                    t[k] = 3600 * t[k].hour + 60 * t[k].minute + t[k].second + t[
                        k].microsecond / 1e6
                    lat_oxts[k] = oxts_k[0].lat
                    lon_oxts[k] = oxts_k[0].lon
                    alt_oxts[k] = oxts_k[0].alt
                    acc[k, 0] = oxts_k[0].af
                    acc[k, 1] = oxts_k[0].al
                    acc[k, 2] = oxts_k[0].au
                    acc_bis[k, 0] = oxts_k[0].ax
                    acc_bis[k, 1] = oxts_k[0].ay
                    acc_bis[k, 2] = oxts_k[0].az
                    gyro[k, 0] = oxts_k[0].wf
                    gyro[k, 1] = oxts_k[0].wl
                    gyro[k, 2] = oxts_k[0].wu
                    gyro_bis[k, 0] = oxts_k[0].wx
                    gyro_bis[k, 1] = oxts_k[0].wy
                    gyro_bis[k, 2] = oxts_k[0].wz
                    roll_oxts[k] = oxts_k[0].roll
                    pitch_oxts[k] = oxts_k[0].pitch
                    yaw_oxts[k] = oxts_k[0].yaw
                    v_gt[k, 0] = oxts_k[0].ve
                    v_gt[k, 1] = oxts_k[0].vn
                    v_gt[k, 2] = oxts_k[0].vu
                    v_rob_gt[k, 0] = oxts_k[0].vf
                    v_rob_gt[k, 1] = oxts_k[0].vl
                    v_rob_gt[k, 2] = oxts_k[0].vu
                    p_gt[k] = oxts_k[1][:3, 3]
                    Rot_gt_k = oxts_k[1][:3, :3]
                    roll_gt[k], pitch_gt[k], yaw_gt[k] = IEKF.to_rpy(Rot_gt_k)

                t0 = t[0]
                t = np.array(t) - t[0]
                # some data can have gps out
                if np.max(t[:-1] - t[1:]) > 0.1:
                    cprint(date_dir2 + " has time problem", 'yellow')
                ang_gt = np.zeros((roll_gt.shape[0], 3))
                ang_gt[:, 0] = roll_gt
                ang_gt[:, 1] = pitch_gt
                ang_gt[:, 2] = yaw_gt

                p_oxts = lla2ned(lat_oxts, lon_oxts, alt_oxts, lat_oxts[0], lon_oxts[0],
                                 alt_oxts[0], latlon_unit='deg', alt_unit='m', model='wgs84')
                p_oxts[:, [0, 1]] = p_oxts[:, [1, 0]]  # see note

                # take correct imu measurements
                u = np.concatenate((gyro_bis, acc_bis), -1)
                # convert from numpy
                t = torch.from_numpy(t)
                p_gt = torch.from_numpy(p_gt)
                v_gt = torch.from_numpy(v_gt)
                ang_gt = torch.from_numpy(ang_gt)
                u = torch.from_numpy(u)

                # convert to float
                t = t.float()
                u = u.float()
                p_gt = p_gt.float()
                ang_gt = ang_gt.float()
                v_gt = v_gt.float()

                mondict = {
                    't': t, 'p_gt': p_gt, 'ang_gt': ang_gt, 'v_gt': v_gt,
                    'u': u, 'name': date_dir2, 't0': t0
                    }

                t_tot += t[-1] - t[0]
                KITTIDataset.dump(mondict, args.path_data_save, date_dir2)
        print("\n Total dataset duration : {:.2f} s".format(t_tot))

    @staticmethod
    def prune_unused_data(args):
        """
        Deleting image and velodyne
        Returns:

        """

        unused_list = ['image_00', 'image_01', 'image_02', 'image_03', 'velodyne_points']
        date_dirs = ['2011_09_28', '2011_09_29', '2011_09_30', '2011_10_03']

        for date_dir in date_dirs:
            path1 = os.path.join(args.path_data_base, date_dir)
            if not os.path.isdir(path1):
                continue
            date_dirs2 = os.listdir(path1)

            for date_dir2 in date_dirs2:
                path2 = os.path.join(path1, date_dir2)
                if not os.path.isdir(path2):
                    continue
                print(path2)
                for folder in unused_list:
                    path3 = os.path.join(path2, folder)
                    if os.path.isdir(path3):
                        print(path3)
                        shutil.rmtree(path3)

    @staticmethod
    def subselect_files(files, indices):
        try:
            files = [files[i] for i in indices]
        except:
            pass
        return files

    @staticmethod
    def rotx(t):
        """Rotation about the x-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    @staticmethod
    def roty(t):
        """Rotation about the y-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    @staticmethod
    def pose_from_oxts_packet(packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet.
        """
        er = 6378137.  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.
        ty = scale * er * np.log(np.tan((90. + packet.lat) * np.pi / 360.))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        Rx = KITTIDataset.rotx(packet.roll)
        Ry = KITTIDataset.roty(packet.pitch)
        Rz = KITTIDataset.rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t

    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transformation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

    @staticmethod
    def load_oxts_packets_and_poses(oxts_files):
        """Generator to read OXTS ground truth data.
           Poses are given in an East-North-Up coordinate system
           whose origin is the first GPS position.
        """
        # Scale for Mercator projection (from first lat value)
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        oxts = []

        for filename in oxts_files:
            with open(filename, 'r') as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]

                    packet = KITTIDataset.OxtsPacket(*line)

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.)

                    R, t = KITTIDataset.pose_from_oxts_packet(packet, scale)

                    if origin is None:
                        origin = t

                    T_w_imu = KITTIDataset.transform_from_rot_trans(R, t - origin)

                    oxts.append(KITTIDataset.OxtsData(packet, T_w_imu))
        return oxts

    @staticmethod
    def load_timestamps(data_path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'oxts', 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        return timestamps

    def load_timestamps_img(data_path):
        """Load timestamps from file."""
        timestamp_file = os.path.join(data_path, 'image_00', 'timestamps.txt')

        # Read and parse the timestamps
        timestamps = []
        with open(timestamp_file, 'r') as f:
            for line in f.readlines():
                # NB: datetime only supports microseconds, but KITTI timestamps
                # give nanoseconds, so need to truncate last 4 characters to
                # get rid of \n (counts as 1) and extra 3 digits
                t = datetime.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                timestamps.append(t)
        return timestamps

class KITTIArgs():
    path_data_base = "/media/mines/46230797-4d43-4860-9b76-ce35e699ea47/KITTI/raw"
    path_data_save = "data"
    path_results = "results"
    
    path_temp = "temp" # Change to specific path

    # training, cross-validation and test dataset
    cross_validation_sequences = ['2011_09_30_drive_0028_extract']
    test_sequences = []
    continue_training = False

    dataset_class = KITTIDataset
    parameter_class = KITTIParameters

# Snap a number to arbitrary grid size
def round_to_grid(x, grid_size):
    return np.round(x*(1/grid_size), decimals=0) * grid_size

class GlobalMap:
    # Map Semantic KITTI -> AI-IMU data
    scene_map = {
        '00': '2011_10_03_drive_0027_extract',
        '01': '2011_10_03_drive_0042_extract',
        '02': '2011_10_03_drive_0034_extract',
        '03': '2011_09_26_drive_0067_extract',
        '04': '2011_09_30_drive_0016_extract',
        '05': '2011_09_30_drive_0018_extract',
        '06': '2011_09_30_drive_0020_extract',
        '07': '2011_09_30_drive_0027_extract',
        '08': '2011_09_30_drive_0028_extract',
        '09': '2011_09_30_drive_0033_extract',
        '10': '2011_09_30_drive_0034_extract'
    }
    
    # Frames which semantic KITTI has data for
    # All are LIDAR frames, so IMU bounds will be imu_freq/lidar_freq times 
    # bigger. These bounds are off (not synchronized with LIDAR), 08 is the 
    # only one that is kind of corrected.
    frame_bounds = {
        '00': (0, 4540),
        '01': (0, 1100),
        '02': (0, 4660),
        '03': (0, 800),
        '04': (0, 270),
        '05': (0, 2760),
        '06': (0, 1100),
        '07': (0, 1100),
        '08': (1100, 5170),
        '09': (0, 1590),
        '10': (0, 1200)
    }
    
    imu_freq = 100  # IMU frequency in Hz
    lidar_freq = 10  # LIDAR frequency in Hz
    
    def __init__(self, test_scene='09', gt=False, grid_shape=200):
        self.test_scene = test_scene
        self.gt = gt  # Indicates whether it is ground truth or not
        self.grid_shape = grid_shape  # Local BEV maps are (grid_shape, grid_shape)
        self.grid_size = 0.2
        self.N_classes = 15
        self.filter_params = KITTIParameters()
        self.args = KITTIArgs()
        self.args.test_sequences.append(self.scene_map[test_scene])
        self.dataset = KITTIDataset(self.args)

        # Add normalization factors to prevent errors when u_loc is None
        self.dataset.normalize_factors = {
            "u_loc": torch.zeros(6).double(),
            "u_std": torch.ones(6).double()
        }

        t, R, p, ang_gt, p_gt = self.estimate_trajectory(self.args, self.dataset)
        self.trajectory = {
            't': t,
            'R': R,
            'p': p,
            'ang_gt': ang_gt,
            'p_gt': p_gt
        }
        
        # Manually synced indices for scene 08 only.
        # For other scenes, use trim_data() which has correct timespan but 
        # incorrect synchronization.
        if self.test_scene != '08':
            self.trim_data()
        else:
            self.get_closest_indices()

        self.heading = self.compute_bev_heading()
        self.initialize_global_map()
    
    def get_closest_indices(self):
        # Load manually synced indices for scene 08
        closest_idx_path = './closest_idx_08.txt'
        closest_idx_timestamp_path = open(closest_idx_path, 'r')
        closest_idx_timestamp = closest_idx_timestamp_path.readlines()

        self.closest_idx = np.zeros((len(closest_idx_timestamp)))
        c = 0
        for line in closest_idx_timestamp:
            self.closest_idx[c] = int(line)
            c += 1
    
    def initialize_global_map(self):
        # Pad array so indexing doesn't "leak"
        p_x, p_y = self.trajectory['p'][:, 0], self.trajectory['p'][:, 1]
        p_gt_x, p_gt_y = np.array(self.trajectory['p_gt'][:, 0]), np.array(
            self.trajectory['p_gt'][:, 1])

        # For certain scenes where AI-IMU estimate is bad, use the commented 
        # lines below.
        # min_x = np.minimum(np.min(p_x), np.min(p_gt_x)) - (self.grid_shape * 
        # self.grid_size)
        # max_x = np.maximum(np.max(p_x), np.max(p_gt_x)) + (self.grid_shape * 
        # self.grid_size)
        
        # min_y = np.minimum(np.min(p_y), np.min(p_gt_y)) - (self.grid_shape * 
        # self.grid_size)
        # max_y = np.maximum(np.max(p_y), np.max(p_gt_y)) + (self.grid_shape * 
        # self.grid_size)
        
        # For model evaluation on scene 08, the lines below ensure consistent 
        # output map size.
        min_x = np.min(p_gt_x) - (self.grid_shape * self.grid_size)
        max_x = np.max(p_gt_x) + (self.grid_shape * self.grid_size)

        min_y = np.min(p_gt_y) - (self.grid_shape * self.grid_size)
        max_y = np.max(p_gt_y) + (self.grid_shape * self.grid_size)
        
        # Pad by 100 just in case
        self.N_x = int((round_to_grid(max_x, self.grid_size) - round_to_grid(
            min_x, self.grid_size)) / self.grid_size) + 100
        self.N_y = int((round_to_grid(max_y, self.grid_size) - round_to_grid(
            min_y, self.grid_size)) / self.grid_size) + 100
        
        self.min_x = min_x
        self.min_y = min_y
        self.global_map = np.ones((self.N_x, self.N_y, self.N_classes)) * 1e-10
    
    def find_global_indices(self, x, y, heading):
        # Compute global indices for the given position and heading
        dx = np.arange(start=-self.grid_shape // 2, stop=self.grid_shape // 2, 
                       step=1) * self.grid_size
        dy = np.arange(start=-self.grid_shape // 2, stop=self.grid_shape // 2, 
                       step=1) * self.grid_size
        
        dxx, dyy = np.meshgrid(dx, dy, indexing='ij')
        
        du = dxx * np.cos(heading) - dyy * np.sin(heading)
        dv = dxx * np.sin(heading) + dyy * np.cos(heading)
        
        x_pos = x + du
        y_pos = y + dv
        
        x_ind = ((x_pos.flatten() - self.min_x) // self.grid_size).astype(int)
        y_ind = ((y_pos.flatten() - self.min_y) // self.grid_size).astype(int)
        
        return x_ind, y_ind
    
    def trim_data(self):
        # Trim data to match the frame bounds for the test scene
        (low_bound, upper_bound) = self.frame_bounds[self.test_scene]
        m = self.imu_freq // self.lidar_freq  # Multiple to use for bounds
        self.trajectory['t'] = self.trajectory['t'][low_bound * m:upper_bound * m]
        self.trajectory['R'] = self.trajectory['R'][low_bound * m:upper_bound * m, ...]
        self.trajectory['p'] = self.trajectory['p'][low_bound * m:upper_bound * m, ...]
        self.trajectory['ang_gt'] = self.trajectory['ang_gt'][low_bound * m:upper_bound * m, ...]
        self.trajectory['p_gt'] = self.trajectory['p_gt'][low_bound * m:upper_bound * m, ...]
        return
    
    def estimate_trajectory(self, args, dataset):
        # Estimate trajectory using IEKF and TORCHIEKF
        iekf = IEKF()
        torch_iekf = TORCHIEKF()

        # Set KITTI parameters
        iekf.filter_parameters = KITTIParameters()
        iekf.set_param_attr()
        torch_iekf.filter_parameters = KITTIParameters()
        torch_iekf.set_param_attr()

        torch_iekf.load(args, dataset)
        torch_iekf.get_normalize_u(dataset)  # Add this line
        iekf.set_learned_covariance(torch_iekf)

        dataset_name = self.args.test_sequences[0]

        print("Test filter on sequence: " + dataset_name)
        t, ang_gt, p_gt, v_gt, u = prepare_data(args, dataset, dataset_name, 0, to_numpy=True)
        N = None
        u_t = torch.from_numpy(u).double()
        measurements_covs = torch_iekf.forward_nets(u_t)
        measurements_covs = measurements_covs.detach().numpy()

        Rot, _, p, _, _, _, _ = iekf.run(t, u, measurements_covs, v_gt, p_gt, N, ang_gt[0])
        
        t, ang_gt, p_gt, _, _ = dataset.get_data(dataset_name)
        
        return t, Rot, p, ang_gt, p_gt

    def load_bev_labels(self, frame):
        # Load BEV labels for the given frame
        if not self.gt:
            # Path for MotionNet predicted labels (predicted version)
            root_path = '/scratch/eecs568s001w25_class_root/eecs568s001w25_class/yiweigui/bev-global-mapping/dataset/sequences/08/MotionNet_Prediciton_200x200'
            file_name = os.path.join(root_path, 'bev_labels', f'{frame:06d}.bin')
            # Load BEV labels
            print(f"[INFO] Loading BEV labels from {file_name}")
            labels = np.fromfile(file_name, dtype=np.uint8).reshape(256, 256)[28:228, 28:228]
            return labels
        else:
            # Path for Ground Truth labels (ground truth version)
            root_path = '/scratch/eecs568s001w25_class_root/eecs568s001w25_class/yiweigui/bev-global-mapping/MotionNet_Prediction/MotionNet_Prediction'
            file_name = os.path.join(root_path, f'scene_{self.test_scene}', 'bev_labels', f'{frame:06d}.bin')
        
            # Load BEV labels
            print(f"[INFO] Loading BEV labels from {file_name}")
            labels = np.fromfile(file_name, dtype=np.uint8).reshape(self.grid_shape, self.grid_shape)
            return labels

    def compute_bev_heading(self):
        # Compute BEV heading based on trajectory
        if self.gt:
            R = Rotation.from_euler('xyz', self.trajectory['ang_gt']).as_matrix()
        else:
            R = self.trajectory['R']
        e = np.array([1, 0, 0])  # Unit vector in y direction
        rotated_e = np.einsum('kij,j->ki', R, e)  # Should be (N_frames, 3)
        
        # Project onto xy plane and find angle
        x, y = rotated_e[:, 0], rotated_e[:, 1]
        theta = np.arctan2(y, x)
        return theta  # Radians
    
    def generate_global_map_parameters(self):
        # Generate global map parameters for the test scene
        (low_bound, upper_bound) = self.frame_bounds[self.test_scene]
        n_frames = upper_bound - low_bound
        for i in range((n_frames + 1)):
            labels = self.load_bev_labels(i).flatten()
            
            # Use manually synced indices for scene 08
            if self.test_scene != '08':
                imu_ind = i * (self.imu_freq // self.lidar_freq)
                if i == n_frames:
                    imu_ind -= 1
            else:
                imu_ind = int(self.closest_idx[i + low_bound])

            theta = self.heading[imu_ind]
            
            if self.gt:
                cur_pos = np.array(self.trajectory['p_gt'][imu_ind, :])
            else: 
                cur_pos = self.trajectory['p'][imu_ind, :]

            x, y = cur_pos[0], cur_pos[1]
            
            x_ind, y_ind = self.find_global_indices(x, y, theta)
            self.global_map[x_ind, y_ind, labels] += 1
        return

# Initialize global map object, involves computing predicted trajectory.
gm = GlobalMap(gt=True, test_scene='08')
gm.generate_global_map_parameters()

FIGSIZE = (8, 8)
DPI = 300

# Save ground truth mean image
labels = np.argmax(gm.global_map, axis=-1)
np.save('KITTI_Scene08_GT_Labels.npy', labels)
colored_map = LABEL_COLORS[labels.astype(np.uint8)]
plt.figure(figsize=FIGSIZE, dpi=DPI)
plt.imshow(colored_map)
plt.axis('off')
plt.savefig(f"scene_{gm.test_scene}_gt_mean.png", bbox_inches='tight')
print(f"[INFO] Ground truth image saved as scene_{gm.test_scene}_gt_mean.png")


# Initialize global map object, involves computing predicted trajectory.
gm = GlobalMap(gt=False, test_scene='08')
gm.generate_global_map_parameters()

# Save test mean and variance image
labels = np.argmax(gm.global_map, axis=-1)
np.save('KITTI_Scene08_Test_Labels.npy', labels)
colored_map = LABEL_COLORS[labels.astype(np.uint8)]
alpha_sum = np.sum(gm.global_map, axis=-1)
A = np.max(gm.global_map, axis=-1) / alpha_sum
variance = (A * (1 - A)) / (alpha_sum + 1)

# 保存 mean 图像
plt.figure(figsize=FIGSIZE, dpi=DPI)
plt.imshow(colored_map)
plt.axis('off')
plt.savefig(f"scene_{gm.test_scene}_test_mean.png", bbox_inches='tight')
plt.close()
print(f"[INFO] Test mean image saved as scene_{gm.test_scene}_test_mean.png")

# 保存 variance 图像
plt.figure(figsize=FIGSIZE, dpi=DPI)
plt.imshow(variance)
plt.axis('off')
plt.savefig(f"scene_{gm.test_scene}_test_variance.png", bbox_inches='tight')
plt.close()
print(f"[INFO] Test variance image saved as scene_{gm.test_scene}_test_variance.png")

# Save trajectory heading image
p_gt = gm.trajectory['p_gt']
theta = gm.heading
# 顺时针旋转90度：x' = y, y' = -x
rotated_p_gt = np.zeros_like(p_gt)
rotated_p_gt[:, 0] = p_gt[:, 1]
rotated_p_gt[:, 1] = -p_gt[:, 0]

# 旋转后的角度 theta：顺时针旋转相当于 -90°
rotated_theta = theta - np.pi / 2

# 绘图
plt.figure(figsize=FIGSIZE, dpi=DPI)
plt.plot(rotated_p_gt[:, 0], rotated_p_gt[:, 1])
for i in range(rotated_p_gt.shape[0]):
    if i % 300 == 0:
        plt.arrow(
            rotated_p_gt[i, 0],
            rotated_p_gt[i, 1],
            np.cos(rotated_theta[i]),
            np.sin(rotated_theta[i]),
            width=3,
            fc='r',
            ec='r'
        )
plt.axis('equal')
plt.axis('off')  
plt.savefig(f"scene_{gm.test_scene}_trajectory_heading_rotated.png", bbox_inches='tight')
print(f"[INFO] Rotated trajectory heading image saved as scene_{gm.test_scene}_trajectory_heading_rotated.png")
