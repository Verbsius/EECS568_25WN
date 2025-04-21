import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import yaml

config_file = r"E:\EECS568\HW\Project\bev-global-mapping\motionnet\Data\semantic-kitti.yaml"
kitti_config = yaml.safe_load(open(config_file, 'r'))
remapdict = kitti_config["learning_map"]

LABELS_REMAP = kitti_config["learning_map"]
LABEL_INV_REMAP = kitti_config["learning_map_inv"]
LABEL_COLORS = np.array([
    (  255,   255,   255,), # unlabled
    (245, 150, 100,), #car
    (245, 230, 100,), #bike
    ( 30,  30, 255,), #person
    (255,   0, 255,), #road
    (255, 150, 255,), #parking
    ( 75,   0,  75,), #sidewalk
    (  0, 200, 255,), #building
    ( 50, 120, 255,), #fence
    (  0, 175,   0,), #vegetation
    ( 80, 240, 150,), #terrain
    (150, 240, 255,), #pole
    (90,  30, 150,), #traffic-sign
    (255,  0,  0,), #moving-car
    ( 0,  0, 255,), #moving-person
]).astype(np.uint8)
pc_path = r"C:\Users\Ding Zhong\Downloads\data_odometry_velodyne\dataset\sequences\08\velodyne\000000.bin"
label_path = r"C:\Users\Ding Zhong\Downloads\data_odometry_labels\dataset\sequences\08\labels\000000.label"
label_ref_path = r"E:\EECS568\HW\Project\bev-global-mapping\MotionNet_Prediction\MotionNet_Prediction\scene_08\bev_labels\000000.bin"

class test_BEV:
    def __init__(self,
        pc_path = pc_path,
        label_path = label_path,
        label_ref_path = label_ref_path,
        device='cuda',
        num_frames=4,
        past_frames=10,
        split='train',
        transform_pose=False,
        get_gt = True,
        grid_size = [200,200,32],
        grid_size_train = [256,256,32]
        ):
        self.get_gt = get_gt
        self._grid_size = grid_size
        self.grid_dims = np.asarray(self._grid_size)
        self._grid_dims_train = np.asarray(grid_size_train)
        self._eval_size = list(np.uint32(self._grid_size))
        self.coor_ranges = [-25.6,-25.6,-6.4] + [25.6,25.6,6.4]
        self.voxel_sizes = [abs(self.coor_ranges[3] - self.coor_ranges[0]) / self._grid_dims_train[0], 
                      abs(self.coor_ranges[4] - self.coor_ranges[1]) / self._grid_dims_train[1],
                      abs(self.coor_ranges[5] - self.coor_ranges[2]) / self._grid_dims_train[2]]
        self.min_bound = np.asarray(self.coor_ranges[:3])
        self.max_bound = np.asarray(self.coor_ranges[3:])
        self.voxel_sizes = np.asarray(self.voxel_sizes)

        self._num_frames = num_frames
        self._past_frames = past_frames
        self.device = device
        self.split = split
        self.transform_pose = transform_pose

        self._velodyne_list = []
        self._frames_list = []
        self._bev_labels = []
        self._poses = np.empty((0,12))
        self._Tr = np.empty((0,12))

        self._num_frames_scene = []

        self.pc_path = pc_path
        self.label_path = label_path
        self.label_ref_path = label_ref_path

    def get_pose(self, idx):
        pose = np.zeros((4, 4))
        pose[3, 3] = 1
        pose[:3, :4] = self._poses[idx,:].reshape(3, 4)

        Tr = np.zeros((4, 4))
        Tr[3, 3] = 1
        Tr[:3, :4] = self._Tr[idx,:].reshape(3,4)

        Tr = Tr.astype(np.float32)
        pose = pose.astype(np.float32)
        global_pose = np.matmul(np.linalg.inv(Tr), np.matmul(pose, Tr))

    def points_to_voxels(self, voxel_grid, label_grid, points, labels, t_i):
        # Valid voxels (make sure to clip)
        valid_point_mask= np.all(
            (points < self.max_bound) & (points >= self.min_bound), axis=1)
        valid_points = points[valid_point_mask, :]
        voxels = np.floor((valid_points - self.min_bound) / self.voxel_sizes).astype(int)
        # Clamp to account for any floating point errors
        maxes = np.reshape(self._grid_dims_train - 1, (1, 3))
        mins = np.zeros_like(maxes)
        voxels = np.clip(voxels, mins, maxes).astype(int)
        
        voxel_grid[t_i, voxels[:, 0], voxels[:, 1], voxels[:, 2]] += 1
        label_grid[t_i, voxels[:, 0], voxels[:, 1], voxels[:, 2]] = labels[valid_point_mask]
        return voxel_grid, label_grid

    def form_bev(self, labels):
        bev_map = np.zeros((256, 256))

        bev_mask = np.sum(labels,axis=-1) != 0
        bev_x,bev_y = np.where(bev_mask)

        for i in range(bev_x.shape[0]):
            x = bev_x[i]
            y = bev_y[i]

            lables_column = labels[x,y]
            
            mask_zero = lables_column != 0
            mask = np.sum(mask_zero) >= 1
            
            if mask:
                indx = np.where(mask_zero)[0][-1]
                bev_map[x, y] = labels[x,y,indx]

        return bev_map.astype(np.uint8)


    def get_bev(self):
        # pdb.set_trace()
        points = np.fromfile(self.pc_path, dtype=np.float32).reshape(-1, 4)[:, :3]  # (x, y, z, intensity)
        print(points.shape)
        labels = np.fromfile(self.label_path, dtype=np.uint32)
        labels_ref = np.fromfile(self.label_ref_path, dtype=np.uint8).reshape(200,200)

        semantic_labels = labels & 0xFFFF  # 语义类别（去掉 instance ID）
        # map by learning_map
        mapped_labels = np.vectorize(LABELS_REMAP.get)(semantic_labels)

        # pdb.set_trace()
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        current_horizon = np.zeros((1, int(self._grid_dims_train[0]), int(self._grid_dims_train[1]), int(self._grid_dims_train[2])), dtype=np.float32)
        current_horizon_label = np.zeros((1, int(self._grid_dims_train[0]), int(self._grid_dims_train[1]), int(self._grid_dims_train[2])), dtype=np.float32)
        t_i = 0
        current_horizon, current_horizon_label = self.points_to_voxels(current_horizon, current_horizon_label, points, mapped_labels, t_i)
        # pdb.set_trace()
        bev_label = self.form_bev(current_horizon_label[0])
        #plot bev_label
        # pdb.set_trace()
        return current_horizon, bev_label, labels_ref
    
from scipy.ndimage import generic_filter

import numpy as np
from scipy.ndimage import distance_transform_edt

def fill_missing_labels(bev_label):
    """
    使用最近邻填充 BEV 语义标签，确保不会填充成 0
    :param bev_label: (H, W) numpy 数组，BEV 语义标签
    :return: 填充后的 BEV
    """
    # 先创建一个 mask，标记哪些点是 0（ignored label）
    mask = bev_label == 0  

    if np.all(mask):  # 处理极端情况，整个 BEV 全是 0
        print("WARNING: BEV 全是 0，填充可能无效")
        return bev_label
    
    # 计算距离变换（找到每个 0 像素到最近的非 0 像素的距离）
    dist, nearest_idx = distance_transform_edt(mask, return_indices=True)

    # 通过 nearest_idx 找到最近的非 0 类别
    filled_bev = bev_label[tuple(nearest_idx)]

    return filled_bev


def visualize_bev(bev_label, labels_ref, filled_bev, LABEL_COLORS):
    # 创建一个包含3个子图的图像
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # 创建一个函数来生成RGB图像
    def create_rgb_image(label_map):
        rgb = np.zeros((label_map.shape[0], label_map.shape[1], 3), dtype=np.uint8)
        for label in range(len(LABEL_COLORS)):
            mask = label_map == label
            rgb[mask] = LABEL_COLORS[label]
        return rgb
    
    # 生成三个RGB图像
    bev_rgb = create_rgb_image(bev_label)
    ref_rgb = create_rgb_image(labels_ref)
    filled_rgb = create_rgb_image(filled_bev)
    
    # 显示三个图像
    axes[0].imshow(bev_rgb)
    axes[0].set_title('Original BEV')
    axes[0].axis('off')
    
    axes[1].imshow(ref_rgb)
    axes[1].set_title('Reference BEV')
    axes[1].axis('off')
    
    axes[2].imshow(filled_rgb)
    axes[2].set_title('Filled BEV')
    axes[2].axis('off')
    
    # 调整子图之间的间距
    plt.tight_layout()
    plt.show()

test = test_BEV()
bev, bev_label, labels_ref = test.get_bev()

print(bev.shape)
print(bev_label.shape)
# pdb.set_trace()
print(np.unique(bev_label))

# 填充空白区域
filled_bev = fill_missing_labels(bev_label)
# pdb.set_trace()

# 可视化所有BEV标签
visualize_bev(bev_label, labels_ref, filled_bev, LABEL_COLORS)