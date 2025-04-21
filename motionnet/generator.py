import os
import numpy as np
import yaml
from PIL import Image as im
from tqdm import tqdm

# =============================================================================
#  1) 定义 ShapeContainer 类（和原始一致）
# =============================================================================
class ShapeContainer:
    def __init__(self, grid_size,
        min_bound=np.array([0, -1.0*np.pi, 0], dtype=np.float32),
        max_bound=np.array([20, 1.0*np.pi, 10], dtype=np.float32),
        num_channels=25,
        coordinates="cylindrical"):
        """
        Constructor that creates the cylinder volume container
        """
        self.coordinates = coordinates
        self.grid_size = grid_size
        self.num_classes = num_channels
        self.max_bound = max_bound
        self.min_bound = min_bound
        self.intervals = None
        self.voxels = None

        self.reset_grid()

    def reset_grid(self):
        """
        Recomputes voxel grid and intializes all values to 0
        """
        crop_range = self.max_bound - self.min_bound
        self.intervals = crop_range / self.grid_size
        if (self.intervals == 0).any(): 
            print("Error zero interval detected...")
            return
        self.voxels = np.zeros(list(self.grid_size.astype(np.uint32)) + [self.num_classes],
                               dtype=np.float32)

    def get_voxels(self):
        """Returns the volumetric grid."""
        return self.voxels

    def accumulate_points(self, input_xyz, input_label):
        """
        一次性把 (N,3) 的点和 (N,) 的标签累加到 self.voxels。
        self.voxels.shape = (X, Y, Z, num_classes)。
        """
        if self.coordinates == "cylindrical":
            rho = np.sqrt(input_xyz[:, 0]**2 + input_xyz[:, 1]**2)
            phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
            xyz_pol = np.stack([rho, phi, input_xyz[:, 2]], axis=-1)
        else:
            xyz_pol = input_xyz

        # 1) 边界裁剪
        valid_mask = np.all(
            (xyz_pol >= self.min_bound) & (xyz_pol < self.max_bound),
            axis=1
        )
        xyz_pol = xyz_pol[valid_mask]
        labs = input_label[valid_mask].astype(np.int32)

        # 2) 计算所在体素索引 (x_idx, y_idx, z_idx)
        grid_idx = (xyz_pol - self.min_bound) / self.intervals
        grid_idx = np.floor(grid_idx).astype(np.int32)
        grid_idx = np.clip(grid_idx, 0, self.grid_size.astype(int) - 1)

        # 3) 将 (x_idx, y_idx, z_idx) 展平成 flat 索引
        X, Y, Z = self.grid_size.astype(int)
        flat_idx = grid_idx[:, 0] + grid_idx[:, 1] * X + grid_idx[:, 2] * X * Y

        # 4) 用 np.add.at 在 (X*Y*Z, num_classes) 上累加
        voxels_2d = self.voxels.reshape(-1, self.num_classes)
        np.add.at(voxels_2d, (flat_idx, labs), 1)

    def get_voxel_labels(self):
        """
        对 self.voxels 的最后一个维度做 argmax，返回 (X, Y, Z) 的类别ID。
        """
        return np.argmax(self.voxels, axis=3).astype(np.uint8)

# =============================================================================
#  2) 一些辅助函数
# =============================================================================

def find_horizon(idx, past_frames=50, future_frames=50, total=100):
    """
    计算前后帧索引
    """
    idx_past = np.arange(idx - past_frames, idx) + 1
    idx_past = idx_past[idx_past >= 0]  # 只保留合法索引

    idx_future = np.arange(idx, idx + future_frames) + 1
    idx_future = idx_future[idx_future < total]  # 只保留合法索引

    idx_total = np.hstack((idx_past, idx_future))
    return idx_total

def initialize_grid(grid_size=np.array([256, 256, 32]),
                    min_bound=np.array([-25.6, -25.6, -2], dtype=np.float32),
                    max_bound=np.array([25.6, 25.6,  4.4], dtype=np.float32),
                    num_channels=15,
                    coordinates="cartesian"):
    """
    初始化 ShapeContainer
    """
    return ShapeContainer(grid_size, min_bound, max_bound, num_channels, coordinates)

def get_pose(idx, poses, Tr_s):
    """
    把语义KITTI给出的 pose.txt 中每一行的3x4位姿与 calib.txt 的Tr 相乘
    得到修正后的全局位姿(4x4)。
    """
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :4] = poses[idx, :].reshape(3, 4)

    Tr = np.eye(4, dtype=np.float32)
    Tr[:3, :4] = Tr_s[idx, :].reshape(3, 4)

    global_pose = np.linalg.inv(Tr) @ (pose @ Tr)
    return global_pose

def form_bev(voxel_labels):
    """
    voxel_labels.shape = (X, Y, Z)
    生成一个简单的 BEV label (X, Y)，取最高的那层（Z越大越高）。
    """
    Xdim, Ydim, Zdim = voxel_labels.shape
    bev_map = np.zeros((Xdim, Ydim), dtype=np.uint8)
    # 找出在每个 (x, y) 处，最高的不为 0 的类别
    # 方法：从上往下（z大到z小）遍历，选最后一个非0
    # 也可以从下往上找第一个非0。这里给出一种直接做法：

    # (a) 先找非零掩码
    nonzero_mask = (voxel_labels != 0)
    # (b) 逐个位置找最后一层非零的索引
    #     由于 Z 维度往后，因此可以把 Zdim-1 当成“最高”
    #     这里的简单方法是从上到下遍历，也可以用argmax/argwhere等加快
    for x in range(Xdim):
        yz_slice = voxel_labels[x]  # shape = (Ydim, Zdim)
        mask_slice = nonzero_mask[x]
        # yz_slice[y,z] != 0 => mask_slice[y,z] = True
        # 找到 z 里最后一个 True 的位置
        # 简单写法：np.where(mask_slice[y]) => 所有 z
        # 取最后一个 z
        for y in range(Ydim):
            z_nonzero = np.where(mask_slice[y])[0]
            if z_nonzero.size > 0:
                bev_map[x, y] = yz_slice[y, z_nonzero[-1]]

    return bev_map

def bev_img(bev_map, label_colors):
    """
    将 (X, Y) 的 int label 转成彩色图像。
    label_colors 是一个 (num_classes, 3) 的数组。
    """
    colored_data = label_colors[bev_map]
    img = im.fromarray(colored_data, 'RGB')
    return img

# =============================================================================
#  3) 主逻辑：对语义KITTI数据进行加载、拼接、体素化，并输出BEV
# =============================================================================

def main():
    # 你可以根据需要自行改动
    PAST_FRAME = 50  # 包含当前帧之前
    FUTURE_FRAME = 50  # 当前帧之后
    grid_dims = np.array([256, 256, 64])
    min_bound = np.array([-25.6, -25.6, -6.4], dtype=np.float32)
    max_bound = np.array([25.6, 25.6,  6.4], dtype=np.float32)

    # 语义 KITTI 中自己定义的重映射表 (根据你的yaml自行调整)
    config_file = r"E:\EECS568\HW\Project\bev-global-mapping\motionnet\Data\semantic-kitti.yaml"
    kitti_config = yaml.safe_load(open(config_file, 'r'))
    LABELS_REMAP_TEMP = kitti_config["LABELS_REMAP_TEMP"]

    # 颜色表(根据你的需求自行调整/增减)
    LABEL_COLORS = np.array([
        (255, 255, 255),   # unlabled or 0
        (245, 150, 100),   # car
        (245, 230, 100),   # bike
        ( 30,  30, 255),   # person
        (255,   0, 255),   # road
        (255, 150, 255),   # parking
        ( 75,   0,  75),   # sidewalk
        (  0, 200, 255),   # building
        ( 50, 120, 255),   # fence
        (  0, 175,   0),   # vegetation
        ( 80, 240, 150),   # terrain
        (150, 240, 255),   # pole
        ( 90,  30, 150),   # traffic-sign
        (255,   0,   0),   # moving-car
        (  0,   0, 255),   # moving-person
    ], dtype=np.uint8)

    # 数据集根目录
    sequence_path = r"C:\Users\Ding Zhong\Downloads\data_odometry_velodyne\dataset\sequences"

    # 是否保存体素结果（.label）
    SAVE_VOXELS = False

    for scene in sorted(os.listdir(sequence_path)):
        # 例如跳过 scene <= 1
        if int(scene) <= 2:
            continue

        print("Processing scene:", scene)
        scene_path = os.path.join(sequence_path, scene)
        points_path = os.path.join(scene_path, 'velodyne')
        labels_path = os.path.join(scene_path, 'labels')
        poses_path = os.path.join(scene_path, "poses.txt")
        tr_path = os.path.join(scene_path, "calib.txt")

        bev_gt_path = os.path.join(scene_path, 'bev_gt2')
        bev_img_path = os.path.join(scene_path, 'bev_img2')

        if not os.path.exists(bev_gt_path):
            os.makedirs(bev_gt_path)
        if not os.path.exists(bev_img_path):
            os.makedirs(bev_img_path)

        if SAVE_VOXELS:
            voxel_path = os.path.join(scene_path, 'voxel_new')
            if not os.path.exists(voxel_path):
                os.makedirs(voxel_path)

        # 读取位姿信息
        poses = np.loadtxt(poses_path)  # shape = [N, 12]
        # calib.txt 中末行是 Tr: shape (3,4) => 我们只取最后一行
        # 并将其重复N次，这里的做法和原代码一致
        Tr_calib = np.genfromtxt(tr_path)
        # 注意：Tr_calib 里真正用到的是最后一行(3x4), 可能索引方式需根据calib文件结构
        # 这里原文只取了 [-1, 1:]
        Tr_raw = Tr_calib[-1, 1:]
        Tr_raw = np.repeat(np.expand_dims(Tr_raw, axis=0), poses.shape[0], axis=0)

        # 预先计算每一帧的全局位姿
        # get_pose() 返回的是 4x4 矩阵
        poses_transformed = []
        for i in range(poses.shape[0]):
            poses_transformed.append(get_pose(i, poses, Tr_raw))
        poses_transformed = np.array(poses_transformed)  # (N,4,4)

        # 再预先算出每帧位姿的逆
        poses_inv = np.linalg.inv(poses_transformed)     # (N,4,4)

        # 读取所有激光 & 标签到内存
        labels_total_list = []
        points_total_list = []

        print("Loading points & labels to memory...")
        label_files = sorted(os.listdir(labels_path))
        for i in tqdm(range(len(label_files))):
            lbl_file = os.path.join(labels_path, f"{i:06d}.label")
            pts_file = os.path.join(points_path, f"{i:06d}.bin")

            labels = np.fromfile(lbl_file, dtype=np.uint32) & 0xFFFF
            # 重映射
            for k in range(labels.shape[0]):
                labels[k] = LABELS_REMAP_TEMP[labels[k]]
            labels = labels.astype(np.uint8)

            points = np.fromfile(pts_file, dtype=np.float32).reshape(-1, 4)[:, :3]

            labels_total_list.append(labels)
            points_total_list.append(points)

        # 初始化体素网格
        voxel_grid = initialize_grid(
            grid_size=grid_dims,
            min_bound=min_bound,
            max_bound=max_bound,
            num_channels=LABEL_COLORS.shape[0],  # 分类数
            coordinates="cartesian"
        )

        print("Processing each frame ...")
        n_frames = len(label_files)
        for i in tqdm(range(n_frames)):
            # 1) 重置网格
            voxel_grid.reset_grid()

            # 2) 找到当前帧前后范围
            idx_list = find_horizon(i, PAST_FRAME, FUTURE_FRAME, n_frames)

            # 3) 计算当前帧的逆位姿
            inv_pose_i = poses_inv[i]

            # 4) 整合所有点到一个 array 做一次性累加
            all_points = []
            all_labels = []
            for j in idx_list:
                labels_j = labels_total_list[j]
                points_j = points_total_list[j]

                # 过滤掉 label=0（可选，看你需不需要累加无类别点）
                nz_mask = (labels_j != 0)
                labels_j = labels_j[nz_mask]
                points_j = points_j[nz_mask]

                # 做坐标变换：把 j 帧的点云变到 i 帧坐标系
                # relative_pose = inv_pose_i @ poses_transformed[j]
                # new_xyz = R * xyz + t
                relative_pose = inv_pose_i @ poses_transformed[j]
                R = relative_pose[:3, :3]
                t = relative_pose[:3, 3]
                transformed_points = (R @ points_j.T).T + t

                all_points.append(transformed_points)
                all_labels.append(labels_j)

            # 合并
            if len(all_points) > 0:
                all_points = np.concatenate(all_points, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                # 5) 体素化
                voxel_grid.accumulate_points(all_points, all_labels)

            # 6) 取 argmax
            voxel_labels = voxel_grid.get_voxel_labels()  # shape=(X,Y,Z)

            # 7) 生成 BEV
            bev_map = form_bev(voxel_labels)              # shape=(X,Y)
            img = bev_img(bev_map, LABEL_COLORS)

            # 8) 保存
            #   体素标签（可选）
            if SAVE_VOXELS:
                voxel_file = os.path.join(voxel_path, f"{i:06d}.label")
                voxel_labels.tofile(voxel_file)

            #   BEV 的二进制
            bev_file = os.path.join(bev_gt_path, f"{i:06d}.bin")
            bev_map.tofile(bev_file)

            #   BEV 的可视化图
            bev_img_file = os.path.join(bev_img_path, f"{i:06d}.png")
            img.save(bev_img_file)

        print(f"Scene {scene} done.")
    print("All scenes processed.")


if __name__ == "__main__":
    main()
