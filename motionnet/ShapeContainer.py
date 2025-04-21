import os
import numpy as np

"""
    Generic Volumetric container
"""
class ShapeContainer:
    def __init__(self, grid_size,
        min_bound=np.array([0, -1.0*np.pi, 0], dtype=np.float32),
        max_bound=np.array([20, 1.0*np.pi, 10], dtype=np.float32),
        num_channels=25,
        coordinates="cylindrical"):
        """
        Constructor that creates the cylinder volume container

        :param grid_size: 1x3 np array that represents the number of cells in each dimension
        :param max_bound: [max in 3 dimensions]
        :param min_bound: [min in 3 dimensions]
        :param num_channels: number of semantic channels
        """
        self.coordinates = coordinates
        
        self.grid_size = grid_size
        self.num_classes = num_channels
        self.max_bound = max_bound
        self.min_bound = min_bound

        # Class variables to be set by reset_grid
        self.intervals = None
        self.voxels = None

        self.reset_grid()

    def reset_grid(self):
        """
        Recomputes voxel grid and intializes all values to 0

        Condition:  Requires that grid_size, max_bound, and min_bound be set prior to 
                    calling function
        """
        crop_range = self.max_bound - self.min_bound
        self.intervals = crop_range / self.grid_size

        if (self.intervals == 0).any(): 
            print("Error zero interval detected...")
            return
        # Initialize voxel grid with float32
        self.voxels = np.zeros(list(self.grid_size.astype(np.uint32)) + [self.num_classes])

        # print("Initialized voxel grid with {num_cells} cells".format(
        #     num_cells=np.prod(self.grid_size)))


    def __len__(self):
        return self.grid_size

    def get_voxels(self):
        """
        Returns an instance of the voxels grid
        """
        return self.voxels

    def __getitem__(self, input_xyzl):
        """
        Returns the voxel centroid that the cartesian coordinate falls in

        :param input_xyzl:  nx4 np array where rows are points and cols are x,y,z
                            and last col is semantic label idx
        :return: nx1 np array where rows are points and col is value at each point
        """
        
        # Reshape coordinates for 2d indexing
        input_idxl = self.grid_ind(input_xyzl).astype(int)

        return self.voxels[ list(input_idxl[:, 0]),
                            list(input_idxl[:, 1]),
                            list(input_idxl[:, 2]),
                            list(input_idxl[:, 3])
                        ]

    def __setitem__(self, input_xyzl, input_value):
        """
        Sets the voxel to the input cell (cylindrical coordinates)

        :param input_xyzl:  nx4 np array where rows are points and cols are x,y,z 
                            and last col is semantic label idx
        :param input_value: scalar value for how much to increment cell by
        """
        # Reshape coordinates for 2d indexing
        input_idxl = self.grid_ind(input_xyzl).astype(int)

        self.voxels[input_idxl[:,0],
                    input_idxl[:, 1],
                    input_idxl[:, 2],
                    input_idxl[:, 3]] = input_value


    def grid_ind(self, input_xyzl):
        """
        Returns index of each cartesian coordinate in grid

        :param input_xyz:   nx4 np array where rows are points and cols are x,y,z 
                            and last col is semantic label idx
        :return:    nx4 np array where rows are points and cols are x,y,z 
                    and last col is semantic label idx
        """
        input_xyzl  = input_xyzl.reshape(-1, 4)
        input_xyz   = input_xyzl[:, 0:3]
        labels      = input_xyzl[:, 3].reshape(-1, 1)
        
        xyz_pol = self.cart2grid(input_xyz)

        valid_input_mask= np.all(
            (xyz_pol < self.max_bound) & (xyz_pol >= self.min_bound), axis=1)
        valid_xyz_pol   = xyz_pol[valid_input_mask]
        valid_labels    = labels[valid_input_mask]

        grid_ind = (np.floor((valid_xyz_pol
                    - self.min_bound) / self.intervals)).astype(np.int)
        # Clip due to edge cases
        maxes = np.reshape(self.grid_size - 1, (1, 3))
        mins = np.zeros_like(maxes)
        grid_ind = np.clip(grid_ind, mins, maxes)
        grid_ind = np.hstack( (grid_ind, valid_labels) )
        return grid_ind

    def get_voxel_centers(self, input_xyzl):
        """
        Return voxel centers corresponding to each input xyz cartesian coordinate

        :param input_xyzl:  nx4 np array where rows are points and cols are x,y,z 
                            and last col is semantic label idx

        :return:    nx4 np array where rows are points and cols are x,y,z 
                    and last col is semantic label idx
        """
        
        # Center data on each voxel centroid for cylindrical coordinate PTnet
        valid_idxl  = self.grid_ind(input_xyzl)
        valid_idx   = valid_idxl[:, 0:3]
        valid_labels= valid_idxl[:, 3].reshape(-1, 1)

        valid_idx  = ( (valid_idx+0.5) * self.intervals ) + self.min_bound
        voxel_centers = np.hstack( (valid_idx, valid_labels))

        return self.grid2cart(voxel_centers)

    def cart2grid(self, input_xyz):
        """
        Converts cartesian to grid's coordinates system

        :param input_xyz_polar: nx3 or 4 np array where rows are points and cols are x,y,z 
                                and last col is semantic label idx

        :return:    size of input np array where rows are points and cols are r,theta,z, 
                    label (optional)
        """
        if self.coordinates == "cylindrical":
            rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2).reshape(-1, 1)
            phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0]).reshape(-1, 1)

            return np.hstack((rho, phi, input_xyz[:, 2:]))
        else:
            return input_xyz


    def grid2cart(self, input_xyz_polar):
        """
        Converts grid to cartesian coordinates

        :param input_xyz_polar: nx3 or 4 np array where rows are points and cols 
                                are r,theta,z

        :return:    nx3 or 4 np array where rows are points and cols are 
                    x,y,z,label (optional)
        """
        if self.coordinates == "cylindrical":
            x = (input_xyz_polar[:, 0] * np.cos(input_xyz_polar[:, 1])).reshape(-1, 1)
            y = (input_xyz_polar[:, 0] * np.sin(input_xyz_polar[:, 1])).reshape(-1, 1)

            return np.hstack((x, y, input_xyz_polar[:, 2:]))
        else:
            return input_xyz_polar


    def accumulate_points(self, input_xyz, input_label):
        """
        一次性把 (N,3) 的点和 (N,) 的标签累加到 self.voxels。
        self.voxels.shape = (X, Y, Z, num_classes)。
        最后一个维度是类别计数。
        """

        # 如果是柱坐标系，需要先把输入的 (x,y,z) 转到 (rho,phi,z)
        if self.coordinates == "cylindrical":
            rho = np.sqrt(input_xyz[:, 0]**2 + input_xyz[:, 1]**2)
            phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
            xyz_pol = np.stack([rho, phi, input_xyz[:, 2]], axis=-1)
        else:
            # 如果是 cartesian，就直接拿过来
            xyz_pol = input_xyz

        # 1) 先做边界裁剪
        valid_mask = np.all(
            (xyz_pol >= self.min_bound) & (xyz_pol < self.max_bound),
            axis=1
        )
        xyz_pol = xyz_pol[valid_mask]
        labs = input_label[valid_mask].astype(np.int32)

        # 2) 计算所在体素的整数索引 (x_idx, y_idx, z_idx)
        #    intervals = (max_bound - min_bound) / grid_size
        #    grid_idx = floor((xyz_pol - min_bound)/intervals)
        grid_idx = ((xyz_pol - self.min_bound) / self.intervals)
        grid_idx = np.floor(grid_idx).astype(np.int32)

        # clip 避免四舍五入后越界
        grid_idx = np.clip(grid_idx, 0, self.grid_size.astype(int) - 1)

        # 3) 将 (x_idx, y_idx, z_idx) 展平成 flat 索引
        X, Y, Z = self.grid_size.astype(int)
        flat_idx = grid_idx[:, 0] + grid_idx[:, 1] * X + grid_idx[:, 2] * X * Y

        # 4) 用 np.add.at 在 (X*Y*Z, num_classes) 上累加
        #    self.voxels.shape = (X, Y, Z, num_classes)
        #    先 reshape 成 (X*Y*Z, num_classes)
        voxels_2d = self.voxels.reshape(-1, self.num_classes)
        np.add.at(voxels_2d, (flat_idx, labs), 1)

    def get_voxel_labels(self):
        """
        对 self.voxels 的最后一个维度做 argmax，返回 (X, Y, Z) 的类别ID。
        """
        return np.argmax(self.voxels, axis=3).astype(np.uint8)