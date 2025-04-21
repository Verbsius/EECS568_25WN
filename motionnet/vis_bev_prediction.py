import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import yaml

from PIL import Image as im
from tqdm import tqdm


eval_size = (256, 256, 15)

config_file = os.path.join(r'E:\EECS568\HW\Project\bev-global-mapping\motionnet\Data\semantic-kitti.yaml')
kitti_config = yaml.safe_load(open(config_file, 'r'))

LABELS_REMAP = kitti_config["learning_map"]
color_map = kitti_config["color_map"]

VISUALIZE = False
SAVE = True
TOP = True

LABEL_COLORS = np.array([
    (  255,   255,   255,),
    (245, 150, 100,),
    (245, 230, 100,),
    (150,  60,  30,),
    (180,  30,  80,),
    (250,  80, 100,),
    ( 30,  30, 255,),
    (200,  40, 255,),
    ( 90,  30, 150,),
    (255,   0, 255,),
    (255, 150, 255,),
    ( 75,   0,  75,),
    ( 75,   0, 175,),
    (  0, 200, 255,),
    ( 50, 120, 255,),
    (  0, 175,   0,),
    (  0,  60, 135,),
    ( 80, 240, 150,),
    (150, 240, 255,),
    (  0,   0, 255,),
]).astype(np.uint8)


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed


def form_bev(labels):
    bev_map = np.zeros((eval_size[0], eval_size[1]))

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


def bev_img(bev_map):
    colored_data = LABEL_COLORS[bev_map.astype(int)]
    img = im.fromarray(colored_data, 'RGB')
    return img


sequence_path = r"C:\Users\Ding Zhong\Downloads\data_odometry_labels\dataset\sequences"

for scene in sorted(os.listdir(sequence_path)):
    print(scene)
    if int(scene) != 8:
        continue
    sequence_path_scene = os.path.join(sequence_path, scene)
    file_path = os.path.join(sequence_path_scene, "labels")
    save_path = os.path.join(sequence_path_scene, "labels2")

    if VISUALIZE:
        for i in range(30, 300, 60):

            labels_file = os.path.join(file_path, str(i).zfill(6) + ".bin")

            #labels = np.fromfile(labels_file,dtype=np.uint16).reshape(eval_size)
            labels = np.fromfile(labels_file,dtype=np.uint32).reshape(eval_size)

            bev_map = np.argmax(labels,axis=-1)
            img = bev_img(bev_map)
            img.show()
            path_save = scene + "_" + str(i) + ".png"
            img.save(path_save)

            
    if SAVE:
        bev_path = os.path.join(sequence_path_scene, "bev_gt")

        if not os.path.exists(bev_path):
            os.makedirs(bev_path)

        total_files = int(len(os.listdir(file_path))/4)
        for i in tqdm(range(total_files)):
            labels_file = os.path.join(file_path, str(i).zfill(6) + ".label")

            labels = np.fromfile(labels_file,dtype=np.uint16).reshape(eval_size)

            bev_map = form_bev(labels)
            bev_file = os.path.join(bev_path, str(i).zfill(6) + ".bin")
            bev_map.tofile(bev_file)
            



    
    
    