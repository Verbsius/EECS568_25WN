import sys

from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import yaml
from Data.kitti_dataset import *
from MotionNet.Model import *
from utils import *
from tqdm import tqdm
from compute_iou import fast_hist, per_class_iu
import pdb
MODEL_CONFIG = "MotionNet"
DATA_CONFIG = "kitti"
CLASS_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]

model_params_file = "/scratch/eecs568s001w25_class_root/eecs568s001w25_class/yiweigui/bev-global-mapping/motionnet/Configs/MotionNet.yaml"
print(model_params_file)
with open(model_params_file, "r") as stream:
    try:
        model_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

data_params_file = "/scratch/eecs568s001w25_class_root/eecs568s001w25_class/yiweigui/bev-global-mapping/motionnet/Configs/kitti.yaml"
with open(data_params_file, "r") as stream:
    try:
        data_params = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

T = model_params["T"]
past_frames = model_params["past_frames"]
train_dir = data_params["train_dir"]
val_dir = data_params["val_dir"]
test_dir = data_params["test_dir"]
num_workers = data_params["num_workers"]
remap = data_params["remap"]
voxelize_input = model_params["voxelize_input"]
binary_counts = model_params["binary_counts"]
transform_pose = model_params["transform_pose"]
seed = model_params["seed"]
B = model_params["B"]
BETA1 = model_params["BETA1"]
BETA2 = model_params["BETA2"]
decayRate = model_params["DECAY"]
lr = model_params["lr"]
epoch_num = model_params["epoch_num"]

coor_ranges = data_params['min_bound'] + data_params['max_bound']
voxel_sizes = [abs(coor_ranges[3] - coor_ranges[0]) / data_params["x_dim"],
              abs(coor_ranges[4] - coor_ranges[1]) / data_params["y_dim"],
              abs(coor_ranges[5] - coor_ranges[2]) / data_params["z_dim"]]

model = MotionNet(height_feat_size=data_params["z_dim"], num_classes=data_params["num_classes"], T=T).to(device)
model.load_state_dict(torch.load(model_params["model_path"]))
# model.eval()

grid_size_fromyaml =[data_params["x_dim"], data_params["y_dim"], data_params["z_dim"]]
grid_size_fromyaml2 =[data_params["x_dim2"], data_params["y_dim2"], data_params["z_dim2"]]

# test_ds = KittiDataset(directory=val_dir, device=device, num_frames=T, remap=remap, binary_counts=binary_counts, transform_pose=transform_pose, split="valid", grid_size=grid_size_fromyaml, get_gt=False)
# dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers)
test_ds = KittiDataset(directory=val_dir, device=device, num_frames=T, past_frames=past_frames, transform_pose=True, split="valid", grid_size=grid_size_fromyaml, grid_size_train=grid_size_fromyaml2)
dataloader_test = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=test_ds.collate_fn, num_workers=num_workers, pin_memory=True)

if device == "cuda":
    torch.cuda.empty_cache()

setup_seed(seed)

local_hist = torch.zeros(15, 15, dtype=torch.long, device=device)
# Testing
model.eval()
with torch.no_grad():
    index_num = 0
    for input_data, output in tqdm(dataloader_test):
        input_data = torch.tensor(np.array(input_data)).to(device)
        preds = model(input_data)
        #preds = torch.permute(preds, (0, 2, 3, 1))
        preds = torch.permute(preds, (0, 2, 3, 1))

        B, H, W, C = preds.shape
        probs = torch.nn.functional.softmax(preds, dim=3).view(H, W, C)
        fname = test_ds._velodyne_list[index_num]

        data_dir, file_num = fname.split("/velodyne/")
        file_num_nobin = file_num.split(".")[0]

        if index_num == 0:
            dir_2 = os.path.join(data_dir, "MotionNet_Prediciton_200x200")
            pred_dir = os.path.join(dir_2, "bev_labels")
            pred_dir_bev = os.path.join(dir_2, "bev_images")

            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            if not os.path.exists(pred_dir_bev):
                os.makedirs(pred_dir_bev) 

        save_path = os.path.join(pred_dir, file_num)
        
        save_path_bev = os.path.join(pred_dir_bev, file_num_nobin + ".png")
        preds_np = probs.detach().cpu().numpy().astype(np.float32)
        # pred_label = torch.argmax(probs, dim=2)
        # preds_np = pred_label.detach().cpu().numpy().astype(np.uint8)
        #print(preds_np.shape)
        pred_label = np.argmax(preds_np, axis=2).astype(np.uint8)
        img = bev_img(pred_label)
        # img.show()
        # print(save_path_bev)
        pred_label.tofile(save_path)
        img.save(save_path_bev)
        # pdb.set_trace()
        hist_np = fast_hist(output[0].flatten(), pred_label.flatten(), 15)
        hist_torch = torch.from_numpy(hist_np).to(device, dtype=torch.long)
        local_hist += hist_torch
        global_hist_np = local_hist.cpu().numpy()
        mIoUs = per_class_iu(global_hist_np)
        
        cur_mIoU = round(np.nanmean(mIoUs) * 100, 2)
        print(f"===> mIoU:\t {cur_mIoU}")
        # for i in range(T): 
        #     save_temp_dir_frame_input = os.path.join(pred_dir_bev, file_num_nobin +"_Input_T" + str(i) + ".png")

        #     input_temp = input_data[-1,i,28:228,28:228,:].detach().cpu().numpy().astype(np.uint8) # Get 200 x 200 instead  256 x 256 grid
        #     input_bev = form_bev(input_temp, grid_size_fromyaml[:2])
        #     img_bev = bev_img(input_bev)
        #     img_bev.save(save_temp_dir_frame_input)
        index_num += 1

    for idx, class_name in enumerate(CLASS_NAMES):
        class_mIoU = round(mIoUs[idx] * 100, 2)
        print(f"===> {class_name:<15}:\t {class_mIoU}")