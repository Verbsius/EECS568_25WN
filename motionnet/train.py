import os
import sys
import yaml
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
from datetime import datetime

from Data.kitti_dataset import KittiDataset
from MotionNet.Model import MotionNet
from utils import setup_seed, iou_one_frame, bev_img, form_bev

def load_configs(model_cfg_path, data_cfg_path):
    with open(model_cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f)
    with open(data_cfg_path, "r") as f:
        data_cfg = yaml.safe_load(f)
    return model_cfg, data_cfg

def get_dataloaders(train_dir, val_dir, T, past_frames, transform_pose, 
    grid_size, grid_size2, num_workers, B, device):
        
    train_dataset = KittiDataset(
        directory=train_dir,
        device=device,
        num_frames=T,
        past_frames=past_frames,
        transform_pose=transform_pose,
        split="train",
        grid_size=grid_size,
        grid_size_train=grid_size2)

    val_dataset = KittiDataset(
        directory=val_dir,
        device=device,
        num_frames=T,
        past_frames=past_frames,
        transform_pose=transform_pose,
        split="valid",
        grid_size=grid_size,
        grid_size_train=grid_size2)

    train_loader = DataLoader(train_dataset, batch_size=B, shuffle=True, 
        collate_fn=train_dataset.collate_fn, num_workers=num_workers, 
        pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=B, shuffle=True, 
        collate_fn=val_dataset.collate_fn, num_workers=num_workers, 
        pin_memory=True)

    return train_loader, val_loader

def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, 
    writer, config, grid_size, scaler, use_amp):
    print(f"Training epoch {epoch} started.")
    model.train()
    total_loss, total_acc, train_count = 0.0, 0.0, 0


    for input_data, output in tqdm(dataloader):
        optimizer.zero_grad()
        input_data = torch.tensor(np.array(input_data)).to(device)
        output = torch.tensor(np.array(output)).to(device)

        if use_amp:
            from torch import amp
            with amp.autocast(device_type='cuda'):
                preds = model(input_data)
                preds = torch.permute(preds, (0, 2, 3, 1))
                output = output.view(-1).long()
                preds = preds.contiguous().view(-1, preds.shape[3])
                if config["remove_zero"]:
                    mask = output != 0
                    output, preds = output[mask], preds[mask]
                loss = criterion(preds, output)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update() 
        else:
            preds = model(input_data)
            preds = torch.permute(preds, (0, 2, 3, 1))
            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[3])
            if config["remove_zero"]:
                mask = output != 0
                output, preds = output[mask], preds[mask]
            loss = criterion(preds, output)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(probs, dim=1)
            acc = (preds == output).float().mean().item()

        writer.add_scalar(f'{config["model_name"]}/Loss/Train', loss.item(), train_count)
        writer.add_scalar(f'{config["model_name"]}/Accuracy/Train', acc, train_count)

        total_loss += loss.item()
        total_acc += acc
        train_count += 1
    
    print(f"[Debug] Training epoch {epoch} finished.", flush=True)
    print(f"[Debug] train_count: {train_count}", flush=True)
    torch.cuda.empty_cache()


    return total_loss / train_count, total_acc / train_count

def validate_one_epoch(model, dataloader, criterion, device, writer, epoch, num_classes, remove_zero, model_name):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_intersections = torch.zeros(num_classes - 1)
    all_unions = torch.ones(num_classes - 1) * 1e-6

    with torch.no_grad():
        for input_data, output in tqdm(dataloader):
            input_data = torch.tensor(np.array(input_data)).to(device)
            output = torch.tensor(np.array(output)).to(device)

            preds = model(input_data)
            preds = torch.permute(preds, (0, 2, 3, 1))

            output = output.view(-1).long()
            preds = preds.contiguous().view(-1, preds.shape[3])

            if remove_zero:
                mask = output != 0
                output, preds = output[mask], preds[mask]

            loss = criterion(preds, output)
            running_loss += loss.item()

            probs = torch.nn.functional.softmax(preds, dim=1)
            preds = torch.argmax(probs, dim=1)

            correct += (preds == output).sum().item()
            total += output.numel()

            intersection, union, uni_classes = iou_one_frame(preds, output, n_classes=num_classes)
            for cls in range(uni_classes.shape[0]):
                c = uni_classes[cls] - 1
                all_intersections[c] += intersection[cls]
                all_unions[c] += union[cls]

    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    miou = torch.mean(all_intersections / all_unions).item()

    writer.add_scalar(f'{model_name}/Loss/Val', avg_loss, epoch)
    writer.add_scalar(f'{model_name}/Accuracy/Val', accuracy, epoch)
    writer.add_scalar(f'{model_name}/mIoU/Val', miou, epoch)

    return avg_loss, accuracy, miou

def setup_logger(log_path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # 防止重复添加多个 handler（多次调用时）

    # 日志格式
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 文件 handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def main():
    model_cfg_path = "./Configs/MotionNet.yaml"
    data_cfg_path = "./Configs/kitti.yaml"
    model_cfg, data_cfg = load_configs(model_cfg_path, data_cfg_path)

    MODEL_CONFIG = model_cfg.get("model_name", "MotionNet")
    DATA_CONFIG = model_cfg.get("data_name", "kitti")
    DEBUG = model_cfg.get("debug", False)
    USE_AMP = model_cfg.get("use_amp", False)

    print("CUDA available:", torch.cuda.is_available())
    print("Device count:", torch.cuda.device_count())
    print("Device name:", torch.cuda.get_device_name(0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(model_cfg["seed"])

    log_dir = f"./Models/Logs/{MODEL_CONFIG}"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logger = setup_logger(log_path)
    logger.info("Training started.")

    model = MotionNet(height_feat_size=data_cfg["z_dim"], num_classes=data_cfg["num_classes"], T=model_cfg["T"]).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=model_cfg["lr"], betas=(model_cfg["BETA1"], model_cfg["BETA2"]))
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9, 12, 15, 18], gamma=0.5)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda') if USE_AMP else None

    grid_size = [data_cfg["x_dim"], data_cfg["y_dim"], data_cfg["z_dim"]]
    grid_size2 = [data_cfg["x_dim2"], data_cfg["y_dim2"], data_cfg["z_dim2"]]



    train_loader, val_loader = get_dataloaders(data_cfg["train_dir"], data_cfg["val_dir"], model_cfg["T"], model_cfg["past_frames"], model_cfg["transform_pose"], grid_size, grid_size2, data_cfg["num_workers"], model_cfg["B"], device)

    writer = SummaryWriter(f"./Models/Runs/{MODEL_CONFIG}")
    save_dir = f"./Models/Weights/{MODEL_CONFIG}"
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(model_cfg["epoch_num"]):
        print(f"Epoch {epoch} / {model_cfg['epoch_num']}")
        logger.info(f"Epoch {epoch} / {model_cfg['epoch_num']}")

    
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, writer, model_cfg, grid_size, scaler, USE_AMP)
        print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}", flush=True)
        logger.info(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")

        val_loss, val_acc, val_miou = validate_one_epoch(model, val_loader, criterion, device, writer, epoch, data_cfg["num_classes"], model_cfg["remove_zero"], MODEL_CONFIG)
        print(f"[Epoch {epoch}] Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, mIoU: {val_miou:.4f}", flush=True)
        logger.info(f"Val Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}, mIoU: {val_miou:.4f}")

        torch.save(model.state_dict(), os.path.join(save_dir, f"Epoch{epoch}.pt"))
        scheduler.step()

    writer.close()
    logger.info("Training finished.")

if __name__ == "__main__":
    main()
