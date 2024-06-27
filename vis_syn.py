import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import datetime
import albumentations as A
import unet
import torch.nn as nn
import dataset

import torchmetrics
import torchmetrics.detection
import cv2
import numpy as np
import lr_scheduler
import csv
import string
import random

from ConfigArgsParser import ConfigArgsParser
from Args import GlobalArgumentParser

from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)


def main():
    parser = GlobalArgumentParser()
    args = parser.parse_args()
    checkpoint_path = "checkpoints/SYNHLE/SYN_HLE_2024-06-26-13:10:04_0RF4Q9"

    vis_path = os.path.join(
        checkpoint_path,
        "vis",
    )
    os.makedirs(vis_path, exist_ok=True)

    eval_transform = A.load(
        os.path.join(checkpoint_path, "val_transform.yaml"), data_format="yaml"
    )

    key_pairs = [
        ["CF", "CM"],
        ["DD", "FH"],
        ["LS", "RH"],
        ["MK", "MS"],
        ["SS", "TM"],
    ]

    datasets = []
    for key_pair in key_pairs:
        ds = dataset.HLEPlusPlus(
            args.hle_path,
            key_pair,
            how_many=-1,
            transform=eval_transform,
        )
        datasets.append(ds)

    loaders = []
    for ds in datasets:
        loader = DataLoader(
            ds,
            batch_size=8,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )
        loaders.append(loader)

    model = unet.UNet(
        state_dict=torch.load(os.path.join(checkpoint_path, "best_model.pth.tar")),
    ).to(DEVICE)

    count = 0
    for loader in loaders:
        count = visualize(loader, model, vis_path, count)


def visualize(val_loader, model, vis_path, count=0):
    for images, gt_seg in tqdm(val_loader):
        if images.shape[0] != 8:
            continue

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long().to(device=DEVICE)

        pred_seg = model(images).squeeze()
        pred_seg = pred_seg.softmax(dim=1).argmax(dim=1)

        for i in range(images.shape[0]):
            gt_seg_path = os.path.join(vis_path, f"{count:5d}_gt.png")
            pred_seg_path = os.path.join(vis_path, f"{count:5d}_seg.png")
            im_path = os.path.join(vis_path, f"{count:5d}.png")

            gt = gt_seg[i].cpu().numpy()
            seg = pred_seg[i].cpu().numpy()
            im = images[i, 0].cpu().numpy()

            seg = seg / 2
            gt = gt / 2
            im = (im * 255).astype(np.uint8)

            cv2.imshow("IM", im)
            cv2.imshow("GT", gt)
            cv2.imshow("SEG", seg)
            cv2.waitKey(0)

            # cv2.imwrite(im_path, im)
            # cv2.imwrite(gt_seg_path, gt)
            # cv2.imwrite(pred_seg_path, seg)

            count += 1

    return count


if __name__ == "__main__":
    main()
