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


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def main():
    parser = GlobalArgumentParser()
    args = parser.parse_args()
    checkpoint_path = (
        "checkpoints/SYN_GLOTTIS_ONLY/GO_DATAAUG2024-06-23-23:50:04_HWCWZW"
    )

    eval_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    test_ds = dataset.HLEOnlyGlottis(
        args.hle_path,
        ["CF", "CM", "DD", "FH", "LS", "MK", "MS", "RH", "SS", "TM"],
        how_many=-1,
        transform=eval_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    model = unet.UNet(out_channels=1).to(DEVICE)
    model.load_from_dict(os.path.join(checkpoint_path))
    dice, dice_std, iou, iou_std = evaluate(test_loader, model)


def evaluate(val_loader, model):
    running_average = 0.0
    count = 0

    model.eval()

    dice = torchmetrics.F1Score(task="binary")
    iou = torchmetrics.JaccardIndex(task="binary")

    for images, gt_seg in val_loader:
        if images.shape[0] != 8:
            continue

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long().to(device=DEVICE)

        pred_seg = model(images).squeeze()
        softmax = pred_seg.sigmoid()
        dice(softmax.cpu(), gt_seg.cpu())
        iou(softmax.cpu(), gt_seg.cpu())

    dice_score = dice.compute()
    iou_score = iou.compute()

    return dice_score, iou_score, running_average / count


if __name__ == "__main__":
    main()
