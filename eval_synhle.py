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

    checkpoints = [
        "checkpoints/SYNHLE/SYN_HLE_2024-06-26-13:10:04_BIQ562",
        "checkpoints/SYNHLE/SYN_HLE_2024-06-26-13:10:04_0RF4Q9",
    ]

    eval_transform = A.load(
        os.path.join(checkpoints[0], "val_transform.yaml"), data_format="yaml"
    )

    key_pairs_a = [
        ["CF", "CM"],
        ["DD", "FH"],
        ["LS", "RH"],
        ["MK", "MS"],
    ]

    key_pairs_b = [["SS", "TM"]]

    # Here its getting ugly...
    key_pairs_combined = [key_pairs_a, key_pairs_b]

    dices = []
    ious = []

    for checkpoint_path, key_pairs in zip(checkpoints, key_pairs_combined):
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
                shuffle=True,
            )
            loaders.append(loader)

        model = unet.UNet(
            state_dict=torch.load(os.path.join(checkpoint_path, "best_model.pth.tar")),
        ).to(DEVICE)

        for loader, key in zip(loaders, key_pairs):
            dice, iou = evaluate(loader, model)
            dices.append(dice)
            ious.append(iou)

            print(key)
            print(dice, iou)

    dices = np.array(dices)
    ious = np.array(ious)

    print(
        f"Dice-Mean: {dices.mean()}, Dice-STD: {dices.std()}, IoU-Mean: {ious.mean()}, IoU-STD: {ious.std()}"
    )


def evaluate(val_loader, model):
    dice = torchmetrics.F1Score(task="multiclass", num_classes=3)
    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)

    model.eval()
    for images, gt_seg in tqdm(val_loader):
        if images.shape[0] != 8:
            continue

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long().to(device=DEVICE)

        pred_seg = model(images).squeeze()
        softmax = pred_seg.softmax(dim=1)
        dice(softmax.cpu(), gt_seg.cpu())
        iou(softmax.cpu(), gt_seg.cpu())

        test = images[0].cpu().numpy()
        seg_inf = softmax.argmax(dim=1)[0].cpu().numpy()
        gt_seg = gt_seg[0].cpu().numpy()

        test = (test * 255).astype(np.uint8)
        seg_inf = ((seg_inf / 2) * 255).astype(np.uint8)
        gt_seg = ((gt_seg / 2) * 255).astype(np.uint8)

    return dice.compute().item(), iou.compute().item()


if __name__ == "__main__":
    main()
