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

import monai.losses

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
    train_keys = args.train_keys.split(",")
    eval_keys = args.eval_keys.split(",")

    checkpoint_name = (
        "SYN_HLE_SINGLE"
        + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        + "_"
        + id_generator(6)
    )
    checkpoint_path = os.path.join("checkpoints/", checkpoint_name)
    os.mkdir(checkpoint_path)
    os.mkdir(os.path.join("checkpoints", checkpoint_name, "results"))
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    f = open(os.path.join(checkpoint_path, "args.txt"), "a")
    f.write(",".join([str(batch_size), str(num_epochs), str(learning_rate)]))
    f.write(f"Eval keys: {args.eval_keys}")
    f.write(f"Train keys: {args.train_keys}")
    f.close()

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=(-40, 40), p=0.5),
            A.Affine(translate_percent=0.15, p=0.5),
            A.RandomGamma(),
            A.RandomBrightnessContrast(),
            A.Perspective(scale=(0.05, 0.2), p=0.5),
            A.Normalize(
                mean=[0.0],
                std=[1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
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
    train_ds = dataset.FFHLESingleImage(
        ff_path=os.path.join(args.ff_path, "train"),
        hle_path=args.hle_path,
        keys=train_keys,
        transform=train_transform,
    )

    val_ds = dataset.FirefliesDataset(
        os.path.join(args.ff_path, "eval"), transform=eval_transform
    )

    val_real = dataset.HLEPlusPlus(
        args.hle_path,
        eval_keys,
        transform=eval_transform,
        how_many=10,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )
    test_loader_shuffled = DataLoader(
        val_real,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        val_real,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    # Save config stuff
    A.save(
        train_transform,
        "checkpoints/" + checkpoint_name + "/train_transform.yaml",
        data_format="yaml",
    )
    A.save(
        eval_transform,
        "checkpoints/" + checkpoint_name + "/val_transform.yaml",
        data_format="yaml",
    )

    csv_filename = "eval.csv"
    with open(os.path.join(checkpoint_path, csv_filename), "a+", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            [
                "DiceSyn",
                "IoUSyn",
                "LossSyn",
                "DiceReal",
                "IoUReal",
                "LossReal",
                "TrainLoss",
            ]
        )

    model = unet.UNet().to(DEVICE)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=0.9,
    )
    scheduler = lr_scheduler.PolynomialLR(optimizer, num_epochs, power=0.9)

    best_iou = 0.0
    for epoch in tqdm(range(num_epochs)):
        scheduler.update_lr()

        # Train the network
        train_loss = train(train_loader, loss_func, model, scheduler)

        # Evaluate on Validation Set
        eval_dice, eval_iou, eval_loss = evaluate(val_loader, model, loss_func)

        # Evaluate on Real Data
        real_dice, real_iou, real_loss = evaluate(
            test_loader_shuffled, model, loss_func
        )

        if real_iou.item() > best_iou:
            checkpoint = {"optimizer": optimizer.state_dict()} | model.get_statedict()
            torch.save(
                checkpoint,
                "checkpoints/" + checkpoint_name + f"/best_model.pth.tar",
            )
            best_iou = real_iou.item()

        # Save images on real data
        visualize(test_loader, model, epoch, checkpoint_path)

        with open(
            os.path.join(checkpoint_path, csv_filename), "a", newline=""
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    eval_dice.item(),
                    eval_iou.item(),
                    eval_loss,
                    real_dice.item(),
                    real_iou.item(),
                    real_loss,
                    train_loss,
                ]
            )

    checkpoint = {"optimizer": optimizer.state_dict()} | model.get_statedict()
    torch.save(checkpoint, "checkpoints/" + checkpoint_name + "/model.pth.tar")

    print("\033[92m" + "Training Done!")


def train(train_loader, loss_func, model, scheduler):
    model.train()
    running_average = 0.0
    count = 0
    for images, gt_seg in train_loader:
        if images.shape[0] != 8:
            continue
        scheduler.zero_grad()

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        # forward
        pred_seg = model(images)
        loss = loss_func(pred_seg.float(), gt_seg.long().unsqueeze(1))

        loss.backward()
        scheduler.step()

        running_average += loss.item()
        count += images.shape[0]

    return running_average / count


def visualize(loader, model, epoch, checkpoint_path):
    folder_path = os.path.join(checkpoint_path, "results", str(epoch))
    os.makedirs(folder_path, exist_ok=True)

    count = 0
    model.eval()
    for images, gt_seg in loader:
        if images.shape[0] != 8:
            continue
        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)
        pred_seg = model(images).softmax(dim=1).argmax(dim=1)

        pred_seg = pred_seg.cpu().numpy()
        pred_seg = pred_seg / pred_seg.max()
        pred_seg = pred_seg.astype(np.float32)

        gt_segs = gt_seg.cpu().numpy()
        gt_segs = gt_segs / gt_segs.max()
        gt_segs = gt_segs.astype(np.float32)

        imagas = images.cpu().numpy()

        ims = cv2.hconcat([imagas[0, 0], pred_seg[0], gt_segs[0]])
        ims = (ims * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folder_path, "{0:05d}.png".format(count)), ims)
        count += 1


def evaluate(val_loader, model, loss_func):
    running_average = 0.0
    count = 0

    model.eval()

    dice = torchmetrics.F1Score(task="multiclass", num_classes=3)
    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)

    for images, gt_seg in val_loader:
        if images.shape[0] != 8:
            continue

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long().to(device=DEVICE)

        pred_seg = model(images)
        softmax = pred_seg.softmax(dim=1).detach()

        dice(softmax.cpu(), gt_seg.cpu())
        iou(softmax.cpu(), gt_seg.cpu())

        loss = loss_func(pred_seg.detach(), gt_seg.unsqueeze(1)).item()
        running_average += loss
        count += images.shape[0]

    dice_score = dice.compute()
    iou_score = iou.compute()

    print("DICE: {0:03f}, IoU: {1:03f}".format(dice_score, iou_score))

    return dice_score, iou_score, running_average / count


if __name__ == "__main__":
    main()
