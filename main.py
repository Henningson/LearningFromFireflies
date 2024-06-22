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

    checkpoint_name = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    checkpoint_name = checkpoint_name + "_" + id_generator(6) + "_WD_W_AFFINE"
    checkpoint_path = os.path.join("checkpoints/", checkpoint_name)
    os.mkdir(checkpoint_path)
    os.mkdir(os.path.join("checkpoints", checkpoint_name, "results"))
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate

    f = open(os.path.join(checkpoint_path, "args.txt"), "a")
    f.write(",".join([str(batch_size), str(num_epochs), str(learning_rate)]))
    f.close()

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
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

    train_ds = dataset.FirefliesDataset(
        os.path.join(args.ff_path, "train"), transform=train_transform
    )
    class_weights = 1 - train_ds.generate_dataset_fingerprint().float().to(DEVICE)
    val_ds = dataset.FirefliesDataset(
        os.path.join(args.ff_path, "eval"), transform=eval_transform
    )
    test_ds = dataset.HLEPlusPlus(
        args.hle_path,
        ["CF", "CM", "DD", "FH", "LS", "MK", "MS", "RH", "SS", "TM"],
        transform=eval_transform,
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
        test_ds,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
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
                "AccuracySyn",
                "LossSyn",
                "DiceReal",
                "IoUReal",
                "AccuracyReal",
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
    for epoch in tqdm(range(num_epochs)):
        scheduler.update_lr()

        # Train the network
        train_loss = train(train_loader, loss_func, model, scheduler)

        # Evaluate on Validation Set
        eval_dice, eval_f1, eval_iou, eval_loss = evaluate(val_loader, model, loss_func)

        # Evaluate on Real Data
        real_dice, real_f1, real_iou, real_loss = evaluate(
            test_loader_shuffled, model, loss_func
        )

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
                    eval_f1.item(),
                    eval_loss,
                    real_dice.item(),
                    real_iou.item(),
                    real_f1.item(),
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
        scheduler.zero_grad()

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.to(device=DEVICE)

        # forward
        pred_seg = model(images)
        loss = loss_func(pred_seg.float(), gt_seg.long())

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

    dice = torchmetrics.Dice(num_classes=3)
    iou = torchmetrics.JaccardIndex(task="multiclass", num_classes=3)
    f1 = torchmetrics.Accuracy(task="multiclass", num_classes=3)

    for images, gt_seg in val_loader:

        images = images.to(device=DEVICE)
        gt_seg = gt_seg.long().to(device=DEVICE)

        pred_seg = model(images)
        softmax = pred_seg.softmax(dim=1).detach()

        dice(softmax.cpu(), gt_seg.cpu())
        iou(softmax.cpu(), gt_seg.cpu())
        f1(softmax.cpu(), gt_seg.cpu())

        loss = loss_func(pred_seg.detach(), gt_seg).item()
        running_average += loss
        count += images.shape[0]

    dice_score = dice.compute()
    f1_score = f1.compute()
    iou_score = iou.compute()

    print(
        "DICE: {0:03f}, IoU: {1:03f}, F1: {2:03f}".format(
            dice_score, iou_score, f1_score
        )
    )

    return dice_score, f1_score, iou_score, running_average / count


if __name__ == "__main__":
    main()
