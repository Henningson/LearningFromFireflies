import os
import cv2
import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def vis_checkpoints_over_time(path: str, image_file: str) -> None:
    result_path = os.path.join(path, "results")

    for epoch_index in sorted(os.listdir(result_path)):
        img = cv2.imread(os.path.join(result_path, epoch_index, image_file))
        cv2.imshow("A", img)
        cv2.waitKey(0)


def vis_segmentation_and_image(path: str):
    img_path = os.path.join(path, "images")
    segmentation_path = os.path.join(path, "segmentation")

    for im_id in os.listdir(img_path):
        image = cv2.imread(os.path.join(img_path, im_id), 0)
        segmentation = cv2.imread(os.path.join(segmentation_path, im_id), 0)
        segmentation = ((segmentation / segmentation.max()) * 255).astype(np.uint8)

        final = cv2.hconcat([image, segmentation])
        cv2.imshow("Final", final)
        cv2.waitKey(0)


def plot_key_in_ax(indices, keys, data_frame, ax):
    for key in keys:
        ax.plot(indices, data_frame[key].values, label=key)


def generate_csv_graph(path: str) -> None:
    eval_path = os.path.join(path, "eval.csv")
    fig, ax = plt.subplots(nrows=2, ncols=2)

    train = ax[0, 0]
    dice = ax[0, 1]
    iou = ax[1, 0]
    f1 = ax[1, 1]

    data_frame = pd.read_csv(eval_path)
    indices = np.arange(0, data_frame.shape[0])

    plot_key_in_ax(indices, ["TrainLoss", "LossSyn", "LossReal"], data_frame, train)
    plot_key_in_ax(indices, ["DiceSyn", "DiceReal"], data_frame, dice)
    plot_key_in_ax(indices, ["IoUSyn", "IoUReal"], data_frame, iou)
    plot_key_in_ax(indices, ["F1Syn", "F1Real"], data_frame, f1)

    train.legend(loc="upper right")
    dice.legend(loc="upper right")
    iou.legend(loc="upper right")
    f1.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    path = "checkpoints/GO2024-06-22-19:47:52_5YXPKL"

    # vis_segmentation_and_image("fireflies_dataset/train/")
    generate_csv_graph(path)
    vis_checkpoints_over_time(path, "00015.png")
