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


def generate_csv_graph_new(path: str) -> None:
    eval_path = os.path.join(path, "eval.csv")
    fig, ax = plt.subplots(ncols=1, nrows=1)
    fig.canvas.set_window_title(path.split("/")[-1])

    data_frame = pd.read_csv(eval_path)
    indices = np.arange(0, data_frame.shape[0])

    for key in data_frame.keys():
        plot_key_in_ax(indices, [key], data_frame, ax)

    ax.legend(loc="upper right")


def find_metric_with_std(base_dir: str, metric_key: str):
    metric = []
    for dir in os.listdir(base_dir):
        data_frame = pd.read_csv(os.path.join(base_dir, dir, "eval.csv"))
        metric.append(find_last_metric_in_data_frame(data_frame, metric_key))

    metric = np.array(metric)
    print(f"{metric_key}: Mean: {metric.mean()}     STD:{metric.std()}")

    return metric.mean(), metric.std()


def find_best_metric_with_std(base_dir: str, metric_key: str):
    metric = []
    for dir in os.listdir(base_dir):
        data_frame = pd.read_csv(os.path.join(base_dir, dir, "eval.csv"))
        metric.append(find_best_metric_in_data_frame(data_frame, metric_key))

    metric = np.array(metric)
    print(f"{metric_key}: Mean: {metric.mean()}     STD:{metric.std()}")

    return metric.mean(), metric.std()


def find_last_metric_in_data_frame(data_frame, metric_key: str):
    return data_frame[metric_key].values[-1]


def find_best_metric_in_data_frame(data_frame, metric_key: str):
    return data_frame[metric_key].values.max()


def swap_classes_from_v4_to_v3(path: str, bg_index, glottis_index, vocal_folds_index):
    img_path = os.path.join(path, "images")
    segmentation_path = os.path.join(path, "segmentation")

    for im_id in os.listdir(img_path):
        image = cv2.imread(os.path.join(img_path, im_id), 0)
        segmentation = cv2.imread(os.path.join(segmentation_path, im_id), 0)

        glottis = (segmentation == glottis_index) * 1
        background = (segmentation == bg_index) * 1
        vocal_folds = (segmentation == vocal_folds_index) * 1

        new_seg = np.zeros_like(segmentation)
        new_seg[vocal_folds == 1] = 0
        new_seg[glottis == 1] = 1
        new_seg[background == 1] = 2

        cv2.imwrite(os.path.join(segmentation_path, im_id), new_seg)


if __name__ == "__main__":

    swap_classes_from_v4_to_v3("fireflies_dataset_v4/train", 1, 2, 0)
    swap_classes_from_v4_to_v3("fireflies_dataset_v4/eval", 1, 2, 0)

    path = "checkpoints/HLE_GLOTTIS_ONLY/"
    find_metric_with_std(path, "DiveEval")
    find_metric_with_std(path, "IoUEval")

    """
    dir = "DIFO_2024-06-23-12:46:41_89ZKFK"
    for dir in os.listdir(path):
        generate_csv_graph_new(os.path.join(path, dir))
    plt.show()

    for dir in os.listdir(path):
        vis_checkpoints_over_time(os.path.join(path, dir), "00000.png")
    """
