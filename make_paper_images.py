import cv2
import numpy as np
import os


def colorize_glottis(float_im):
    return cv2.applyColorMap(float_im, cv2.COLORMAP_INFERNO)


def copy_color_to_label(image, pred, pred_colored, labels):
    bla = image.copy()
    for label in labels:
        for i in range(3):
            channel = bla[:, :, i]
            channel = np.where(pred == label, pred_colored[:, :, i], channel)

            bla[:, :, i] = channel

    return bla


def make_glottis_im(path):
    for file in os.listdir(path):
        complete_path = os.path.join(path, file)

        im_path = os.path.join(path, "IM" + file)
        gt_path = os.path.join(path, "GT" + file)
        pred_path = os.path.join(path, "PRED" + file)

        img = cv2.imread(complete_path, 0)
        im = img[:, 0:256].astype(np.uint8)
        pred = img[:, 256:512].astype(np.uint8)
        gt = img[:, 512:].astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        pred_col = colorize_glottis(pred)
        gt_col = colorize_glottis(gt)

        pred_inpainted = copy_color_to_label(im, pred, pred_col, [255])
        gt_inpainted = copy_color_to_label(im, gt, gt_col, [255])

        cv2.imshow("IM", im)
        cv2.imshow("PRED", pred_inpainted)
        cv2.imshow("GT", gt_inpainted)
        cv2.waitKey(0)

        cv2.imwrite(im_path, im)
        cv2.imwrite(gt_path, gt_inpainted)
        cv2.imwrite(pred_path, pred_inpainted)


def rearrange_labels(image, old_indices, new_indices):
    temp = np.zeros((len(old_indices), image.shape[0], image.shape[1]))

    for i in range(len(old_indices)):
        temp[i] = np.where(image == old_indices[i], new_indices[i], 0)

    return temp.sum(axis=0)


def make_vf_im(path):
    for file in os.listdir(path):
        complete_path = os.path.join(path, file)

        im_path = os.path.join(path, "IM" + file)
        gt_path = os.path.join(path, "GT" + file)
        pred_path = os.path.join(path, "PRED" + file)

        img = cv2.imread(complete_path, 0)
        im = img[:, 0:256].astype(np.uint8)
        pred = img[:, 256:512].astype(np.uint8)
        gt = img[:, 512:].astype(np.uint8)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        pred = rearrange_labels(pred, [255, 127, 0], [0, 255, 127])
        gt = rearrange_labels(gt, [255, 127, 0], [0, 255, 127])

        pred_col = colorize_glottis(pred.astype(np.uint8))
        gt_col = colorize_glottis(gt.astype(np.uint8))

        pred_inpainted = copy_color_to_label(im, pred, pred_col, [127, 255])
        gt_inpainted = copy_color_to_label(im, gt, gt_col, [127, 255])

        cv2.imshow("IM", im)
        cv2.imshow("PRED", pred_inpainted)
        cv2.imshow("GT", gt_inpainted)
        cv2.waitKey(0)

        cv2.imwrite(im_path, im)
        cv2.imwrite(gt_path, gt_inpainted)
        cv2.imwrite(pred_path, pred_inpainted)


if __name__ == "__main__":
    make_glottis_im("Paper_Vis_Segmentations_Glottis")
    # make_vf_im("Paper_Vis_Segmentations_VF")
