import cv2
import os
import numpy as np

if __name__ == "__main__":
    train_seg_path = os.path.join("fireflies_dataset", "train", "segmentation")
    for image in sorted(os.listdir(train_seg_path)):
        bla = cv2.imread(os.path.join(train_seg_path, image)).astype(np.float32)
        bla /= bla.max()
        cv2.imshow("Bla", bla)
        cv2.waitKey(0)
