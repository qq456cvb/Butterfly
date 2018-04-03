import numpy as np
import config
import scipy.ndimage
import cv2
import utils
import glob
import os


def data_generator(name2idx, prior_bboxes):
    img_files = glob.glob(os.path.join(config.ROOT_PATH, "Images\\*.jpg"))
    while True:
        fn = np.random.choice(img_files)
        img = cv2.imread(fn)
        ratio, img = utils.resize_keep_ratio(img, config.IMG_SIZE)
        xml = os.path.join(config.ROOT_PATH, "Annotations\\%s.xml" % fn.split('.')[0])
        cls_name, bbox_resized = utils.get_class_and_bbox(xml, ratio)
        cls_idx = name2idx[cls_name]
        bbox_center_normalized = np.array([(bbox_resized[0] + bbox_resized[2]) / 2 / config.IMG_SIZE, (bbox_resized[1] + bbox_resized[3]) / 2 / config.IMG_SIZE])

        # generate bbox offset target
        N = config.IMG_SIZE // 32
        base_offset = int(bbox_center_normalized[0] * N), int(bbox_center_normalized[1] * N)
        target_coarsest = np.zeros([N, N, 3, (4 + 1 + config.NUM_CLASSES)])
        target_coarsest[base_offset[0], base_offset[1], :, 4] = 1

        pos_delta = bbox_center_normalized * N - base_offset
        scale_delta = np.log()

