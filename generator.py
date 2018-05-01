import numpy as np
import config
import scipy.ndimage
import cv2
import utils
import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import preprocess


# prior bboxes are fed in coarse -> fine order, in unnormalized format
def data_generator(folder_name, name2idx, prior_bboxes):
    img_files = glob.glob(os.path.join(config.ROOT_PATH, folder_name, "JPEGImages/*.jpg"))
    while True:
        fn = np.random.choice(img_files)
        img = cv2.imread(fn, cv2.IMREAD_COLOR)
        ratio, img, bbox_offset = utils.resize_keep_ratio(img, config.IMG_SIZE)
        xml = os.path.join(config.ROOT_PATH, folder_name, "Annotations/%s.xml" % fn.split('\\' if os.name == 'nt' else '/')[-1].split('.')[0])
        cls_name, bbox_resized = utils.get_class_and_bbox(xml, ratio)
        bbox_resized[:2] += bbox_offset
        bbox_resized[2:] += bbox_offset

        # random horizontal flip
        # TODO: add random crop
        if np.random.rand() < 0.5:
            img = img[:, ::-1, :]
            bbox_resized[::2] = config.IMG_SIZE - bbox_resized[::2] - 1
            bbox_resized[::2] = bbox_resized[2::-2]
        cls_idx = name2idx[cls_name]
        bbox_size_normalized = np.array([(bbox_resized[2] - bbox_resized[0]) / config.IMG_SIZE, (bbox_resized[3] - bbox_resized[1]) / config.IMG_SIZE])
        bbox_center_normalized = np.array([(bbox_resized[0] + bbox_resized[2]) / 2 / config.IMG_SIZE, (bbox_resized[1] + bbox_resized[3]) / 2 / config.IMG_SIZE])

        # generate bbox offset target
        scales = [32, 16, 8]
        targets = []
        for i in range(3):
            N = config.IMG_SIZE // scales[i]
            base_offset = int(bbox_center_normalized[0] * N), int(bbox_center_normalized[1] * N)
            target = np.zeros([N, N, 3, (4 + 1 + config.NUM_CLASSES)])
            target[base_offset[0], base_offset[1], :, 4] = 1

            pos_delta = bbox_center_normalized * N - base_offset
            size_delta = np.log(bbox_size_normalized / (prior_bboxes[i*3:i*3+3] / config.IMG_SIZE))

            # position delta comes first
            target[base_offset[0], base_offset[1], :, :4] = np.concatenate([np.tile(pos_delta[np.newaxis, :], (3, 1)), size_delta], axis=1)
            target[base_offset[0], base_offset[1], :, 5:] = np.eye(config.NUM_CLASSES)[cls_idx]
            targets.append(target)

        if config.DEBUG:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            # coco.showAnns(anns)
            ax.add_patch(
                patches.Rectangle(
                    (bbox_resized[0], bbox_resized[1]),
                    bbox_resized[2] - bbox_resized[0],
                    bbox_resized[3] - bbox_resized[1],
                    edgecolor="red",
                    fill=False  # remove background
                )
            )
            plt.show()
        assert img.shape[0] == 256 and img.shape[1] == 256
        yield img, targets[0], targets[1], targets[2]


if __name__ == '__main__':
    classes, name2idx, _, _ = preprocess.get_classes_and_bboxes()
    prior_bboxes = np.load('prior_bboxes.npy')
    gen = data_generator('train', name2idx, prior_bboxes)
    for _ in range(3):
        batch = next(gen)
        for b in batch:
            print(b.shape)
        # print(next(gen)[0].shape)
