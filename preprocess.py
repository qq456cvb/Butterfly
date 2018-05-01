import numpy as np
import cv2
import glob
import os
import sys
from shutil import copyfile
from xml.dom import minidom
import config


def get_classes_and_bboxes():
    name2idx = {}
    idx2name = {}
    filenames = glob.glob(os.path.join(config.ROOT_PATH, "Annotations/*.xml"))
    num_classes = 0
    bboxes = []
    for fn in filenames:
        doc = minidom.parse(fn)
        name = doc.getElementsByTagName('name')[0].firstChild.nodeValue
        if name not in name2idx:
            name2idx[name] = num_classes
            idx2name[num_classes] = name
            num_classes += 1
        width, height = [int(doc.getElementsByTagName(n)[0].firstChild.nodeValue) for n in ['width', 'height']]
        if width == 0 or height == 0:
            continue
        xmin, ymin, xmax, ymax = [int(doc.getElementsByTagName(n)[0].firstChild.nodeValue) for n in ['xmin', 'ymin', 'xmax', 'ymax']]
        # we scale the bbox according to training image size
        ratio = min(config.IMG_SIZE / width, config.IMG_SIZE / height)
        bbox = ratio * np.array([xmin, ymin, xmax, ymax])
        bboxes.append(bbox)
    bboxes = np.stack(bboxes, axis=0)
    return num_classes, name2idx, idx2name, bboxes


def kmeans(bboxes, k=9, runs=50):
    # convert into width and height
    bboxes = np.vstack([bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]]).transpose()

    # assume bboxes are in 2d Euclidean space

    best_centers = None
    best_cost = np.inf
    for _ in range(runs):
        centers = bboxes[np.random.choice(bboxes.shape[0], k, replace=False), :]
        assignment = np.zeros(bboxes.shape[0])
        eps = 1e-6
        while True:
            dist = np.linalg.norm(bboxes[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=-1)
            assignment = np.argmin(dist, axis=1)
            old_centers = centers.copy()
            for i in range(k):
                centers[i] = np.mean(bboxes[assignment == i], axis=0)
            if np.sum(np.linalg.norm(old_centers - centers, axis=1)) < eps:
                break
        cost = 0
        for i in range(k):
            cost += np.sum(np.square(bboxes[assignment == i] - centers[np.newaxis, i, :]))
        if cost < best_cost:
            best_cost = cost
            best_centers = centers
    return best_centers


def split_train_val(train_ratio=0.9):
    print('checking directories...')
    for dir in ['train', 'train/Annotations', 'train/JPEGImages', 'val', 'val/Annotations', 'val/JPEGImages']:
        if not os.path.exists(os.path.join(config.ROOT_PATH, dir)):
            os.makedirs(os.path.join(config.ROOT_PATH, dir))
    filenames = glob.glob(os.path.join(config.ROOT_PATH, "Annotations/*.xml"))
    np.random.shuffle(filenames)
    for fn in filenames[:int(len(filenames) * 0.9)]:
        raw_name = fn.split('\\' if os.name == 'nt' else '/')[-1].split('.')[0]
        copyfile(os.path.join(config.ROOT_PATH, 'Annotations', raw_name + '.xml'), os.path.join(config.ROOT_PATH, 'train/Annotations', raw_name + '.xml'))
        copyfile(os.path.join(config.ROOT_PATH, 'JPEGImages', raw_name + '.jpg'), os.path.join(config.ROOT_PATH, 'train/JPEGImages', raw_name + '.jpg'))
    for fn in filenames[int(len(filenames) * 0.9):]:
        raw_name = fn.split('\\' if os.name == 'nt' else '/')[-1].split('.')[0]
        copyfile(os.path.join(config.ROOT_PATH, 'Annotations', raw_name + '.xml'), os.path.join(config.ROOT_PATH, 'val/Annotations', raw_name + '.xml'))
        copyfile(os.path.join(config.ROOT_PATH, 'JPEGImages', raw_name + '.jpg'), os.path.join(config.ROOT_PATH, 'val/JPEGImages', raw_name + '.jpg'))


if __name__ == '__main__':
    split_train_val()
    # i, name2idx, _, bboxes = get_classes_and_bboxes()
    # centers = kmeans(bboxes)
    # idxlist = np.argsort(-centers[:, 0] * centers[:, 1])
    # np.save('prior_bboxes', centers[idxlist])
    # bbox = np.load('prior_bboxes.npy')
    # print(bbox)
    # print(centers[idxlist])
