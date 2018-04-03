import numpy as np
from xml.dom import minidom
import cv2


# reference https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
def resize_keep_ratio(img, target_size):
    target_size = np.array(target_size)
    old_size = img.shape[:2]
    ratio = np.amin(target_size / old_size)
    new_size = [int(x * ratio) for x in old_size]
    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta = target_size - np.array(new_size)
    top, bottom = delta[0] // 2, delta[0] // 2
    left, right = delta[1] // 2, delta[1] // 2

    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return ratio, img, np.array([left, top])


def get_class_and_bbox(xml, ratio=1):
    doc = minidom.parse(xml)
    name = doc.getElementsByTagName('name')[0].firstChild.nodeValue
    xmin, ymin, xmax, ymax = [int(doc.getElementsByTagName(n)[0].firstChild.nodeValue) for n in
                              ['xmin', 'ymin', 'xmax', 'ymax']]
    bbox = ratio * np.array([xmin, ymin, xmax, ymax])
    return name, bbox