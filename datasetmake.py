import os
import numpy as np
from PIL import Image
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm


json_file = os.listdir("deepfashion2/train/annos/") 
picture_file = os.listdir("deepfashion2/train/image/")

def plot_png(ID, coordinate):
    a = np.array([coordinate], dtype = np.int32)
    return a

def get_ID_and_XY(json_file):
    ID = []
    coordinate = []
    with open(json_file, encoding='utf-8') as f:
        label = json.load(f)
        for key, val in label.items():
            if 'item' in key:
                occlusion = label[key]['occlusion']
                viewpoint = label[key]['viewpoint']
                category = label[key]['category_id']
                xy = label[key]['segmentation']
                ID.append(category)
                coordinate.append(xy)
        f.close()
    return ID, coordinate

for a in tqdm(range(len(picture_file)), position=0):
    picture = picture_file[a]
    name, houzhui = os.path.splitext(picture)
    ID, coordinate = get_ID_and_XY('deepfashion2/train/annos/' + name + '.json')
    picture = cv2.imread('deepfashion2/train/image/' + picture)
    im = np.zeros_like(picture)
    for i in range(len(ID)):
        a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        for j in range(len(coordinate[i])):
            coordinate_1 = [coordinate[i][j][k: k + 2] for k in range(0, len(coordinate[i][j]), 2)]
            a[j] = plot_png(ID[i], coordinate_1)
            if ID[i] < 3:
                im = cv2.fillPoly(im, a[j], (255, 255, 255))
            if ID[i] > 2 and ID[i] < 5:
                im = cv2.fillPoly(im, a[j], (0, 128, 0))
            if ID[i] > 4 and ID[i] < 7:
                im = cv2.fillPoly(im, a[j], (255, 255, 255))
            if ID[i] > 6 and ID[i] < 9:
                im = cv2.fillPoly(im, a[j], (128, 128, 0))
            if ID[i] == 9:
                im = cv2.fillPoly(im, a[j], (114, 0, 114))
            if ID[i] > 9:
                im = cv2.fillPoly(im, a[j], (128, 0, 128))
        cv2.imwrite('deepfashion2/train/png/{name}.png'.format(name = name), im)