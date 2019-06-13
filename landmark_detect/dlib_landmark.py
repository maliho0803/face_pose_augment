# -*- coding: utf-8 -*-
import os
import dlib
import numpy as np
from skimage import io
import scipy.io as sio
import glob

def get_landmarks(im):
    rects = detector(im, 1)
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

predictor_path = "./net-data/shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()

predictor = dlib.shape_predictor(predictor_path)
image_folders = sorted(glob.glob('/data/zhoumi/datasets/IJB-A/real/*'))
for image_folder in image_folders:
    types = ('*.jpg', '*.png')
    image_path_list = []
    for files in types:
        image_path_list.extend(sorted(glob.glob(os.path.join(image_folder, files))))
    for face_path in image_path_list:
        # print(face_path)
        img = io.imread(face_path)
        [h, w, c] = img.shape
        # print(img.shape)
        if c > 3:
            img = img[:, :, :3]
            img = (img * 255).astype(np.uint8)
        dets = detector(img, 1)
        # print("Number of faces detected: {}".format(len(dets)))
        if 0 == len(dets):
            continue
        for k, d in enumerate(dets):
            # print("dets{}".format(d))
            # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))
            shape = predictor(img, d)
            # print("Part 0: {}, Part 1: {} ...".format(shape.part(0),  shape.part(1)))
        # print("face_landmark:")

        # print(get_landmarks(img))
        member_path = '/data/zhoumi/datasets/IJB-A/real_landmark/' + face_path.split('/')[-2]
        if not os.path.exists(member_path):
            os.mkdir(member_path)

        save_name = member_path + '/' + face_path.split('/')[-1].replace('.jpg', '') + '_landmark' + '.mat'
        print(save_name)
        sio.savemat(save_name, {'pts': get_landmarks(img).astype(np.double)})
