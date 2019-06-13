import scipy.io as sio
import numpy as np
import os
import glob
from PIL import Image
from fitting_3dmm.FittingModel import FittingModel
from utility.param_parse import RotationMatrix

face_mode_path = './models/'
#keypoints,mu_shape,segbin,segbin_tri,sigma,symlist,symlist_tri, tex, tri, w
bfm_model = sio.loadmat(face_mode_path + 'Model_Shape.mat')#The Basel Face Model
tri = bfm_model['tri']
mu_shape = bfm_model['mu_shape']
w = bfm_model['w']
sigma = bfm_model['sigma']
keypoints = bfm_model['keypoints']

#mu_exp, sigma_exp, w_exp
expression_model = sio.loadmat(face_mode_path + 'Model_Expression.mat')
mu_exp = expression_model['mu_exp']
sigma_exp = expression_model['sigma_exp']
w_exp = expression_model['w_exp']

mu = mu_shape + mu_exp

#parallel parallel_face_contour
Modelplus_parallel = sio.loadmat(face_mode_path + 'Modelplus_parallel.mat')
parallel = Modelplus_parallel['parallel']
parallel_face_contour = Modelplus_parallel['parallel_face_contour']

## Parameter
layer = 3
layer_width = [0.2,0.5,0.8]

base_path = '/data/zhoumi/datasets/IJB-A'

img_folders = glob.glob(os.path.join(base_path, 'real/*'))

for img_folder in img_folders:
    dst_folder = img_folder.replace('real', 'real_3dmm')
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    img_paths = glob.glob(os.path.join(img_folder, '*.jpg'))
    for img_path in img_paths:
        #load image, 3dmm parameters
        mat_path = img_path.replace('.jpg', '_landmark.mat').replace('real', 'real_landmark')

        img = Image.open(img_path)
        img = np.array(img) / 255.0

        height, width, nChannels = img.shape

        if os.path.exists(mat_path):
            landmarks = sio.loadmat(mat_path)['pts']
            pt2d = landmarks.T

            pt2d[1, :] = height + 1 - pt2d[1, :]


            f, phi, gamma, theta, t3d, alpha, alpha_exp = FittingModel(pt2d, mu, w, sigma, w_exp, sigma_exp, tri, parallel, keypoints, img)

            Pose_Para = np.hstack([np.array([phi]), np.array([gamma]), np.array([theta]), t3d, np.array([f])])
            Shape_Para = alpha
            Exp_Para = alpha_exp

            save_name = mat_path.replace('landmark', '3dmm')
            print(save_name)
            sio.savemat(save_name, {"Pose_Para" : Pose_Para, "Shape_Para" : Shape_Para, "Exp_Para" : Exp_Para, "img" : img, "pt2d" : pt2d})

            # R = RotationMatrix(phi, gamma, theta)

            # express3d = mu_exp + np.dot(w_exp, alpha_exp)
            # express3d = np.reshape(express3d, (int(express3d.shape[0] / 3), 3)).T
            # shape3d = mu_shape + np.dot(w, alpha)
            # shape3d = np.reshape(shape3d, (int(shape3d.shape[0] / 3), 3)).T
            # vertex3d = shape3d + express3d
            #
            # ProjectVertex = np.dot(f * R, vertex3d) + np.tile(t3d, (vertex3d.shape[1], 1)).T
            # ProjectVertex[1, :] = height + 1 - ProjectVertex[1, :] # Fitting结果，模型上每一点在图像上的位置

            # sio.savemat('./new.mat', {'ProjectVertex' : ProjectVertex, "tri":tri})