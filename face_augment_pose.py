import scipy.io as sio
import cv2
import sys
import os
import glob
from utility.param_parse import *
from utility.ModelCompletionBFM import *
from PIL import Image
import cv2
import numpy as np
from utility.display_face_model import DrawSolidHead
from utility.ImageMeshing import ImageMeshing
import matplotlib.pyplot as plt
from utility.ImageRotation import ImageRotation
from mesh.Zbuffer_mesh import ZBufferTri
from mesh.FaceFrontalization_mesh import FaceFrontalizationMapping, FaceFrontalizationFilling

face_mode_path = './models/'

#face_contour, face_contour_line, face_contour_front, face_contour_front_line
face_contour_trimed = sio.loadmat(face_mode_path + 'Model_face_contour_trimed.mat')
face_contour = face_contour_trimed['face_contour']

#keypointsfull_contour, parallelfull_contour
face_fullmod_contour = sio.loadmat(face_mode_path + 'Model_fullmod_contour.mat')
keypointsfull_contour = face_fullmod_contour['keypointsfull_contour']
parallelfull_contour  = face_fullmod_contour['parallelfull_contour']

#tri_mouth
face_tri_mouth = sio.loadmat(face_mode_path + 'Model_tri_mouth')

#keypoints
face_keypoints = sio.loadmat(face_mode_path + 'Model_keypoints')
keypoints = face_keypoints['keypoints']

#keypoints,mu_shape,segbin,segbin_tri,sigma,symlist,symlist_tri, tex, tri, w
bfm_model = sio.loadmat(face_mode_path + 'Model_Shape.mat')#The Basel Face Model
tri = bfm_model['tri']
mu_shape = bfm_model['mu_shape']
w = bfm_model['w']

#mu_exp, sigma_exp, w_exp
expression_model = sio.loadmat(face_mode_path + 'Model_Expression.mat')
mu_exp = expression_model['mu_exp']
sigma_exp = expression_model['sigma_exp']
w_exp = expression_model['w_exp']

#Model_FWH, vertex_noear_BFM
FWH_model = sio.loadmat(face_mode_path + 'Model_FWH.mat')
Model_FWH = FWH_model['Model_FWH']#The Full Face Model
vertex_noear_BFM = FWH_model['vertex_noear_BFM']
vertex_noear_BFM = np.transpose(vertex_noear_BFM)

tri_mouth = face_tri_mouth['tri_mouth']

tri_plus = np.hstack((tri, tri_mouth))
layer_width = [0.1, 0.15, 0.2, 0.25, 0.3]
mu = mu_shape + mu_exp

base_dir = '/data/zhoumi/datasets/IJB-A'
img_folders = glob.glob(os.path.join(base_dir, 'real/*'))

for img_folder in img_folders:
    dst_folder = img_folder.replace('real', 'syn')
    if not os.path.exists(dst_folder):
        os.mkdir(dst_folder)

    img_paths = glob.glob(os.path.join(img_folder, '*.jpg'))
    for img_path in img_paths:
        #load image, 3dmm parameters
        mat_path = img_path.replace('.jpg', '_3dmm.mat').replace('real', 'real_3dmm')
        param_3dmm = sio.loadmat(mat_path)
        img = Image.open(img_path)
        img = np.array(img) / 255.0
        Exp_Para = param_3dmm['Exp_Para']
        Pose_Para = param_3dmm['Pose_Para']
        Shape_Para = param_3dmm['Shape_Para']

        #construct 3D face
        f, phi, gamma, theta, t3d = ParaMap_Pose(Pose_Para)

        R = RotationMatrix(phi, gamma, theta)
        P = np.array([[1, 0, 0], [0, 1, 0]])
        alpha = Shape_Para
        alpha_exp = Exp_Para

        express = np.matmul(w_exp, alpha_exp)
        express = np.resize(express, (int(express.shape[0] / 3), 3))

        shape = mu + np.matmul(w, alpha)
        shape = np.resize(shape, (int(shape.shape[0] / 3), 3))
        vertex = shape + express
        # print(vertex)

        vertex_full, tri_full = ModelCompletionBFM(vertex, tri_plus, Model_FWH)
        vertexm_full, temp = ModelCompletionBFM(vertex_noear_BFM, tri_plus, Model_FWH)

        # print(img)
        height, width, nChannels = img.shape
        # print(height, width, nChannels)

        print(t3d)
        ProjectVertex = np.matmul(f * R, np.transpose(vertex)) + np.transpose(np.tile(t3d, (vertex.shape[0], 1)))
        ProjectVertex[1, :] = height + 1 - ProjectVertex[1, :]
        # DrawSolidHead(ProjectVertex, tri)
        # sio.savemat('./ProjectVertexrtex.mat', {'Vertex': ProjectVertex, 'tri':tri})


        ProjectVertex_full = np.matmul(f * R, np.transpose(vertex_full)) + np.transpose(np.tile(t3d, (vertex_full.shape[0], 1)))
        ProjectVertex_full[1, :] = height + 1 - ProjectVertex_full[1, :]
        # print(ProjectVertex_full)
        face_contour_ind = face_contour
        # DrawSolidHead(ProjectVertex_full, tri_full)
        # sio.savemat('./ProjectVertex_full.mat', {'Vertex': ProjectVertex_full, 'tri':tri_full})

        contlist_src, bg_tri, keypointsfull_contour_pose = ImageMeshing(vertex, tri_plus, vertex_full, tri_full, vertexm_full,
                                                                        f, phi, gamma, theta, t3d, keypoints, keypointsfull_contour, parallelfull_contour, img, layer_width)

        bg_vertex_src = contlist_src[0]
        for i in range(1, len(contlist_src)):
            bg_vertex_src = np.hstack([bg_vertex_src, contlist_src[i]])

        # print(bg_vertex_src.shape, ProjectVertex_full.shape, bg_tri.shape, tri_full.shape)
        all_vertex_src = np.hstack([bg_vertex_src, ProjectVertex_full])
        all_tri = np.hstack([bg_tri, np.transpose(tri_full) + bg_vertex_src.shape[1]])
        # sio.savemat('./all_vertex_src.mat', {'Vertex': all_vertex_src, 'tri':all_tri})
        # DrawSolidHead(all_vertex_src, all_tri)

        phi_delta = 0/180 * np.pi
        gamma_delta = 10/180 * np.pi
        theta_delta = 0/180 * np.pi
        phi_ref = phi
        gamma_ref = gamma
        theta_ref = theta
        while np.abs(gamma_ref) < np.pi / 3:
            ## 3. Rotating and Anchor Adjustment
            # Get delta rotation;
            phi_delta = np.abs(phi_delta) * phi / np.abs(phi)
            gamma_delta = np.abs(gamma_delta) * gamma / np.abs(gamma)
            theta_delta = np.abs(theta_delta) * theta / np.abs(theta)
            phi_ref = phi_ref + phi_delta
            gamma_ref = gamma_ref + gamma_delta
            theta_ref = theta_ref + theta_delta
            print(gamma_ref)
            R_ref = RotationMatrix(phi_ref, gamma_ref, theta_ref)

            center_src = np.mean(ProjectVertex_full, axis=1)
            center_src[1] = height + 1 - center_src[1]

            t3d_ref = center_src - np.mean(np.matmul(f * R_ref, np.transpose(vertex_full)), axis=1)

            # print(R_ref.shape, vertex_full.shape, t3d_ref.shape)
            RefVertex = np.matmul(f * R_ref, np.transpose(vertex_full)) + np.tile(np.expand_dims(t3d_ref, axis=1), (1, vertex_full.shape[0]))
            RefVertex[1, :] = height + 1 - RefVertex[1, :]

            Pose_Para_src = np.array([[phi, gamma, theta, t3d[0], t3d[1], t3d[2], f]])
            Pose_Para_ref = np.array([[phi_ref, gamma_ref, theta_ref, t3d_ref[0], t3d_ref[1], t3d_ref[2], f]])

            contlist_ref, t3d_ref = ImageRotation(contlist_src, bg_tri, np.transpose(vertex_full), tri_full, keypointsfull_contour, parallelfull_contour, Pose_Para_src, Pose_Para_ref, img)

            bg_vertex_ref = contlist_ref[0]
            for i in range(1, len(contlist_src)):
                bg_vertex_ref = np.hstack([bg_vertex_ref, contlist_ref[i]])
            all_vertex_ref = np.hstack([bg_vertex_ref, RefVertex])

            bg_tri_num = bg_tri.shape[1]
            bg_ver_num = bg_vertex_src.shape[1]


            ## Further adjust z

            if gamma > 0:
                face_contour_modify = np.hstack([np.array(range(8)), np.array(range(24, 30))])
            else:
                face_contour_modify = np.array(range(9, 23))

            face_contour_nonmodify = np.setdiff1d(np.array(range(keypointsfull_contour.shape[1])), face_contour_modify)
            cont = contlist_ref[0]
            bin = np.zeros((1, cont.shape[1]))
            bin[0, face_contour_nonmodify] = 1
            zmax_bin = bin
            for i in range(1, len(contlist_ref) - 1):
                cont = contlist_ref[i]
                bin = np.zeros((1, cont.shape[1]))
                bin[0, face_contour_nonmodify] = 1
                zmax_bin = np.hstack([zmax_bin, bin])

            zmax_bin = np.hstack([zmax_bin, np.zeros((1, contlist_ref[-1].shape[1]))])
            zmax_ind = np.where(zmax_bin)
            zmax_ind = (zmax_ind[0], zmax_ind[1] + 1)
            bin = np.in1d(bg_tri[0, :], zmax_ind) | np.in1d(bg_tri[1, :], zmax_ind) | np.in1d(bg_tri[2, :], zmax_ind)
            bin = np.where(bin == True)
            zmax_ind = np.unique(np.squeeze(bg_tri[:, bin]))
            #all_vertex_ref[2, zmax_ind] = np.max(all_vertex_ref[2, :])

            ## 4. Get Rotating Result
            valid_half = np.zeros((tri_full.shape[0], 1))
            symlist_tri = np.hstack([np.array([range(tri_full.shape[0])]), [[0]]])
            symlist_tri = np.transpose(np.reshape(symlist_tri, (int(symlist_tri.shape[1] / 2), 2)))

            comp_map, tri_ind = ZBufferTri(all_vertex_ref, all_tri, np.zeros((1, all_tri.shape[1])), -1 * np.ones((height, width, 1)))

            corres_map, corres_map_sym = FaceFrontalizationMapping(np.zeros((img.shape[0], img.shape[1])), tri_ind, all_vertex_src, all_vertex_ref,
                                                                         all_tri, bg_tri_num, valid_half, vertex.shape[0], tri_plus.shape[1], symlist_tri)
            # corres_map = sio.loadmat('./corres_map.mat')['corres_map']
            # img = sio.loadmat('./corres_map.mat')['img']
            # print(np.max(corres_map), np.min(corres_map))

            des_img = FaceFrontalizationFilling(img, corres_map)

            save_name = os.path.join(dst_folder, img_path.split('/')[-1][:-4] + '_' + str(gamma_ref) + '.jpg')
            print(save_name)
            des_img = cv2.cvtColor((des_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_name, des_img)

            # plt.subplot(1,2,1)
            # plt.imshow(img)
            # plt.subplot(1,2,2)
            # plt.imshow(des_img)
            #
            # plt.show()





