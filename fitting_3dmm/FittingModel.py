from .FittingShape import FittingShape
from .FittingExpression import FittingExpression
from .KeypointsWithPose import KeypointsWithPose
from .FittingPose import *
import numpy as np
from utility.param_parse import RotationMatrix

def FittingModel(pt2d, mu, w, sigma, w_exp, sigma_exp, tri, parallel, keypoints, img):
    #   fit the 3DMM with the 68 landmarks and return the pose, shape and
    #   expression parameters
    #
    #   Paras:
    #   @pt2d      : The 68 landmarks.
    #   @mu        : The mean shape of 3DMM
    #   @w         : The PCA axes of 3DMM shape
    #   @sigma     : The PCA standard variations of 3DMM shape
    #   @w_exp     : The PCA axes of 3DMM expression
    #   @sigma_exp : The PCA standard variations of 3DMM expression
    #   @parallel  : The parallel on the face model see section 2.2
    #   @keypoints : the 68 landmarks make up on the 3D model, which has the
    #                same semantic positions as pt2d
    #   @img       : the input image
    #
    #   Outputs:
    #   ==========
    #   @f         : The scale parameter of pose
    #   @phi       : The pitch angle of pose
    #   @gamma     : The yaw angle of pose
    #   @theta     : The roll angle of pose
    #   @t3d       : The translation of pose
    #   @alpha     : The shape paramters
    #   @alpha_exp : The expression paramters

    height, width, nChannels = img.shape

    PI = 3.1415926

    LeftVis =  np.hstack([np.array(range(9)), np.array(range(17, 48)), np.array([48]), np.array([60]), np.array([64]), np.array([54])])
    RightVis = np.hstack([np.array(range(8,17)), np.array(range(17, 48)), np.array([48]), np.array([60]), np.array([64]), np.array([54])])

    # Firstly Pose Coarse Estimation, set valid keypoints
    vertex = np.reshape(mu, (int(mu.shape[0] / 3), 3)).T
    pt3d = np.squeeze(vertex[:, keypoints - 1], axis=1)

    ## Coarse Pose Estimate
    Pl = estimate_affine_matrix_3d22d(pt3d[:, LeftVis].T, pt2d[:, LeftVis].T)
    Pr = estimate_affine_matrix_3d22d(pt3d[:, RightVis].T, pt2d[:, RightVis].T)
    _, Rl, _ = P2sRt(Pl)
    _, Rr, _ = P2sRt(Pr)
    phil, gammal, thetal = matrix2angle(Rl)
    phir, gammar, thetar = matrix2angle(Rr)

    if abs(gammal) > abs(gammar):
        phi = phil
        gamma = gammal
        theta = thetal
    else:
        phi = phir
        gamma = gammar
        theta = thetar

    R = RotationMatrix(phi, gamma, theta)

    valid_key = np.array(range(68))
    valid_key1 = np.vstack([3 * valid_key, 3*valid_key + 1, 3*valid_key + 2])
    valid_key1 = np.reshape(valid_key1.T, [-1, 1])


    ## modify keypoints in each interation
    # normal algorithm

    # shape and expression parameters
    alpha = np.zeros((w.shape[1], 1))
    alpha_exp = np.zeros((w_exp.shape[1], 1))

    iteration = 0
    maxiteration = 4

    ## Firstly pose and expression
    while(True):
        if(iteration > maxiteration):
            break

        iteration = iteration + 1

        ## 1. Update keypoints
        if(gamma < 0):
            modify_ind = np.array(range(9, 17))
        else:
            modify_ind = np.array(range(8))

        keypoints_cur = KeypointsWithPose(phi, gamma, theta, vertex, tri, parallel, keypoints, modify_ind).astype(np.uint32)
        keypoints_cur1 = np.vstack([3 * keypoints_cur - 2, 3 * keypoints_cur - 1, 3 * keypoints_cur])
        keypoints_cur1 = np.reshape(keypoints_cur1.T, [-1, 1])

        # truncate model with keypoint index
        mu_key = np.squeeze(mu[keypoints_cur1 - 1], axis=1)
        w_key = np.squeeze(w[keypoints_cur1 - 1, :], axis=1)
        w_exp_key = np.squeeze(w_exp[keypoints_cur1 - 1, :], axis=1)

        express3d_key = np.dot(w_exp_key, alpha_exp)
        express3d_key = np.reshape(express3d_key, (int(express3d_key.shape[0] / 3), 3)).T

        shape3d_key = mu_key + np.dot(w_key, alpha)
        shape3d_key = np.reshape(shape3d_key, (int(shape3d_key.shape[0] / 3), 3)).T
        vertex_key = shape3d_key + express3d_key

    #     pt3d1 = f * R * vertex_key + repmat(t3d, 1, size(vertex_key,2))
    #     pt3d1(2,:) = height + 1 - pt3d1(2,:)
    #     pt2d1 = pt2d
    #     pt2d1(2,:) = height + 1 - pt2d1(2,:)
    #     imshow(img)
    #     hold on
    #     plot(pt2d1(1,:), pt2d1(2,:), 'b.')
    #     plot(pt3d1(1,:), pt3d1(2,:), 'r.')
    #     hold off

        ## 2. Pose Estimate
        pose_fitting_ind = np.hstack([np.array(range(17)), np.array(range(27, 36)), np.array(range(36, 48)), np.array([48]), np.array([60]), np.array([64]), np.array([54])])
        pt3d = vertex_key
        P = estimate_affine_matrix_3d22d(pt3d[:, pose_fitting_ind].T, pt2d[:, pose_fitting_ind].T)
        f, R, t3d = P2sRt(P)
        phi, gamma, theta = matrix2angle(R)
        R = RotationMatrix(phi, gamma, theta)

        ## 2.expression fitting
        # update shape3d
        # Usually we can't fit expression accuratly, which always lead to over
        # normalization, thus we weaken expression coefficients
        # f = 4.5770e-04
        # t3d = np.array([86.0867, 82.9362, -37.2772])
        # R = np.array([[0.9694, -0.0852, -0.2304], [0.0910, 0.9957, 0.0149], [0.2282, -0.0354, 0.9730]])

        beta_exp = 1000
        alpha_exp = FittingExpression(pt2d[:, valid_key], shape3d_key[:, valid_key], np.array(range(valid_key1.shape[0])), R, t3d, f, np.squeeze(w_exp_key[valid_key1, :], axis=1), sigma_exp, beta_exp)
        express3d_key = np.dot(w_exp_key, alpha_exp)
        express3d_key = np.reshape(express3d_key, (int(express3d_key.shape[0] / 3), 3)).T

        ## 3.shape fitting
        beta_shape = 3000
        alpha = FittingShape(pt2d[:, valid_key], express3d_key[:, valid_key], np.array(range(valid_key1.shape[0])), R, t3d, f,
                             np.squeeze(mu_key[valid_key1], axis=1), np.squeeze(w_key[valid_key1,:], axis=1), sigma, beta_shape)
    return f, phi, gamma, theta, t3d, alpha, alpha_exp



