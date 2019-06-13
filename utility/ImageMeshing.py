import numpy as np
from .param_parse import *
from scipy.spatial import Delaunay
import scipy.io as sio
import matplotlib.pyplot as plt
from mesh.Zbuffer_mesh import ZBuffer

def imgContourBbox(bbox, wp_num):
    # IMGCONTOUR Summary of this function goes here
    # Detailed explanation goes here
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    wp_num = int(wp_num - 2)
    hp_num = int(round(height / width * (wp_num + 2)) - 2)
    w_inter = (width - 1) / (wp_num + 1)
    h_inter = (height - 1) / (hp_num + 1)

    #up edge
    start_point = [bbox[0], bbox[1]]
    interval = [w_inter, 0]
    img_contour = np.expand_dims(np.array(start_point), axis=1)
    for i in range(1, wp_num + 1):
        temp = [start_point[m] + i * interval[m] for m in range(len(start_point))]
        img_contour = np.hstack([img_contour, np.expand_dims(np.array(temp), axis=1)])

    #right edge
    start_point = [bbox[2], bbox[1]]
    interval = [0, h_inter]
    # img_contour = np.hstack([img_contour, np.expand_dims(np.array(start_point), axis=1)])
    for i in range(hp_num + 1):
        temp = [start_point[m] + i * interval[m] for m in range(len(start_point))]
        img_contour = np.hstack([img_contour, np.expand_dims(np.array(temp), axis=1)])

    #down edge
    start_point = [bbox[2], bbox[3]]
    interval = [-w_inter, 0]
    # iimg_contour = np.hstack([img_contour, np.expand_dims(np.array(start_point), axis=1)])
    for i in range(wp_num + 1):
        temp = [start_point[m] + i * interval[m] for m in range(len(start_point))]
        img_contour = np.hstack([img_contour, np.expand_dims(np.array(temp), axis=1)])

    #left edge
    start_point = [bbox[0], bbox[3]]
    interval = [0, -h_inter]
    # img_contour = np.hstack([img_contour, np.expand_dims(np.array(start_point), axis=1)])

    for i in range(hp_num + 1):
        temp = [start_point[m] + i * interval[m] for m in range(len(start_point))]
        img_contour = np.hstack([img_contour, np.expand_dims(np.array(temp), axis=1)])

    return img_contour

def KeypointsWithPose(phi, gamma, theta, vertex, tri, isoline, keypoints, modify_ind):
    ProjectVertex = np.matmul(RotationMatrix(phi, gamma, 0), vertex)
    ProjectVertex = ProjectVertex - np.transpose(np.tile(np.min(ProjectVertex, axis=1), (ProjectVertex.shape[1], 1))) + 1
    temp = np.reshape(ProjectVertex, (ProjectVertex.shape[0] * ProjectVertex.shape[1], 1))
    ProjectVertex = ProjectVertex / np.max(np.abs(temp))

    keypoints_pose = keypoints - 1
    modify_key = modify_ind
    # print('keypoints_pose', keypoints_pose.shape)

    # contour_line = []
    # for i in range(modify_key.shape[0]):
    #     contour_line.append(isoline[modify_key[i], 0])

    # 3.get the outest point on the contour line
    for i in range(modify_key.shape[0]):
        temp = np.squeeze(isoline[modify_key[i], 0]) - 1
        if (gamma >= 0):
            min_ind = np.where(ProjectVertex[0, temp] == np.min(ProjectVertex[0, temp]))
            keypoints_pose[0, modify_key[i]] = temp[min_ind]
        else:
            max_ind = np.where(ProjectVertex[0, temp] == np.max(ProjectVertex[0, temp]))
            keypoints_pose[0, modify_key[i]] = temp[max_ind]

    return keypoints_pose


def EliminateInternalTri(cont_ver, tri):
    # DrawHeadMesh(cont_ver, tri, cont_ver);
    valid_bin = np.zeros((tri.shape[1], 1))
    for i in range(1, cont_ver.shape[1]):
        # for each contour point, find its related tri
        bin = np.where(tri[0, :] == i, 1, 0) | np.where(tri[1, :] == i, 1, 0) | np.where(tri[2, :] == i, 1, 0)
        conn_tri_ind = np.where(bin == 1)#find(bin)
        conn_tri = np.squeeze(tri[:, conn_tri_ind], axis=1)
        angle_list = []
        for j in range(conn_tri.shape[1]):
            # for each connected tri, find the angle centered at i
            other_point = np.setdiff1d(conn_tri[:, j], i)
            line1 = np.array([i, other_point[0]]) - 1
            line1 = cont_ver[0:2, line1]
            line1 = line1[:, 1] - line1[:, 0]
            line2 = np.array([i, other_point[1]]) - 1
            line2 = cont_ver[0:2, line2]
            line2 = line2[:, 1] - line2[:, 0]
            angle_cos = np.matmul(np.transpose(line1), line2) / np.sqrt(np.matmul(np.transpose(line1), line1)) / np.sqrt(np.matmul(np.transpose(line2), line2))
            angle = np.arccos(angle_cos)
            angle_list.append(angle)
        # print(sum(angle_list))

        if (sum(angle_list) > (350 / 180 * np.pi)):
            #if the sum of angles around the vertex i is 360, it is a
            #concave point
            for j in range(conn_tri.shape[1]):
                # for each connected tri, find the angle centered at i
                other_point = np.setdiff1d(conn_tri[:,j], i)
                # if edge connecting point i is the contour edge, it is a
                # valid triangle
                bin1 = abs(i - other_point[0]) == 1 | abs(i-other_point[0]) == cont_ver.shape[1] - 1
                bin2 = abs(i - other_point[1]) == 1 | abs(i-other_point[1]) == cont_ver.shape[1] - 1
                if (bin1 & bin2):
                    valid_bin[conn_tri_ind[j]] = 1
    return valid_bin


def AnchorAdjustment_Z(contour_all, contour_all_ref, adjust_bin, tri, img):
    PI = 3.1415926

    height, width, nChannels = img.shape
    adjust_bin = np.squeeze(adjust_bin)

    adjust_ind = np.where(1 == adjust_bin)[0] + 1
    print(adjust_ind.shape)

    # get only z coordinates
    # we sovle the equation Y = AX
    # where X is the(x, y) of outpoint_des
    # Y and A represent relations between inpoints_src and outpoint_src
    MAX_EQU_NUM = 2500
    Y_Equ = np.zeros((MAX_EQU_NUM, 1))
    A_Equ = np.zeros((MAX_EQU_NUM, len(adjust_ind)))
    equ_num = 0

    for i in range(len(adjust_ind)):
    # for each outpoint
        pt = adjust_ind[i]

        # find the corresponding tri
        bin = np.in1d(tri[0, :], pt) | np.in1d(tri[1, :], pt) | np.in1d(tri[2, :], pt)
        bin = np.where(True == bin)

        # find connecting point
        temp = np.squeeze(tri[:, bin], axis=1)
        connect = np.unique(np.reshape(temp, (temp.shape[1] * temp.shape[0], 1)))
        connect = np.setdiff1d(connect, pt)
        for j in range(connect.shape[0]):
            pt_con = connect[j]
            if adjust_bin[pt_con - 1]:
                # if connected to a point need adjustment, we module their relationships
                z_offset = contour_all_ref[2, pt - 1] - contour_all_ref[2, pt_con - 1]

                dis = contour_all_ref[[1, 2], pt - 1] - contour_all_ref[[1, 2], pt_con - 1]
                dis = np.sqrt(np.matmul(np.transpose(dis), dis))
                weight = 1 / dis
                weight = 1

                pt1 = np.where(adjust_ind == pt)
                pt_con1 = np.where(adjust_ind == pt_con)

                A = np.zeros((1, len(adjust_ind)))
                A[0, pt1] = 1
                A[0, pt_con1] = -1
                Y = z_offset
                A_Equ[equ_num, :] = A * weight
                Y_Equ[equ_num] = Y * weight
                equ_num = equ_num + 1
            else:
                # if connected to solid point, we module the positions
                z_new = contour_all_ref[2, pt - 1] - contour_all_ref[2, pt_con - 1] + contour_all[2, pt_con - 1]

                dis = contour_all_ref[0: 2, pt - 1] - contour_all_ref[0 :2, pt_con - 1]
                dis = np.sqrt(np.matmul(np.transpose(dis), dis))
                weight = 1 / dis
                weight = 1

                pt1 = np.where(adjust_ind == pt)

                A = np.zeros((1, len(adjust_ind)))
                A[0, pt1] = 1
                Y = z_new
                A_Equ[equ_num, :] = A * weight
                Y_Equ[equ_num] = Y * weight
                equ_num = equ_num + 1

    A_Equ = A_Equ[0:equ_num, :]
    Y_Equ = Y_Equ[0:equ_num]

    ## get the new position
    X = np.squeeze(np.linalg.lstsq(A_Equ, Y_Equ)[0])
    contour_all_z = contour_all
    contour_all_z[2, adjust_ind - 1] = X

    return contour_all_z


def ImageMeshing(vertex, tri_plus, vertex_full, tri_full, vertexm_full,
                 f, phi, gamma, theta, t3d, keypoints, keypointsfull_contour, parallelfull_contour, img, layer_width):
    height, width, nChannels = img.shape
    layer = len(layer_width)

    R = RotationMatrix(phi, gamma, theta)

    ProjectVertex_full = np.matmul(f * R, np.transpose(vertex_full)) + np.transpose(np.tile(t3d, (vertex_full.shape[0], 1)))
    ProjectVertexm_full = np.matmul(f * R, np.transpose(vertexm_full)) + np.transpose(np.tile(t3d, (vertexm_full.shape[0], 1)))

    ProjectVertex_full[1, :] = height + 1 - ProjectVertex_full[1, :]
    ProjectVertexm_full[1, :] = height + 1 - ProjectVertexm_full[1, :]

    contlist = []
    bboxlist = []

    if gamma > 0:
        face_contour_modify = np.hstack([np.array(range(8)), np.array(range(24, 30))])
    else:
        face_contour_modify = np.array(range(9, 23))

    face_contour_ind = KeypointsWithPose(phi, gamma, theta, np.transpose(vertex_full), tri_full, parallelfull_contour, keypointsfull_contour, face_contour_modify)
    face_contour = np.squeeze(ProjectVertex_full[:, face_contour_ind], axis=1)

    contlist.append(face_contour)
    tl = np.min(face_contour[0:2, :], axis=1)
    br = np.max(face_contour[0:2, :], axis=1)
    bbox = np.hstack([tl, br])
    bboxlist.append(bbox)

    #2.Get the MultiLayers between face_contour and img_contour
    #other layers
    nosetip = keypoints[0, 33] - 1

    contour_base = face_contour
    face_center = np.mean(contour_base[0:2,:], axis=1)
    R = RotationMatrix(phi, gamma, theta)
    vertex = np.transpose(vertex)
    vertex_full = np.transpose(vertex_full)

    for i in range(1, layer+1):
        curlayer_width = 1 + layer_width[i-1]
        contour = np.zeros((contour_base.shape[0], contour_base.shape[1]))
        for j in range(contour_base.shape[1]):
            pt = contour_base[:, j]
            x = pt[0] - face_center[0]
            y = pt[1] - face_center[1]
            pt1 = face_center + curlayer_width * np.transpose([x, y])
            contour[0:2, j] = pt1

        t3d_cur = np.matmul(f * R, vertex[:, nosetip]) + t3d - np.matmul(f * curlayer_width * R, vertex[:, nosetip])

        contour3d = np.matmul(f * curlayer_width * R, np.squeeze(vertex_full[:, face_contour_ind])) + np.transpose(np.tile(t3d_cur, (face_contour_ind.shape[1], 1)))
        contour3d[1,:] = height + 1 - contour3d[1, :]

        contour[2, :] = contour3d[2, :]

        contlist.append(contour)
        tl = np.min(contour[0:2, :], axis=1)
        br = np.max(contour[0:2, :], axis=1)
        bbox = np.hstack([tl, br])
        bboxlist.append(bbox)

    #Get the img_contour
    wp_num = 7

    bbox1 = bboxlist[len(bboxlist) - 1]
    bbox2 = bboxlist[len(bboxlist) - 2]
    margin = bbox1 - bbox2
    bbox = bbox1 + margin
    bbox[0] = min(bbox[0], 1)
    bbox[1] = min(bbox[1], 1)
    bbox[2] = max(bbox[2], width)
    bbox[3] = max(bbox[3], height)
    bboxlist.append(bbox)
    wp_num1 = round(wp_num / (bbox1[2] - bbox1[0]) * (bbox[2] - bbox[0]))
    # print(bboxlist)
    img_contour = imgContourBbox(bbox, wp_num1)
    img_contour = np.vstack([img_contour, np.zeros((1, img_contour.shape[1]))])
    contlist.append(img_contour)

    ##Triangulation
    contour_all = contlist[0]
    for i in range(1, len(contlist)):
        contour_all = np.hstack([contour_all, contlist[i]])

    tri_all = Delaunay(np.transpose(contour_all)[:, 0:2]).simplices.copy() + 1

    # plt.triplot(np.transpose(contour_all)[:, 0], np.transpose(contour_all)[:, 1], tri_all - 1)
    # plt.plot(np.transpose(contour_all)[:, 0], np.transpose(contour_all)[:, 1], 'o')
    # plt.show()

    tri_all = np.transpose(tri_all)
    co = np.array(range(face_contour.shape[1])) + 1
    inbin = np.in1d(tri_all[0,:], co) & np.in1d(tri_all[1,:], co) & np.in1d(tri_all[2,:], co)

    #further judge the internal triangles, since there maybe concave tri
    tri_inner = np.squeeze(tri_all[:, np.where(inbin == True)])
    cont_inner = contlist[0]
    #DrawSolidHead(cont_inner, tri_inner, cont_inner);
    #
    # tri_inner = np.array([[24, 4, 5, 9, 6, 12, 2, 16, 22, 22,	25,	25,	25,	30,	17,	29,	11,	14,	14,	15,	14,	14,	6,	16,	24,	18,	29,	19],
    #                       [23, 10, 10,	8,	9,	11,	1,	15,	21,	20,	27,	28,	24,	17,	30,	19,	10,	11,	3,	3,	4,	13,	10,	1,	29,	30,	23,	23],
    #                       [29,	5,	6,	7,	7,	14,	16,	2,	20,	19,	26,	27,	28,	1,	18,	30,	4,	4,	15,	2,	3,	12,	9,	17,	28,	19,	19,	22]])
    # print(tri_inner)
    valid_inner_tri = EliminateInternalTri(cont_inner, tri_inner)
    tri_inner = np.squeeze(tri_all[:, np.where(inbin == True)])
    tri_inner = np.squeeze(tri_inner[:, np.where(valid_inner_tri == 1)])

    if 0 == tri_inner.size:
        tri_all = np.squeeze(tri_all[:, np.where(inbin == False)])
    else:
        tri_all = np.hstack([np.squeeze(tri_all[:, np.where(inbin == False)]), tri_inner])

    # plt.triplot(np.transpose(contour_all)[:, 0], np.transpose(contour_all)[:, 1], np.transpose(tri_all - 1))
    # plt.plot(np.transpose(contour_all)[:, 0], np.transpose(contour_all)[:, 1], 'o')
    # plt.show()

    #DrawHeadMesh(contour_all, tri_all, contlist{1});

    ##Now we need to determine the z coordinates of each contour point
    #Following the two considerations
    #1. There always have face regions in the background
    #2. We don't care about the alignment result of background pixels

    #the z coordinates of img contour out
    # tri_all = sio.loadmat('./tri_all.mat')['tri_all']

    img_contour = contlist[-1]
    img_contour_co = list(range(contour_all.shape[1] - img_contour.shape[1] + 1, contour_all.shape[1] + 1))
    for i in range(len(img_contour_co)):
        ind = img_contour_co[i]
        # find the related triangle
        bin = np.in1d(tri_all[0,:], ind) | np.in1d(tri_all[1, :], ind) | np.in1d(tri_all[2, :], ind)
        bin = np.where(bin == True)
        conn_tri = np.squeeze(tri_all[:, bin], axis=1)
        # print(conn_tri.shape)
        conn_point = np.unique(np.reshape(conn_tri, (conn_tri.shape[0] * conn_tri.shape[1], 1)))
        conn_face_contour_ind = np.setdiff1d(conn_point, img_contour_co) - 1
        if 0 == conn_face_contour_ind.size:
            img_contour[2, i] = np.inf
            continue
        #get the z coordinates of each connect face contour
        z_coordinates = np.squeeze(contour_all[2, conn_face_contour_ind])
        # print(z_coordinates)
        img_contour[2, i] = np.mean(z_coordinates)

    contlist[-1] = img_contour
    contour_all = contlist[0]
    for i in range(1, len(contlist)):
        contour_all = np.hstack([contour_all, contlist[i]])

    # Complement the point with no face contour correspondence
    img_contour_co = np.array(range(contour_all.shape[1] - img_contour.shape[1] + 1, contour_all.shape[1] + 1))

    bin = np.isinf(img_contour[2, :])
    invalid_co = np.where(bin == True)

    while 0 != invalid_co[0].size:
        valid_co = np.where(bin == False)
        img_contour_co_cur = np.squeeze(img_contour_co[valid_co])

        for i in range(img_contour_co.shape[0]):
            ind = img_contour_co[i]
            # find the related triangle
            bin = np.in1d(tri_all[0,:], ind) | np.in1d(tri_all[1,:], ind) | np.in1d(tri_all[2,:], ind)
            bin = np.where(True == bin)
            conn_tri = np.squeeze(tri_all[:, bin], axis=1)
            conn_point = np.unique(np.reshape(conn_tri, (conn_tri.shape[0] * conn_tri.shape[1], 1)))

            conn_face_contour_ind = np.array(np.intersect1d(conn_point, img_contour_co_cur)) - 1

            if 0 == conn_face_contour_ind.size:
                continue

            # get the z coordinates of each connect face contour
            z_coordinates = contour_all[2, conn_face_contour_ind]
            img_contour[2, i] = np.mean(z_coordinates)

        contlist[-1] = img_contour


        contour_all = contlist[0]
        for i in range(1, len(contlist)):
            contour_all = np.hstack([contour_all, contlist[i]])

        bin = np.isinf(img_contour[2,:])
        invalid_co = np.where(True == bin)

    contlist[-1] = img_contour
    # print(img_contour)

    #Showing the result
    contour_all = contlist[0]
    for i in range(1, len(contlist)):
        contour_all = np.hstack([contour_all, contlist[i]])

    ############################################################################################
    ######################## Finally refine the anchor depth with real depth ###################
    ############################################################################################
    depth_ref = np.zeros((height, width, 1))
    depth_ref, tri_ind = ZBuffer(ProjectVertex_full, np.transpose(tri_full), np.expand_dims(ProjectVertexm_full[2,:], axis=0), depth_ref)

    # ZBuffer = sio.loadmat('./ZBuffer.mat')
    # depth_ref = ZBuffer['depth_ref']
    # tri_ind = ZBuffer['tri_ind']

    contour_all_ref = contour_all
    # contlist_ref = contlist
    contlist_new = contlist

    solid_depth_bin_list = []
    for i in range(len(contlist)):
        temp = np.zeros((1, contlist[i].shape[1]))
        solid_depth_bin_list.append(temp)

    solid_depth_bin_list[0] = np.ones((1, contlist[0].shape[1]))

    adjust_ind = list(range(3, 14)) + list(range(18, 30))
    for ai in range(len(adjust_ind)):
        j = adjust_ind[ai]
        count = 0
        for i in range(1, len(contlist_new) - 1):
            ray = contlist_new[i][:, j]
            x = int(round(ray[0])) - 1
            y = int(round(ray[1])) - 1
            if x < 0 or x > width - 1 or y < 0 or y > height - 1:
                continue

            if tri_ind[y, x] == 0:
                continue

            count = count + 1

        if count < 2:
            continue

        for i in range(1, len(contlist_new) - 1):
            ray = contlist_new[i][:, j]
            x = int(round(ray[0])) - 1
            y = int(round(ray[1])) - 1
            if x < 0 or x > width - 1 or y < 0 or y > height - 1:
                continue

            if (tri_ind[y, x] == 0):
                continue

            contlist_new[i][2, j] = depth_ref[y, x]
            solid_depth_bin_list[i][0, j] = 1

    solid_depth_bin = solid_depth_bin_list[0]
    for i in range(1, len(solid_depth_bin_list)):
        solid_depth_bin = np.hstack([solid_depth_bin, solid_depth_bin_list[i]])

    # solid_depth_bin = logical(solid_depth_bin)

    contour_all_new = contlist_new[0]
    for i in range(1, len(contlist_new)):
        contour_all_new = np.hstack([contour_all_new, contlist_new[i]])

    # finally refine non_solid contour
    contour_all_z = AnchorAdjustment_Z(contour_all_new, contour_all_ref, 1 - solid_depth_bin, tri_all, img)

    contour_all_new[2,:] = contour_all_z[2,:]

    # DrawHeadMesh(contour_all_new, tri_all, contour_all_new(:, solid_depth_bin));

    for i in range(len(contlist)):
        # print(contour_all_new.shape)
        contlist[i] = contour_all_new[:, 0: contlist[i].shape[1]]
        contour_all_new = contour_all_new[:, contlist[i].shape[1]: contour_all_new.shape[1] + 1]

    print(contlist[-1].shape)
    return contlist, tri_all, face_contour_ind