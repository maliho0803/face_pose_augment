import numpy
from .param_parse import *
from .ImageMeshing import KeypointsWithPose
from .display_face_model import *

def AnchorAdjustment_Rotate(all_vertex_src, all_vertex_ref, all_vertex_adjust, tri, anchor_flags, img):

    adjust_ind = np.where([anchor_flags == 2, anchor_flags == 3])[-1]
    adjust_ind = np.sort(adjust_ind, axis=0)

    # height, width, nChannels = img.shape

    # we sovle the equation Y = AX
    # where X is the(x, y) of outpoint_des
    # Y and A represent relations between inpoints_src and outpoint_src
    MAX_EQU_NUM = 2500
    Y_Equ = np.zeros((MAX_EQU_NUM, 1))
    A_Equ = np.zeros((MAX_EQU_NUM, 2 * adjust_ind.shape[0]))
    equ_num = 0

    for i in range(adjust_ind.shape[0]):
        # for each outpoint
        pt = adjust_ind[i]

        # find the corresponding tri
        bin = np.in1d(tri[0, :], pt + 1) | np.in1d(tri[1, :], pt + 1) | np.in1d(tri[2, :], pt + 1)
        bin = np.where(bin == True)

        # find connecting point
        temp = np.squeeze(tri[:, bin], axis=1)
        temp = np.reshape(temp, temp.shape[0] * temp.shape[1])
        connect = np.unique(temp)
        connect = np.setdiff1d(connect, pt + 1) - 1
        for j in range(connect.shape[0]):
            pt_con = connect[j] # the relationship of[pt, pt_con]
            if anchor_flags[0, pt] == 2:
                # if base point is a src point, prefer src relation
                if (anchor_flags[0, pt_con] == 1):
                    # if connect to a base point, module the positions
                    x_new = all_vertex_src[0, pt] - all_vertex_src[0, pt_con] + all_vertex_adjust[0, pt_con]
                    y_new = all_vertex_src[1, pt] - all_vertex_src[1, pt_con] + all_vertex_adjust[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)

                    A = np.zeros((1, 2 * adjust_ind.shape[0]))
                    A[0, (2 * pt1[0],)] = 1
                    Y = x_new
                    A_Equ[equ_num, :] = A
                    Y_Equ[equ_num] = Y
                    equ_num = equ_num + 1

                    A = np.zeros((1, 2 * adjust_ind.shape[0]))
                    A[0, (2 * pt1[0] + 1,)] = 1
                    Y = y_new
                    A_Equ[equ_num, :] = A
                    Y_Equ[equ_num] = Y
                    equ_num = equ_num + 1
                else:# (anchor_flags(pt_con) == 2 | | adjust_ind(pt_con) == 3)
                    # src - src and src - ref relationships:
                    # based on src relationship
                    x_offset = all_vertex_src[0, pt] - all_vertex_src[0, pt_con]
                    y_offset = all_vertex_src[1, pt] - all_vertex_src[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)
                    pt_con1 = np.where(adjust_ind == pt_con)

                    A = np.zeros((1, 2 * adjust_ind.shape[0]))
                    A[0, (2 * pt1[0],)] = 1
                    A[0, (2 * pt_con1[0],)] = -1
                    Y = x_offset
                    A_Equ[equ_num, :] = A
                    Y_Equ[equ_num] = Y
                    equ_num = equ_num + 1

                    A = np.zeros((1, 2 * adjust_ind.shape[0]))
                    A[0, (2 * pt1[0] + 1,)] = 1
                    A[0, (2 * pt_con1[0] + 1,)] = -1
                    Y = y_offset
                    A_Equ[equ_num, :] = A
                    Y_Equ[equ_num] = Y
                    equ_num = equ_num + 1
            else: # (anchor_flags(pt) == 3)
                # if it is a ref point, prefer ref relation
                if anchor_flags[0, pt_con] == 1:
                    # if connect to a base point, module the positions
                    x_new = all_vertex_ref[0, pt] - all_vertex_ref[0, pt_con] + all_vertex_adjust[0, pt_con]
                    y_new = all_vertex_ref[1, pt] - all_vertex_ref[1, pt_con] + all_vertex_adjust[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)

                    A = np.zeros((1, 2 * adjust_ind.shape[0]))
                    A[0, (2 * pt1[0],)] = 1
                    Y = x_new
                    A_Equ[equ_num, :] = A
                    Y_Equ[equ_num] = Y
                    equ_num = equ_num + 1

                    A = np.zeros((1, 2 * adjust_ind.shape[0]))
                    A[0, (2 * pt1[0] + 1,)] = 1
                    Y = y_new
                    A_Equ[equ_num, :] = A
                    Y_Equ[equ_num] = Y
                    equ_num = equ_num + 1
                else: # if (adjust_ind(j) == 3)
                    # ref - ref relationships:
                    # based on ref relationship
                    x_offset = all_vertex_ref[0, pt] - all_vertex_ref[0, pt_con]
                    y_offset = all_vertex_ref[1, pt] - all_vertex_ref[1, pt_con]

                    pt1 = np.where(adjust_ind == pt)
                    pt_con1 = np.where(adjust_ind == pt_con)

                    A = np.zeros((1, 2 * adjust_ind.shape[0]))
                    A[0, (2 * pt1[0],)] = 1
                    A[0, (2 * pt_con1[0],)] = -1
                    Y = x_offset
                    A_Equ[equ_num,:] = A
                    Y_Equ[equ_num] = Y
                    equ_num = equ_num + 1

                    A = np.zeros((1, 2 * adjust_ind.shape[0]))
                    A[0, (2 * pt1[0] + 1,)] = 1
                    A[0, (2 * pt_con1[0] + 1,)] = -1
                    Y = y_offset
                    A_Equ[equ_num, :] = A
                    Y_Equ[equ_num] = Y
                    equ_num = equ_num + 1

    A_Equ = A_Equ[0:equ_num, :]
    Y_Equ = Y_Equ[0:equ_num]

    # get the new position
    X = np.squeeze(np.linalg.lstsq(A_Equ, Y_Equ)[0])

    all_vertex_adjust[0: 2, adjust_ind] = np.transpose(np.reshape(X, (int(X.shape[0] / 2), 2)))
    all_vertex_adjust[2, adjust_ind] = all_vertex_ref[2, adjust_ind]

    return all_vertex_adjust

def ImageRotation(contlist_src, bg_tri, vertex, tri, face_contour_ind, isoline_face_contour, Pose_Para_src, Pose_Para_ref, img):
    height, width, nChannels = img.shape
    f, phi, gamma, theta, t3d= ParaMap_Pose(Pose_Para_src)
    f_ref, phi_ref, gamma_ref, theta_ref, t3d_ref = ParaMap_Pose(Pose_Para_ref)

    ProjectVertex_ref = np.matmul(f_ref * RotationMatrix(phi_ref, gamma_ref, theta_ref), vertex) + np.tile(np.expand_dims(t3d_ref, 1), (1, vertex.shape[1]))
    ProjectVertex_ref[1, :] = height + 1 - ProjectVertex_ref[1, :]

    all_vertex = contlist_src[0]
    for i in range(1, len(contlist_src)):
        all_vertex = np.hstack([all_vertex, contlist_src[i]])

    all_vertex_src = all_vertex

    ## 1. get the preliminary position on the ref frame
    all_vertex[1, :] = height + 1 - all_vertex[1, :]

    # The fitting fomula: f * R * vertex + t3d = ProjectVertex
    # Thus: vertex = (1 / f) * R. ^ (-1) * (ProjectVertex - t3d)
    R = RotationMatrix(phi, gamma, theta)
    all_vertex_ref = np.matmul((1 / f) * np.linalg.inv(R), (all_vertex - np.tile(np.expand_dims(t3d, axis=1), (1, all_vertex.shape[1]))))

    # Go to the reference position
    R_ref = RotationMatrix(phi_ref, gamma_ref, theta_ref)
    all_vertex_ref = np.matmul(f * R_ref, all_vertex_ref) + np.tile(np.expand_dims(t3d_ref, 1), (1, all_vertex_ref.shape[1]))
    all_vertex_ref[1, :] = height + 1 - all_vertex_ref[1, :]
    all_vertex[1, :] = height + 1 - all_vertex[1, :]

    # DrawSolidHead(all_vertex, bg_tri)

    # DrawSolidHead(all_vertex_ref, bg_tri);

    ## 2.Landmark marching
    if gamma > 0:
        face_contour_modify = np.hstack([np.array(range(8)), np.array(range(24, 30))])
    else:
        face_contour_modify = np.array(range(9, 23))

    adjust_ind = np.hstack([np.array(range(3, 14)), np.array(range(18, 30))])
    gamma_delta = gamma_ref - gamma
    gamma_temp = gamma + gamma_delta / 2.5
    face_contour_ind = KeypointsWithPose(phi_ref, gamma_temp, theta_ref, vertex, tri, isoline_face_contour, face_contour_ind, face_contour_modify)
    face_contour_ind2 = KeypointsWithPose(phi_ref, gamma, theta_ref, vertex, tri, isoline_face_contour, face_contour_ind + 1, face_contour_modify)
    face_contour_ind[0, adjust_ind] = face_contour_ind2[0, adjust_ind]
    face_contour_ref = np.squeeze(ProjectVertex_ref[:, face_contour_ind], axis=1)

    all_vertex_adjust = np.zeros(all_vertex_ref.shape)
    all_vertex_adjust[:, 0: face_contour_ref.shape[1]] = face_contour_ref

    # DrawSolidHead(ProjectVertex_ref, tri, face_contour_ref);

    # imshow(img);
    # hold on;
    # plot(face_contour_ref(1,:), face_contour_ref(2,:), 'b.');
    # hold off;

    # 5. Rotate other img contour
    src_seq = face_contour_modify # favor relationships on src
    ref_seq = np.setdiff1d(np.array(range(face_contour_ind.shape[1])), src_seq) # face relationships on ref

    # 1 for solid anchor; 2 for src anchor; 3 for ref anchor
    anchor_flags_list = []

    for i in range(len(contlist_src)):
        flags = np.zeros((1, contlist_src[i].shape[1]))
        if i == 0 :
            # the face contour, all are solid anchors
            flags[0, :] = 1
        elif i == len(contlist_src) - 1:
        # the image contour, all are src anchors
            flags[0, :] = 2
        else:
            # middle points
            flags[0, src_seq] = 2
            flags[0, ref_seq] = 3
        anchor_flags_list.append(flags)

    anchor_flags = anchor_flags_list[0]
    print(anchor_flags.shape)
    for i in range(1, len(anchor_flags_list)):
        anchor_flags = np.hstack([anchor_flags, anchor_flags_list[i]])

    all_vertex_adjust = AnchorAdjustment_Rotate(all_vertex_src, all_vertex_ref, all_vertex_adjust, bg_tri, anchor_flags, img)
    # print(all_vertex_adjust)

    # trisurf2D(all_vertex_adjust, bg_tri, img)

    contlist_ref = []
    for i in range(len(contlist_src)):
        contour = all_vertex_adjust[:, 0: contlist_src[i].shape[1]]
        contlist_ref.append(contour)
        all_vertex_adjust = all_vertex_adjust[:, contlist_src[i].shape[1]: all_vertex_adjust.shape[1]]
        # print(contour)
    return contlist_ref, t3d_ref