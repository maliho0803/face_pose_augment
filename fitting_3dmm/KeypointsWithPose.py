import numpy as np
from utility.param_parse import RotationMatrix

def KeypointsWithPose(phi, gamma, theta, vertex, tri, isoline, keypoints, modify_ind):
    ProjectVertex = np.dot(RotationMatrix(phi, gamma, 0), vertex)
    ProjectVertex = ProjectVertex - np.tile(np.min(ProjectVertex, axis=1), (ProjectVertex.shape[1], 1)).T + 1
    ProjectVertex = ProjectVertex/max(abs(np.reshape(ProjectVertex, [-1, 1])))
    
    keypoints_pose = keypoints
    
    # 1. get the keypoints needing modifying
    #     if(gamma < 0)
    #         modify_key = [10:17]
    #     else
    #         modify_key = [1:8]
    #     end
    modify_key = modify_ind
    
    # 2. get the contour line(�ȸ���) of each modify key
    #     contour_line = cell(length(modify_key), 1)
    #     line_width = 0.05
    #     for i = 1:length(modify_key)
    #         # find contour line(�ȸ���)
    #         pt = ProjectVertex(:, keypoints(modify_key(i)))
    #         line_ind = find(abs(ProjectVertex(2,:) - pt(2)) < line_width)
    #         line_ind = intersect(line_ind, find(face_bin==1))
    #         contour_line{i} = line_ind
    #     end
    
    contour_line = np.squeeze(isoline[modify_key])
    
    # 3. get the outest point on the contour line
    for i in range(len(modify_key)):
        tmp = np.squeeze(ProjectVertex[0, contour_line[i] - 1])
        if(gamma >= 0):
            min_ind = np.where(min(tmp) == tmp)
            keypoints_pose[:, modify_key[i]] = contour_line[i][:, min_ind]
        else:
            max_ind = np.where(max(tmp) == tmp)
            keypoints_pose[:, modify_key[i]] = contour_line[i][:, max_ind]
    
    #     t = zeros(size(ProjectVertex))
    #     t(1,keypoints_pose)=1
    #     DrawTextureHead(ProjectVertex, tri, t)
    #     DrawSolidHead(ProjectVertex, tri, ProjectVertex(:,keypoints_pose))
    return keypoints_pose
