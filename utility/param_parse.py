import numpy as np

def ParaMap_Pose(para_Pose):
    phi = para_Pose[0][0]
    gamma = para_Pose[0][1]
    theta = para_Pose[0][2]
    t3dx = para_Pose[0][3]
    t3dy = para_Pose[0][4]
    t3dz = para_Pose[0][5]
    f = para_Pose[0][6]

    t3d = np.array([t3dx, t3dy, t3dz])

    return f, phi, gamma, theta, t3d

def RotationMatrix(angle_x, angle_y, angle_z):
    #get rotation matrix by rotate angle
    phi = angle_x
    gamma = angle_y
    theta = angle_z

    R_x = np.array([[1, 0, 0 ],
                   [0, np.cos(phi), np.sin(phi)],
                   [0, -np.sin(phi), np.cos(phi)]])

    R_y = np.array([[np.cos(gamma), 0, -np.sin(gamma)],
                   [0, 1, 0],
                   [np.sin(gamma), 0, np.cos(gamma)]])

    R_z = np.array([[np.cos(theta), np.sin(theta), 0],
                    [-np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])

    R = np.matmul(np.matmul(R_x, R_y), R_z)
    return R