import scipy.io as sio
import numpy as np
import sys
from .param_parse import *
from .FittingModel3D_validpoint import *

def ModelCompletionBFM(ProjectVertex, tri, Model_FWH):
    Model_Completion = sio.loadmat('./models/Model_Completion.mat')
    indf_c        = Model_Completion['indf_c']
    indf_c2b      = Model_Completion['indf_c2b'] - 1
    trif_stitch   = Model_Completion['trif_stitch']
    trif_backhead = Model_Completion['trif_backhead']

    muf = np.squeeze(Model_FWH['mu'], axis=0)[0]
    wf = np.squeeze(Model_FWH['w'], axis=0)[0]
    # trif = Model_FWH['tri']

    # print('indf_c2b', indf_c2b)
    ProjectVertex_c2b = np.squeeze(ProjectVertex[indf_c2b, :], axis=1)
    # print(ProjectVertex_c2b)

    [f, phi, gamma, theta, t, alpha] = FittingModel3D_validpoint(ProjectVertex_c2b, Model_FWH, indf_c)
    # print('muf', muf.shape)
    # print('wf', wf.shape)
    # print('alpha', alpha.shape)
    vertexf = muf + np.matmul(wf, alpha)
    vertexf = np.reshape(vertexf , (int(vertexf.shape[0] / 3), 3))

    ProjectVertexf = np.matmul(f * RotationMatrix(phi, gamma, theta), np.transpose(vertexf)) + np.transpose(np.tile(t, (vertexf.shape[0], 1)))
    ProjectVertex_full = np.vstack((ProjectVertex, np.transpose(ProjectVertexf)))
    # print('ProjectVertex_full', ProjectVertex_full.shape)
    tri_full = np.hstack((tri, trif_backhead, trif_stitch))
    tri_full = np.transpose(tri_full)
    # print('tri_full', tri_full.shape)
    #blend
    iteration = 1

    vertex_blend = ProjectVertex_full
    stitch_point = np.unique(trif_stitch)
    # print('stitch_point', stitch_point.shape)
    for iter in range(iteration):
        #iter
        vertex_temp = vertex_blend
        for i in range(stitch_point.shape[0]):
            #i
            ind = stitch_point[i] # blur the ith ind
            # print(ind)
            conn_tri = np.where(tri_full[:, 0] == ind, 1, 0) | np.where(tri_full[:, 1] == ind, 1, 0) | np.where(tri_full[:, 2] == ind, 1, 0)
            conn_tri = np.squeeze(tri_full[np.where(1 == conn_tri), :], axis=0)
            conn_point = np.unique(conn_tri) - 1
            # print(conn_point)
            # print(vertex_blend.shape)
            # print(vertex_blend[conn_point, :].shape)
            # print(np.mean(vertex_blend[conn_point, :], axis=0))
            # assert 0
            vertex_temp[ind - 1, :] = np.mean(vertex_blend[conn_point, :], axis=0)
        vertex_blend = vertex_temp

    ProjectVertex_full = vertex_blend
    # print(ProjectVertex_full)

    return ProjectVertex_full, tri_full

