import numpy as np

def FittingExpression(pt2d, pt3d_shape, keypoints1, R, t, s, w, sigma, beta):


    s3d = np.matmul(s * R, pt3d_shape)
    s2d = s3d[0:2, :]
    t2d = np.tile(t[0:2], (1, pt2d.shape[1]))

    # Firstly the expression component
    m = pt2d.shape[1]
    n = w.shape[1]

    w2d = np.zeros((2 * m, n))
    for i in range(n):
        tempdata = np.reshape(w[keypoints1, i].T, (m, 3)).T #第i个特征脸的关键点3D坐标
        tempdata2d =  np.matmul(s * R, tempdata) #投影到2D上
        tempdata2d = tempdata2d[0 : 2, :]
        w2d[:, i] = np.squeeze(np.reshape(tempdata2d.T, [-1, 1])) #特征脸上的关键点投影到2D的坐标 w2d = keypoint(T * w)= T * keypoint(w) (T为仿射矩阵，w为特征脸)

    #optimize the equation
    #要求目标3D model的关键点经过投影后与2D图像关键点重合，作为形状约束
    #公式为：优化公式为：||x - T(w * alpha + mu)|| + lambda * alpha' * C * alpha
    #求极小值得：(w'T'Tw + lambda*C) * alpha = w'T'x - w'T'T*mu 其中w2d = wT
    equationLeft = np.dot(w2d.T, w2d )+ beta * np.diagflat(1 / (sigma**2))
    equationRight = np.dot(w2d.T, (np.reshape(pt2d.T, [-1, 1]) - np.reshape(s2d.T, [-1,1]) - np.reshape(t2d.T, [-1, 1])))
    alpha_exp = np.dot(np.linalg.inv(equationLeft), equationRight)

    return alpha_exp

