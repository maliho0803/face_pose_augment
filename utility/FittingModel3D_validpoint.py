import numpy as np

def FittingModel3D_validpoint(pt3d, Model, valid_ind):

    iteration = 0
    maxiteration = 4

    mu = np.squeeze(Model['mu'], axis=0)[0]
    w = np.squeeze(Model['w'], axis=0)[0]
    sigma = np.squeeze(Model['sigma'], axis=0)[0]
    # tri = Model['tri']
    keypoints = np.transpose(valid_ind)
    keypoints1 = np.vstack((3 * keypoints - 2, np.vstack((3 * keypoints - 1, 3 * keypoints))))
    keypoints1 = np.reshape(np.transpose(keypoints1), (keypoints1.shape[0] * keypoints1.shape[1], 1)) - 1

    alpha = np.zeros((w.shape[1], 1))
    f = 1
    R = np.eye(3)
    t = np.zeros((3, 1))

    mu_key = np.squeeze(mu[keypoints1], axis=1)
    w_key = np.squeeze(w[keypoints1,:], axis=1)

    # Firstly pose and expression
    while(True):
        if (iteration > maxiteration):
            break
        iteration = iteration + 1

        # vertex = mu + w * alpha;
        # vertex = reshape(vertex, 3, length(vertex) / 3);
        # ProjectVertex = f * R * vertex + repmat(t, 1, size(vertex, 2));
        # DrawSolidHead(ProjectVertex, tri);

        # 1.PoseEstimate
        vertex_key = mu_key + np.matmul(w_key, alpha)
        vertex_key = np.reshape(vertex_key, (int(vertex_key.shape[0] / 3), 3))

        # print(pt3d)
        f, R, t = AlignPoints(vertex_key, pt3d)

        # 2. shape fitting
        beta = 3000
        alpha = FittingShape3D(pt3d, f, R, t, mu_key, w_key, sigma, beta)
    # print(R)
    phi, gamma, theta = RotationMatrix2Angle(R)
    # print(phi, gamma, theta)
    return f, phi, gamma, theta, t, alpha

def RotationMatrix2Angle(R):
    # reference: Extracting Euler Angles from a Rotation Matrix, Mike Day
    # if you are interested in this theme, please refer to
    # http://www.mathworks.com/matlabcentral/newsreader/view_thread/160945

    theta1 = np.arctan2(R[1, 2], R[2, 2])
    c2 = np.sqrt(R[0, 0] * R[0, 0] + R[0, 1] * R[0, 1])
    theta2 = np.arctan2(-R[0, 2], c2)
    s1 = np.sin(theta1)
    c1 = np.cos(theta1)
    theta3 = np.arctan2(s1 * R[2, 0] - c1 * R[1, 0], c1 * R[1, 1] - s1 * R[2, 1])

    phi = theta1
    gamma = theta2
    theta = theta3
    return phi, gamma, theta

def FittingShape3D(pt3d, f, R, t, mu, w, sigma, beta):
    # Initialize Shape with Keypoint
    # @input pt3d: Keypoint on the modal;
    # @input pt2d: Keypoint on the image
    # @input keypoints1: Keypoint index on the modal
    # @input R t s: Pose parameter
    # @input beta: Regularization parameter;
    # @input sigma: Shpae's PCA parameter sigma
    # @output alpha: Shape's PCA paramter

    m = pt3d.shape[0]
    n = w.shape[1]
    # print('pt3d', pt3d.shape)

    t3d = t

    s3d = np.reshape(mu, (int(mu.shape[0] / 3), 3))
    s3d = np.matmul(f * R, np.transpose(s3d))
    # print('s3d', s3d.shape)

    t3d = np.tile(t3d, (pt3d.shape[0], 1))
    # print('t3d', t3d.shape)

    w3d = np.zeros((3 * m, n))
    for i in range(n):
        tempdata = np.reshape(w[:, i], (m, 3))
        tempdata3d =  np.matmul(f * R, np.transpose(tempdata))
        tempdata3d = np.reshape(np.transpose(tempdata3d), (tempdata3d.shape[0]*tempdata3d.shape[1], 1))
        w3d[:, i] = np.squeeze(tempdata3d)

    # print('w3d', w3d.shape)
    # optimize the equation
    # 要求目标3D model的关键点经过投影后与2D图像关键点重合，作为形状约束
    # 公式为：优化公式为：||x - T(w * alpha + mu)|| + lambda * alpha' * C * alpha
    # 求极小值得：(w'T'Tw + lambda*C) * alpha = w'T'x - w'T'T*mu 其中w2d = wT

    equationLeft = np.matmul(np.transpose(w3d), w3d) + beta * np.diag(np.squeeze(1 / (sigma * sigma)))
    # print(equationLeft)
    pt3d = np.reshape(pt3d, (pt3d.shape[0] * pt3d.shape[1], 1))
    s3d  = np.reshape(np.transpose(s3d), (s3d.shape[0] * s3d.shape[1], 1))
    t3d = np.reshape(t3d, (t3d.shape[0] * t3d.shape[1], 1))
    equationRight = np.matmul(np.transpose(w3d), (pt3d - s3d - t3d))
    # print(equationRight)
    alpha = np.dot(np.linalg.inv(equationLeft), equationRight)
    # print(alpha)
    return alpha

def AlignPoints(p1, p2):
    n, d = p1.shape
    # print(p1.shape, p2.shape)

    mu1 = np.mean(p1, 0)
    mu2 = np.mean(p2, 0)

    p1_0 = p1 - np.tile(mu1, (n, 1))
    p2_0 = p2 - np.tile(mu2, (n, 1))
    temp1 = p1_0 * p1_0
    sigma1 = np.sum(temp1[:]) / n
    temp2 = p2_0 * p2_0
    sigma2 = np.sum(temp2[:]) / n
    # print(sigma1, sigma2)

    K = np.matmul(np.transpose(p2_0), p1_0) / n

    U, G, V = np.linalg.svd(K)
    S = np.eye(d)

    if np.linalg.det(K) < 0:
        S[d, d] = -1

    R = np.matmul(np.matmul(U, S), V)
    c = np.trace(G * S) / sigma1
    t = mu2 - np.matmul(c * R, mu1)
    # print(R, c , t)
    return c, R, t


