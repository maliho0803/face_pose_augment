from .Zbuffer_algo import zbuffer_core_cython
import numpy as np

def ZBufferTri(projectedVertex, tri, texture_tri, img_src):
    projectedVertex = projectedVertex - 1
    tri = tri - 1
    height, width, nChannels = img_src.shape
    nver = projectedVertex.shape[1]
    ntri = tri.shape[1]
    projectedVertex = np.transpose(projectedVertex).astype(np.double).copy()
    tri = np.transpose(tri).astype(np.double).copy()
    texture_tri = texture_tri.astype(np.double).copy()
    img_src = img_src.astype(np.double).copy()

    img = np.zeros((height, width, nChannels), dtype=np.double)
    tri_ind = np.zeros((height, width, 1), dtype=np.double)

    zbuffer_core_cython.ZBufferTri_python(projectedVertex, tri, texture_tri, nver, ntri, img_src, width, height, nChannels, img, tri_ind)
    tri_ind = tri_ind + 1
    img = np.squeeze(img, axis=2)
    tri_ind = np.squeeze(tri_ind, axis=2)
    return img, tri_ind


def ZBuffer(projectedVertex, tri, texture, img_src):
    projectedVertex = projectedVertex - 1
    tri = tri - 1
    height, width, nChannels = img_src.shape
    nver = projectedVertex.shape[1]
    ntri = tri.shape[1]
    projectedVertex = np.transpose(projectedVertex).astype(np.double).copy()
    tri = np.transpose(tri).astype(np.double).copy()
    texture = texture.astype(np.double).copy()
    img_src = img_src.astype(np.double).copy()

    img = np.zeros((height, width, nChannels), dtype=np.double)
    tri_ind = np.zeros((height, width, 1), dtype=np.double)

    zbuffer_core_cython.ZBuffer_python(projectedVertex, tri, texture, nver, ntri, img_src, width, height, nChannels, img, tri_ind)
    tri_ind = tri_ind + 1
    img = np.squeeze(img, axis=2)
    tri_ind = np.squeeze(tri_ind, axis=2)
    return img, tri_ind