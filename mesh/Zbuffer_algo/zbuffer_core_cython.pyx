import numpy as np
cimport numpy as np
from libcpp.string cimport string

# use the Numpy-C-API from Cython
np.import_array()

cdef extern from "3DMM.h":
    void ZBuffer(double* vertex, double* tri, double* texture,
        int nver, int ntri,
        double* src_img,
        int width, int height, int nChannels,
        double* img, double* tri_ind)

    void ZBufferTri(double* vertex, double* tri, double* texture_tri,
        int nver, int ntri,
        double* src_img,
        int width, int height, int nChannels,
        double* img, double* tri_ind)

def ZBuffer_python(np.ndarray[double, ndim=2, mode = "c"] vertex not None,
                np.ndarray[double, ndim=2, mode = "c"] tri not None,
                np.ndarray[double, ndim=2, mode="c"] texture not None,
                int nver, int ntri,
                np.ndarray[double, ndim=3, mode = "c"] src_img not None,
                int width, int height, int nChannels,
                np.ndarray[double, ndim=3, mode = "c"] img not None,
                np.ndarray[double, ndim=3, mode = "c"] tri_ind not None,
                ):
    ZBuffer(
        <double*> np.PyArray_DATA(vertex), <double*> np.PyArray_DATA(tri), <double*> np.PyArray_DATA(texture),
        nver, ntri,
        <double*> np.PyArray_DATA(src_img), width, height, nChannels,
        <double*> np.PyArray_DATA(img), <double*> np.PyArray_DATA(tri_ind))

def ZBufferTri_python(np.ndarray[double, ndim=2, mode = "c"] vertex not None,
                np.ndarray[double, ndim=2, mode = "c"] tri not None,
                np.ndarray[double, ndim=2, mode="c"] texture_tri not None,
                int nver, int ntri,
                np.ndarray[double, ndim=3, mode = "c"] src_img not None,
                int width, int height, int nChannels,
                np.ndarray[double, ndim=3, mode = "c"] img not None,
                np.ndarray[double, ndim=3, mode = "c"] tri_ind not None,
                ):
    ZBufferTri(
        <double*> np.PyArray_DATA(vertex), <double*> np.PyArray_DATA(tri), <double*> np.PyArray_DATA(texture_tri),
        nver, ntri,
        <double*> np.PyArray_DATA(src_img), width, height, nChannels,
        <double*> np.PyArray_DATA(img), <double*> np.PyArray_DATA(tri_ind))