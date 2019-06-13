#from FaceFrontalization cimport FaceFrontalization
import numpy as np
cimport numpy as np
from libcpp.string cimport string

# use the Numpy-C-API from Cython
np.import_array()

#definition

cdef extern from "FaceFrontalization.h":
    cdef cppclass FaceFrontalization:
        FaceFrontalization() except +
        FaceFrontalization(double* mask, double* tri_ind, int, int, int,
						double* all_vertex_src, double* all_vertex_ref, int, int,
						double* all_tri, int, int,  int,
						double* valid_tri_half, int, int,
						double* sym_tri_list, int) except +

        FaceFrontalization(double* mask, double* tri_ind, int, int, int,
						double* all_vertex_src, double* all_vertex_ref, int, int,
						double* all_tri, int, int,  int, int,
						double* valid_tri_half, int, int,
						double* sym_list, int) except +

        FaceFrontalization(double* img, int, int, int, double* corres_map) except +

        void frontalization_mapping(double * corres_map, double * corre_map_sym)
        void frontalization_mapping_big_tri(double * corres_map, double * corre_map_sym)
        void frontalization_filling(double * result)

# PyFaceFrontalizationFilling implementation
cdef class PyFaceFrontalizationFilling:
    cdef FaceFrontalization FF

    def __cinit__(self, np.ndarray[double, ndim=3, mode = "c"] img not None,
                       int width, int height, int nChannels,
                       np.ndarray[double, ndim=3, mode = "c"] corres_map not None):
        self.FF = FaceFrontalization(<double*> np.PyArray_DATA(img), width, height, nChannels, <double*> np.PyArray_DATA(corres_map))

    def frontalization_filling(self, np.ndarray[double, ndim=3, mode = "c"] result not None):
        self.FF.frontalization_filling(<double*> np.PyArray_DATA(result))

# PyFaceFrontalizationMappingBigTri implementation
cdef class PyFaceFrontalizationMappingBigTri:
    cdef FaceFrontalization FF

    def __cinit__(self, np.ndarray[double, ndim=2, mode = "c"] mask not None,
                      np.ndarray[double, ndim=2, mode = "c"] tri_ind not None,
                      int width, int height, int nChannels,
                      np.ndarray[double, ndim=2, mode = "c"] all_vertex_src not None,
                      np.ndarray[double, ndim=2, mode = "c"] all_vertex_ref not None,
                      int all_ver_dim, int all_ver_length,
                      np.ndarray[double, ndim=2, mode = "c"] all_tri not None,
                      int all_tri_dim, int all_tri_length,  int bg_tri_num, int bg_vertex_num,
                      np.ndarray[double, ndim=2, mode = "c"] valid_tri_half not None,
                      int vertex_length, int tri_length,
                      np.ndarray[double, ndim=2, mode = "c"] sym_tri_list not None, int symlist_length):
        self.FF = FaceFrontalization(<double*> np.PyArray_DATA(mask), <double*> np.PyArray_DATA(tri_ind), width, height, nChannels,
						<double*> np.PyArray_DATA(all_vertex_src), <double*> np.PyArray_DATA(all_vertex_ref), all_ver_dim, all_ver_length,
						<double*> np.PyArray_DATA(all_tri), all_tri_dim, all_tri_length, bg_tri_num, bg_vertex_num,
						<double*> np.PyArray_DATA(valid_tri_half), vertex_length, tri_length,
						<double*> np.PyArray_DATA(sym_tri_list), symlist_length)

    def frontalization_mapping_big_tri(self, np.ndarray[double, ndim=2, mode = "c"] corres_map not None,
                                      np.ndarray[double, ndim=2, mode = "c"] corres_map_sym not None):
        self.FF.frontalization_mapping_big_tri(<double*> np.PyArray_DATA(corres_map), <double*> np.PyArray_DATA(corres_map_sym))

# PyFaceFrontalizationMapping implementation
cdef class PyFaceFrontalizationMapping:
    cdef FaceFrontalization FF

    def __cinit__(self, np.ndarray[double, ndim=2, mode = "c"] mask not None,
                      np.ndarray[double, ndim=2, mode = "c"] tri_ind not None,
                      int width, int height, int nChannels,
                      np.ndarray[double, ndim=2, mode = "c"] all_vertex_src not None,
                      np.ndarray[double, ndim=2, mode = "c"] all_vertex_ref not None,
                      int all_ver_dim, int all_ver_length,
                      np.ndarray[double, ndim=2, mode = "c"] all_tri not None,
                      int all_tri_dim, int all_tri_length,  int bg_tri_num,
                      np.ndarray[double, ndim=2, mode = "c"] valid_tri_half not None,
                      int vertex_length, int tri_length,
                      np.ndarray[double, ndim=2, mode = "c"] sym_list not None, int symlist_length):
        self.FF = FaceFrontalization(<double*> np.PyArray_DATA(mask), <double*> np.PyArray_DATA(tri_ind), width, height, nChannels,
						<double*> np.PyArray_DATA(all_vertex_src), <double*> np.PyArray_DATA(all_vertex_ref), all_ver_dim, all_ver_length,
						<double*> np.PyArray_DATA(all_tri), all_tri_dim, all_tri_length,  bg_tri_num,
						<double*> np.PyArray_DATA(valid_tri_half), vertex_length, tri_length,
						<double*> np.PyArray_DATA(sym_list), tri_length)

    def frontalization_mapping(self, np.ndarray[double, ndim=3, mode = "c"] corres_map not None,
                                  np.ndarray[double, ndim=3, mode = "c"] corres_map_sym not None):
        self.FF.frontalization_mapping(<double*> np.PyArray_DATA(corres_map), <double*> np.PyArray_DATA(corres_map_sym))