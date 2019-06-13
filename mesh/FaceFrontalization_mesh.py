from .FaceFrontalization_algo import FaceFrontalization_core_cython
import numpy as np

def FaceFrontalizationMapping(mask, tri_ind, all_vertex_src, all_vertex_ref, all_tri, bg_tri_num,
                                            valid_tri_half, vertex_length, tri_length, sym_tri_list):

    tri_ind = tri_ind - 1
    all_vertex_src = all_vertex_src - 1
    all_vertex_ref = all_vertex_ref - 1
    all_tri = all_tri - 1
    # sym_tri_list = sym_tri_list

    height, width = mask.shape
    all_ver_dim = all_vertex_ref.shape[0]
    all_ver_length = all_vertex_ref.shape[1]

    all_tri_dim, all_tri_length = all_tri.shape
    symlist_length = sym_tri_list.shape[1]

    mask = mask.astype(np.double).copy()
    tri_ind = tri_ind.astype(np.double).copy()
    all_vertex_src = np.transpose(all_vertex_src).astype(np.double).copy()
    all_vertex_ref = np.transpose(all_vertex_ref).astype(np.double).copy()
    all_tri = np.transpose(all_tri).astype(np.double).copy()
    valid_tri_half = valid_tri_half.astype(np.double).copy()
    sym_tri_list = np.transpose(sym_tri_list).astype(np.double).copy()

    FF = FaceFrontalization_core_cython.PyFaceFrontalizationMapping(mask, tri_ind,
                      width, height, 1, all_vertex_src,
                      all_vertex_ref, all_ver_dim, all_ver_length,
                      all_tri, all_tri_dim, all_tri_length, bg_tri_num,
                      valid_tri_half,
                      vertex_length, tri_length, sym_tri_list, symlist_length)

    corres_map = np.zeros((height, width, 2), dtype=np.double)

    corres_map_sym = np.zeros((height, width, 2), dtype=np.double)


    FF.frontalization_mapping(corres_map, corres_map_sym)

    corres_map = corres_map + 1

    corres_map_sym = corres_map_sym + 1

    return corres_map, corres_map_sym

def FaceFrontalizationFilling(img, corres_map):
    corres_map = corres_map - 1

    height, width, nChannels = img.shape

    img = img.astype(np.double).copy()
    corres_map = corres_map.astype(np.double).copy()

    FF = FaceFrontalization_core_cython.PyFaceFrontalizationFilling(img, width, height, nChannels, corres_map)

    result = np.zeros((height, width, nChannels), dtype=np.double)

    FF.frontalization_filling(result)
    return result
