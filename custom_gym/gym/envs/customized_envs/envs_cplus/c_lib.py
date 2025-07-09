""" some utility for call C++ code"""

from __future__ import absolute_import

import os
import ctypes
import platform
import multiprocessing


def _load_lib(env_name):
    """ Load library in build/lib. """
    cur_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
    lib_path = os.path.join(cur_path, "./build/")
    if platform.system() == 'Darwin':
        path_to_so_file = os.path.join(lib_path, f"lib{env_name}Env.dylib")
    elif platform.system() == 'Linux':
        path_to_so_file = os.path.join(lib_path, f"lib{env_name}Env.so")
    else:
        raise BaseException("unsupported system: " + platform.system())
    lib = ctypes.CDLL(path_to_so_file, ctypes.RTLD_GLOBAL)
    return lib


def as_double_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def as_int32_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def as_bool_c_array(buf):
    """numpy to ctypes array"""
    return buf.ctypes.data_as(ctypes.POINTER(ctypes.c_bool))


if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count() // 2)

