import cffi
import numpy as np
from typing import Sequence

# -----------------------------------------------------------------------------
ffi = cffi.FFI()
ffi.cdef("""
typedef struct {
    float* data;
    size_t data_size;
    int ref_count;
} Storage;

typedef struct {
    Storage* storage;
    int ndim;
    int* size;
    int* offset;
    int* stride;
    char* repr;
} Tensor;

Tensor* tensor_empty(int* size, int ndim);
size_t nelement(Tensor* t);
void tensor_copy_np(Tensor* t, float* data);
size_t logical_to_physical(Tensor* t, int* idx);
float tensor_getitem(Tensor* t, int* idx);
Tensor* tensor_getitem_astensor(Tensor* t, int* idx);
Tensor* tensor_slice(Tensor* t, int* start, int* end, int* step);
Tensor* tensor_slice_squeeze(Tensor* t, int* start, int* end, int* step, bool* skip);
float tensor_item(Tensor* t);
int* shape(Tensor* t);
int* stride(Tensor* t);
int* offset(Tensor* t);
int ndim(Tensor* t);
void tensor_free(Tensor* t);
char* tensor_to_string(Tensor* t);
""")
lib = ffi.dlopen("./libtensornd.so")  # Make sure to compile the C code into a shared library
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# const

IS_INT = 0
ONLY_INT = 1
ONLY_SLICE = 2
SLICE_AND_INT = 3


# -----------------------------------------------------------------------------
# utils

def _get_nparr_data_pointer(np_arr: np.ndarray):
    return ffi.cast('float *', np_arr.ctypes.data)

def _tuple_it(it):
    if not isinstance(it, Sequence):
        return (it,)
    return it

def _index_type(index, ndim):
    # only return index type
    if isinstance(index, int):
        return IS_INT
    elif isinstance(index, slice):
        return ONLY_SLICE
    else:
        assert isinstance(index, Sequence), \
        f"ValueError: type(index)={type(index)} must be one of int or Sequence."
        
        if all([isinstance(it, int)] for it in index) and len(index) == ndim:
            return ONLY_INT
        elif all([isinstance(it, slice)] for it in index):
            return ONLY_SLICE
        else:
            return SLICE_AND_INT

def _process_index(idx, shape, index_type):
    """ensure the returned value is valid"""
    def _format_index():
        start, end, step, sequeeze = [], [], [], []
        for i, it in enumerate(_tuple_it(idx)):
            if isinstance(it, int):
                start.append(it), end.append(it+1), step.append(1)
                sequeeze.append(True)
            elif isinstance(it, slice):
                start.append(it.start if it.start is not None else 0)
                end.append(it.stop if it.stop is not None else shape[i])
                step.append(it.step if it.step is not None else 1)
                sequeeze.append(False)
            else:
                raise TypeError(f"TypeError: unsupport type '{type(it)}'")
        
        for i in range(len(start), len(shape)):
            start.append(0)
            end.append(shape[i])
            step.append(1)
            sequeeze.append(False)
        
        return start, end, step, sequeeze
    
    if index_type == IS_INT:
        return _format_index()
        
    elif index_type == ONLY_SLICE:
        return _format_index()
        
    elif index_type == ONLY_INT:
        return _format_index()
    
    elif index_type == SLICE_AND_INT:
        return _format_index()


class Tensor:
    def __init__(self, shape=None, data=None) -> None:
        assert shape is not None or data is not None, "please specify shape or data!"

        if data is not None:
            if isinstance(data, list):
                raise NotImplementedError()
            elif isinstance(data, tuple):
                raise NotImplementedError()
            elif isinstance(data, ffi.CData):
                self.data = data
            elif isinstance(data, np.ndarray):
                data = data.astype('float32')
                data.shape
                self.data = lib.tensor_empty(data.shape, data.ndim)
                if data.data.contiguous:
                    lib.tensor_copy_np(self.data, _get_nparr_data_pointer(data))
                else:
                    lib.tensor_copy_np(self.data, _get_nparr_data_pointer(np.ascontiguousarray(data)))
            else:
                raise NotImplementedError()

        else:
            self.data = lib.tensor_empty(shape, len(shape))

    def __str__(self) -> str:
        tensor_str_c = lib.tensor_to_string(self.data)
        tensor_str_py = ffi.string(tensor_str_c).decode('utf-8')
        return tensor_str_py
    
    def __del__(self):
        if lib is not None:
            if hasattr(self, 'data'):
                lib.tensor_free(self.data)

    def _get_item_1d(self, index):
        return lib.tensor_getitem_astensor(self.data, index)

    def __getitem__(self, index):
        if index is None:
            return self
        
        if _index_type(index, self.ndim) == IS_INT:
            start, end, stop, sequeeze = _process_index(index, self.shape, IS_INT)
            c_tensor = lib.tensor_slice_squeeze(self.data, start, end, stop, sequeeze)
            return tensor(c_tensor)
        
        elif _index_type(index, self.ndim) == ONLY_SLICE:
            start, end, step, _ = _process_index(index, self.shape, ONLY_SLICE)
            c_tensor = lib.tensor_slice(self.data, start, end, step)
            return tensor(c_tensor)
        
        elif _index_type(index, self.ndim) == ONLY_INT:
            start, end, stop, sequeeze = _process_index(index, self.shape, ONLY_INT)
            c_tensor = lib.tensor_slice_squeeze(self.data, start, end, stop, sequeeze)
            return tensor(c_tensor)
        
        elif _index_type(index, self.ndim) == SLICE_AND_INT:
            start, end, stop, sequeeze = _process_index(index, self.shape, ONLY_INT)
            c_tensor = lib.tensor_slice_squeeze(self.data, start, end, stop, sequeeze)
            return tensor(c_tensor)
        else:
            raise ValueError("value error in _index_type return")

    def nelement(self) -> int:
        return lib.nelement(self.data)
        
    def item(self) -> float:
        if self.nelement() != 1:
            raise ValueError("ValueError: can only convert an array of size 1 to a Python scalar")
        return lib.tensor_item(self.data)
    
    @property
    def ndim(self):
        return lib.ndim(self.data)
    
    @property
    def shape(self):
        return tuple(ffi.cast('int*', lib.shape(self.data))[i] for i in range(self.ndim))
    
    @property
    def stride(self):
        return tuple(ffi.cast('int*', lib.stride(self.data))[i] for i in range(self.ndim))
    
    @property
    def offset(self):
        return tuple(ffi.cast('int*', lib.offset(self.data))[i] for i in range(self.ndim))

def empty(shape):
    return Tensor(shape=shape)

def tensor(data):
    return Tensor(data=data)