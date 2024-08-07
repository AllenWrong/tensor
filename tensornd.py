import cffi
import numpy as np


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
Tensor* tensor_slice_keepdim(Tensor* t, int* start, int* end, int* step);
Tensor* tensor_slice(Tensor* t, int* start, int* end, int* step);
float tensor_item(Tensor* t);
int* shape(Tensor* t);
int ndim(Tensor* t);
void tensor_free(Tensor* t);
""")
lib = ffi.dlopen("./libtensornd.so")  # Make sure to compile the C code into a shared library
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# utils

def _get_nparr_data_pointer(np_arr: np.ndarray):
    return ffi.cast('float *', np_arr.ctypes.data)

def _is_item_index(idx):
    if isinstance(idx, tuple):
        return all([isinstance(it, int) for it in idx])
    else:
        return isinstance(idx, int)


def _process_index(idx, shape):
    start, end, step = [], [], []
    for i, it in enumerate(idx):
        if isinstance(it, int):
            start.append(it), end.append(it+1), step.append(1)
        elif isinstance(it, slice):
            start.append(it.start if it.start is not None else 0)
            end.append(it.stop if it.stop is not None else shape[i])
            step.append(it.step if it.step is not None else 1)
        else:
            raise TypeError(f"TypeError: unsupport type '{type(it)}'")
    return start, end, step


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

    def __getitem__(self, index):
        if _is_item_index(index):
            return lib.tensor_getitem(self.data, index)
        else:
            start, end, step = _process_index(index, self.shape)
            c_tensor = lib.tensor_slice(self.data, start, end, step)
            return tensor(c_tensor)

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
    

def empty(shape):
    return Tensor(shape=shape)

def tensor(data):
    return Tensor(data=data)