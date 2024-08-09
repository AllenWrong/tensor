import tensornd
import pytest
from functools import reduce
from operator import mul
import numpy as np
import itertools
import math
import itertools
import random


def _prod(arr):
    return reduce(mul, arr)


def is_close(a, b):
    if isinstance(a, tensornd.Tensor):
        a = a.item()
    return math.isclose(a, b, rel_tol=1e-7)

def tensor_eq_nparr(tensor, nparr):
    if isinstance(tensor, float):
        assert isinstance(nparr, float) and (tensor == nparr)

    assert tensor.shape == nparr.shape

    all_indices = list(itertools.product(*[range(dim) for dim in tensor.shape]))
    for it in all_indices:
        assert is_close(tensor[it].item(), nparr[it])


def generate_slices(shape, num=None):
    """thank chatgpt for writting this function"""
    dims = len(shape)
    all_slices = []

    for indices in itertools.product(*[range(shape[i]+1) for i in range(dims)]):
        slice_obj = []
        for i, index in enumerate(indices):
            if index == 0:
                slice_obj.append(slice(None))
            elif index == shape[i]:
                slice_obj.append(slice(index))
            else:
                slice_obj.append(slice(index-1, index))
        all_slices.append(tuple(slice_obj))

    if num is None:
        num = _prod(shape) // 2
    
    return random.sample(all_slices, num)


@pytest.mark.parametrize("size", [
    [0], [1], [10], [10000],
    [0, 10], [2, 3], [1, 4], [4, 1],
    [32, 32, 3], [1, 64, 64], [64, 128, 786, 786]
])
def test_nelement(size):
    t = tensornd.empty(size)
    assert t.nelement() == _prod(size), f"size={size}"


@pytest.mark.parametrize("size", [
    [1], [10], [10000],
    [2, 3], [1, 4], [4, 1],
    [2, 3, 4], [10, 10, 10], [32, 32, 3],
    [2, 3, 4, 5, 6, 7, 8]
])
def test_getitem(size):
    data = np.random.rand(*size)
    t = tensornd.tensor(data=data)
    tensor_eq_nparr(t, data)


@pytest.mark.parametrize("size", [
    [1], [1, 1, 1], [1, 2], [2, 1]
])
def test_tensor_item(size):
    data = np.random.rand(*size)
    t = tensornd.tensor(data=data)
    if _prod(size) != 1:
        with pytest.raises(ValueError):
            t.item()
    else:
        t.item()


@pytest.mark.parametrize("size", [
    (50,),
])
def test_slice_1d(size):
    data = np.random.rand(*size)
    t = tensornd.tensor(data=data)

    assert is_close(t[10], data[10])
    tensor_eq_nparr(t[:3], data[:3])
    tensor_eq_nparr(t[4:10], data[4:10])
    tensor_eq_nparr(t[10:], data[10:])
    tensor_eq_nparr(t[:], data[:])


@pytest.mark.parametrize("size", [
    (10,10),
])
def test_slice_2d(size):
    data = np.random.rand(*size)
    t = tensornd.tensor(data=data)

    assert is_close(t[0, 0], data[0, 0])
    assert is_close(t[1, 2], data[1, 2])
    assert is_close(t[9, 9], data[9, 9])
    
    tensor_eq_nparr(t[:, :], data[:, :])

    tensor_eq_nparr(t[0, 1:5], data[0, 1:5])
    tensor_eq_nparr(t[9, 3:], data[9, 3:])
    tensor_eq_nparr(t[8, :5], data[8, :5])

    # tensor_eq_nparr(t[2:5, 1], data[2:5, 1])
    # tensor_eq_nparr(t[:6, 2], data[:6, 2])
    # tensor_eq_nparr(t[3:, 4], data[3:, 4])

    # tensor_eq_nparr(t[4:10], data[4:10])
    # tensor_eq_nparr(t[10:], data[10:])
    # tensor_eq_nparr(t[:], data[:])


# import numpy as np

# data = np.random.rand(3)
# print(data)
# t = tensornd.tensor(data)
# t = t[2:]

# print(t.item())
