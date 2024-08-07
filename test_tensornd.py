import tensornd
import pytest
from functools import reduce
from operator import mul
import numpy as np
import itertools
import math


def _prod(arr):
    return reduce(mul, arr)


def tensor_eq_nparr(tensor, nparr):
    assert tensor.shape == nparr.shape
    all_indices = list(itertools.product(*[range(dim) for dim in tensor.shape]))
    for it in all_indices:
        assert math.isclose(tensor[it], nparr[it], rel_tol=1e-7)


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


# import numpy as np

# data = np.random.rand(2, 3).astype('float32')
# print(data.dtype)
# print(data[1, 1])
# t = tensornd.Tensor(data=data)
# print(t.ndim)
# print(t.shape)
# print(t.nelement())
# a = t[1, :]

# print(type(a))
# print(a.shape)
