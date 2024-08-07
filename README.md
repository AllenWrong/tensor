# tensor

**Inspired by Andrej Karpathy's implementation of 1D tensors,  I implemented n-dimensional tensors. Since I am not sure whether a PR (pull request) is needed, I did not submit a PR to his repository. To maintain the integrity of a single file, I did not reuse some code; instead, I copied it directly.**

[`tensor1d.c` readme](https://github.com/EurekaLabsAI/tensor)


**currently support**

- create tensor
  - create empty tensor using shape
  - create tensor using np.ndarray
- tensor getitem
  - get a float item
- tensor nelement
- tensor.item()
- get tensor meta data
  - ndim
  - shape


**currently testing**

- slice tensor
- slice tensor keepdim


**TODOs:**

- feature
  - create tensor using list, tuple
  - get tensor item as tensor
  - slice tensor
  - slice tensor keepdim
  - tensor_setitem()
  - tensor_slice_setitem()
- make tests better
- make the code more readable and cleaner
- write a post to explain my implementation and help greeners approach this comfortable


Good related resources:
- [PyTorch internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [Numpy paper](https://arxiv.org/abs/1102.1523)

### License

MIT