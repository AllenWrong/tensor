/*
Inspired by Andrej Karpathy's implementation of 1D tensors, 
I implemented n-dimensional tensors. 

Since I am not sure whether a PR (pull request) is needed, 
I did not submit a PR to his repository.

To maintain the integrity of a single file, 
I did not reuse some code; instead, I copied it directly.
-----

Compile and run like:
gcc -Wall -O3 tensornd.c -o tensornd && ./tensornd

Or create .so for use with cffi:
gcc -O3 -shared -fPIC -o libtensornd.so tensornd.c
*/

#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include "tensornd.h"
#include <string.h>


// ----------------------------------------------------------------------------
// memory allocation

void *malloc_check(size_t size, const char *file, int line) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "Error: Memory allocation failed at %s:%d\n", file, line);
        exit(EXIT_FAILURE);
    }
    return ptr;
}
#define mallocCheck(size) malloc_check(size, __FILE__, __LINE__)

#define myAssert(condition, message) \
    do { \
        if (!(condition)) { \
            fprintf(stderr, "Assertion failed: (%s), file %s, line %d: %s\n", \
                    #condition, __FILE__, __LINE__, message); \
            abort(); \
        } \
    } while (0)

// ----------------------------------------------------------------------------
// utils

size_t _prod(int* size, int ndim, int start) {
    myAssert(start <= ndim, "ValueError: start should < ndim");
    size_t result = 1;
    for (int i = start; i < ndim; i++) {
        result *= size[i];
    }
    return result;
}

void _set_stride(int* stride, int* size, int ndim) {
    stride[ndim-1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        stride[i] = size[i+1] * stride[i+1];
    }
}

void _malloc_size_offset_stride(Tensor* t) {
    t->size = mallocCheck(t->ndim * sizeof(int));
    t->offset = mallocCheck(t->ndim  * sizeof(int));
    t->stride = mallocCheck(t->ndim  * sizeof(int));
}

int ceil_div(int a, int b) {
    // integer division that rounds up, i.e. ceil(a / b)
    return (a + b - 1) / b;
}

// ----------------------------------------------------------------------------
// Storage: simple array of floats. The memory where the data is actually stored.

Storage* storage_new(size_t size) {
    Storage* storage = mallocCheck(sizeof(Storage));
    storage->data_size = size;
    storage->ref_count = 1;
    storage->data = mallocCheck(size * sizeof(float));
    return storage;
}

float storage_getitem(Storage* s, size_t idx) {
    myAssert(s->data_size, "storage_getitem idx out of range");
    return s->data[idx];
}

void storage_setitem(Storage* s, size_t idx, float val) {
    myAssert(idx < s->data_size, "storage_setitem idx out of range");
    s->data[idx] = val;
}

void storage_incref(Storage* s) {
    s->ref_count++;
}

void storage_decref(Storage* s) {
    s->ref_count--;
    if (s->ref_count == 0) {
        free(s->data);
        free(s);
    }
}

void storage_share(Tensor* dest, Tensor* src) {
    dest->storage = src->storage;
    storage_incref(src->storage);
}

// ----------------------------------------------------------------------------
// Tensor class functions

// torch.empty(size)
Tensor* tensor_empty(int* size, int ndim) {
    Tensor* t = mallocCheck(sizeof(Tensor));
    size_t data_size = _prod(size, ndim, 0);
    t->storage = storage_new(data_size);
    // init meta data
    t->ndim = ndim;
    t->idx = 0;
    // malloc mem for meta data
    _malloc_size_offset_stride(t);
    memcpy(t->size, size, ndim * sizeof(int));
    memset(t->offset, 0, ndim * sizeof(int));
    _set_stride(t->stride, t->size, t->ndim);
    t->repr = NULL;

    return t;
}

// tensor_a.nelement()
size_t nelement(Tensor* t) {
    return _prod(t->size, t->ndim, 0);
}

// copy numpy array to tensor
void tensor_copy_np(Tensor* t, float* data) {
    memcpy(t->storage->data, data, t->storage->data_size * sizeof(float));
}

// convert index
// TODO: write a blog to explain this.
size_t logical_to_physical(Tensor* t, int* idx) {
    int step = 1;
    size_t physical_idx = 0;
    for (int i = 0; i < t->ndim; i++) {
        physical_idx += t->stride[i] * (idx[i] * step + t->offset[i]);
    }
    return physical_idx;
}

// get a single value.
float tensor_getitem(Tensor* t, int* idx) {
    size_t physical_idx = logical_to_physical(t, idx);
    myAssert(physical_idx < t->storage->data_size, "index out of range!");
    return storage_getitem(t->storage, physical_idx);
}

// for slice like: tensor_t[1, 1, 1]
// index contains only integer
Tensor* tensor_getitem_astensor(Tensor* t, int* idx) {
    return NULL;
}

// for slice like: tensor_t[1:2, 3:10]
// index contains only slice
Tensor* tensor_slice(Tensor* t, int* start, int* end, int* step) {
    // create new slice tensor
    Tensor* slice_t = mallocCheck(sizeof(Tensor));
    storage_share(slice_t, t);
    slice_t->ndim = t->ndim;
    slice_t->repr = NULL;

    _malloc_size_offset_stride(slice_t);

    for (int i = 0; i < t->ndim; i++) {
        slice_t->size[i] = ceil_div(end[i] - start[i], step[i]);
        slice_t->offset[i] = t->stride[i] * start[i] + t->offset[i];
        slice_t->stride[i] = t->stride[i] * step[i];
    }
    return slice_t;
}

// for slice like: tensor_t[0, 1:2, 4:10]
// index contains slice and integer
Tensor* tensor_slice_squeeze(Tensor* t, int* start, int* end, int* step, bool* skip) {
    // create new slice tensor
    Tensor* slice_t = mallocCheck(sizeof(Tensor));
    storage_share(slice_t, t);

    int size[t->ndim];
    int offset[t->ndim];
    int stride[t->ndim];
    
    int slice_t_ndim = 0;
    int prev_offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (skip[i]) {
            prev_offset = t->stride[i] * start[i] + t->offset[i];
            size[slice_t_ndim] = ceil_div(end[i] - start[i], step[i]);
        } else {
            size[slice_t_ndim] = ceil_div(end[i] - start[i], step[i]);
            offset[slice_t_ndim] = t->stride[i] * start[i] + t->offset[i];
            stride[slice_t_ndim] = t->stride[i] * step[i];

            if (prev_offset != 0) {
                offset[slice_t_ndim] += prev_offset;
                prev_offset = 0;
            }
            slice_t_ndim += 1;
        }
    }

    slice_t->ndim = slice_t_ndim;
    // for 0d tensor
    if (slice_t->ndim == 0) {
        slice_t->idx = logical_to_physical(t, start);
    }
    _malloc_size_offset_stride(slice_t);
    memcpy(slice_t->size, size, slice_t_ndim * sizeof(int));
    memcpy(slice_t->offset, offset, slice_t_ndim * sizeof(int));
    memcpy(slice_t->stride, stride, slice_t_ndim * sizeof(int));
    return slice_t;
}

void tensor_free(Tensor* t) {
    storage_decref(t->storage);
    free(t->size);
    free(t->offset);
    free(t->stride);
    t->repr == NULL ? 0 : free(t->repr);
    free(t);
}

float tensor_item(Tensor* t) {
    if (t->ndim == 0) {
        return storage_getitem(t->storage, t->idx);
    }

    if (nelement(t) != 1) {
        fprintf(stderr, "ValueError: can only convert an array of size 1 to a Python scalar\n");
        return NAN;
    }

    int idx[t->ndim];
    for (int i = 0; i < t->ndim; i++) {
        idx[i] = 0;
    }
    return tensor_getitem(t, idx);
}

// ----------------------------------------------------------------------------
// Tensor metadata

int* shape(Tensor* t) {
    return t->size;
}

int* stride(Tensor* t) {
    return t->stride;
}

int* offset(Tensor* t) {
    return t->offset;
}

int ndim(Tensor* t) {
    return t->ndim;
}


// ----------------------------------------------------------------------------
// temp stuff

// chatgpt write this function.
// just a helper function. help us to see the data.
char* tensor_to_string(Tensor* t) {
    if (!t || !t->storage || !t->storage->data) {
        return strdup("Tensor is null or empty");
    }

    // Determine the total number of elements and check dimensions
    size_t num_elements = nelement(t);
    int rows = t->size[0];
    int cols = t->ndim > 1 ? t->size[1] : 1;
    int total_elements = rows * cols;
    int max_rows = 6;
    int max_cols = 6;
    
    // Allocate buffer for the string representation
    size_t buffer_size = 1024; // Initial size
    char *buffer = malloc(buffer_size);
    if (!buffer) {
        return strdup("Memory allocation failed");
    }
    buffer[0] = '\0'; // Initialize the buffer

    // Print tensor shape
    strcat(buffer, "Tensor(");
    for (int i = 0; i < t->ndim; i++) {
        char dim_str[50];
        snprintf(dim_str, sizeof(dim_str), "%d", t->size[i]);
        strcat(buffer, dim_str);
        if (i < t->ndim - 1) {
            strcat(buffer, ", ");
        }
    }
    strcat(buffer, ")\n");

    // Check if we need to truncate the output
    int should_truncate = (rows > max_rows || cols > max_cols);

    // Print the tensor data
    size_t count = 0;
    size_t index = 0;
    int start_row = 0;
    int end_row = rows;
    int start_col = 0;
    int end_col = cols;

    if (should_truncate) {
        if (rows > max_rows) {
            end_row = max_rows;
        }
        if (cols > max_cols) {
            end_col = max_cols;
        }
    }

    // Print rows
    for (int r = 0; r < rows; ++r) {
        if (r >= start_row && r < end_row) {
            strcat(buffer, "[");
            for (int c = 0; c < cols; ++c) {
                if (c >= start_col && c < end_col) {
                    if (index >= num_elements) break;
                    char value_str[50];
                    snprintf(value_str, sizeof(value_str), c == end_col - 1 ? "%.6f" : "%.6f, ", t->storage->data[index]);
                    strcat(buffer, value_str);
                    ++index;
                } else if (should_truncate && c == end_col) {
                    strcat(buffer, "... ");
                }
            }
            strcat(buffer, "]\n");
        } else if (should_truncate && r == end_row) {
            strcat(buffer, "...\n");
            break;
        } else {
            index += cols;
        }
    }
    return buffer;
}
