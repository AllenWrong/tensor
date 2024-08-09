#ifndef TENSORND_H
#define TENSORND_H

#include <stdlib.h>

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


// void tensor_setitem(Tensor* t, int* index, float val);
// void tensor_slice_set_item(Tensor* t, index* index, Tensor* t);

// Tensor* tensor_addf(Tensor* t, float val);
// Tensor* tensor_add(Tensor* t1, Tensor* t2);

void tensor_incref(Tensor* t);
void tensor_decref(Tensor* t);
void tensor_free(Tensor* t);

char* tensor_to_string(Tensor* t);
// void tensor_print(Tensor* t);

#endif