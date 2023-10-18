// For defining cuda functions be called in interpolation.cpp 
#include <torch/extension.h>

// check whether a tensor is cuda tensor
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
// check flattened tensor is contiguous on the memory
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
// run the two check functions above
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// if you want to return multiple tensors, modify here as well.
torch::Tensor trilinear_fw_cu(
    torch::Tensor feats, // feats (N (num of cubes), 8 (8 points per cube), F (num of features))
    torch::Tensor points // point (N (num of cubes/points), 3 (x, y, z)))
);