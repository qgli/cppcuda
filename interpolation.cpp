#include<torch/extension.h>
// call the cuda kernel and calculate the trilinear interpolation parallely
#include "utils.h"

torch::Tensor trilinear_interpolation(
    torch::Tensor feats,   // feats[N][8][F]: N cubes, 8 points/cube, F features
    torch::Tensor points   // points[N][3]: N points, coords (x, y, z)
){
    CHECK_INPUT(feats);
    CHECK_INPUT(points);
    // check whether the input tensors are cuda tensors and contiguous
    return trilinear_fw_cu(feats, points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("trilinear_interpolation", &trilinear_interpolation); 
    //python function name, c++ function name
}