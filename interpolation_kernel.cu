#include<torch/extension.h>

// trilinear_fw_kernel
template <typename scalar_t>
//__global__ represents the kernel function called on cpu but executed on gpu, is the combination of __host__ and __device__
// other descriptor : __host__ (called and executed on cpu), __device__ (called and executed on gpu)
__global__ void trilinear_fw_kernel(
    // PackedTensorAccessor represents the data can be modified
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats, // 3 means 3 dimensions (N, 8, F)
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,// 2 means 2 dimensions (N, 3)
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp// 2 means 2 dimensions (N, F)
){
    //////////////////////////////////////////////////////////////////////////////////////
    // 计算线程的唯一ID:
    // 图片中显示，我们使用2x1的块结构和16x16的线程结构。
    // 对于n（对应N维度）:
    // blockIdx.x * blockDim.x 计算当前块在x维度的起始索引。
    // + threadIdx.x 为这个块中的线程提供一个唯一的x维度索引。
    // 对于f（对应F维度）:
    // blockIdx.y * blockDim.y 计算当前块在y维度的起始索引。
    // + threadIdx.y 为这个块中的线程提供一个唯一的y维度索引。
        const int n = blockIdx.x * blockDim.x + threadIdx.x;  // Compute the unique x-index based on block and thread ID in the x-dimension
        const int f = blockIdx.y * blockDim.y + threadIdx.y;  // Compute the unique y-index based on block and thread ID in the y-dimension
    // remove unused threads
    if ((n >= feats.size(0)) || (f >= feats.size(2))) return; // If the thread is outside the bounds of the input, exit thread.

    // normalized within the range of [-1, 1]
    const scalar_t u = (points[n][0] + 1) / 2; // distance to x
    const scalar_t v = (points[n][1] + 1) / 2; // distance to y
    const scalar_t w = (points[n][2] + 1) / 2; // distance to z

    //calculate weight
    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = v*w;

    feat_interp[n][f] = (1-u)*(a*feats[n][0][f] + 
                                b*feats[n][1][f] + 
                                c*feats[n][2][f] + 
                                d*feats[n][3][f]) + 
                            u*(a*feats[n][4][f] + 
                                b*feats[n][5][f] + 
                                c*feats[n][6][f] + 
                                d*feats[n][7][f]);
    //   
    //////////////////////////////////////////////////////////////////////////////////////
}


// c++ function trilinear_fw_cu to call the kernel
// for returning multiple tensors, follow this example:
// std::vector<torch::Tensor> trilinear_fw_cu(torch::Tensor feats, torch::Tensor points)
torch::Tensor trilinear_fw_cu(
    torch::Tensor feats,   // feats[N][8][F]: N cubes, 8 points/cube, F features
    torch::Tensor points   // points[N][3]: N points, coords (x, y, z)
){
    // We need to get N and F at first
    // in python for example: N = feats.shape[0]
    const int N = feats.size(0); // size means shape in python
    const int F = feats.size(2);



    // in python : feat_interp = torch.zeros(N, F, dtype=torch.float32, device='cuda:0')
    // options for the same dtype and device as feats
    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());
    // here you can also define the second tensor with the same dtype and device as feats for return the calculation
    /////////////////////////////////////////////////////////
    // options for different dtype and device as points
    // for example: torch::zeros({N,F}, torch::dtype(torch::kInt32).device(feats.device()));

    // cuda parallel computing principle:
    // 1. call kernel from cpu 
    // 2. kernel build grid that includes many blocks on GPU
    // 3. each block has many threads
    // 4. each thread do the same thing but on different data parallelly
    // in our example, each thread do trilinear interpolation on one point with different features
    // what we only need at this time is the number of threads
    // in trilinear interpolation, N and F can be calculated parallelly
    const dim3 threads(16,16); //128-256-512 dim3 is used for calculate multi-dimension data
    // the block must cover all points
    // just like a small square cover all points in a shape.  
    // based on N calculate the number of blocks in x direction, and F in y direction
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);

    // launch kernel
    // normal functions usually have a return value, but kernel functions don't (always void), 
    // so we have to put both input and output into the kernel function
    // packed_accessor is a class to make c++ modify tensor data in kernel function
    // RestrictPtrTraits : all elements won't have overlap with other data
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu",
    ([&] {
        // scalar_t represents the input can be modified, if you know the actual type, you can use it directly 
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(), // 3 means 3 dimensions (N, 8, F)
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),// 2 means 2 dimensions (N, 3)
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>()// 2 means 2 dimensions (N, F)
            // if you want to input a data which is not a tensor, you can define it directly, for example:
            // bool a that you defined with N and F outside it can be defined here with just a
        );
    }));


    return feat_interp; // feat_interp[N][F]: N points, F features
    // if you want to return multiple tensors, follow this example:
    // return {feat_interp, feat_interp2};
} 