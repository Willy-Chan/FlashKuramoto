// Write your CUDA / ThunderKittens kernel for `kuramoto_velocity` here.
//
// Loop: edit this file, then run
//   uv run --with modal modal run scripts/kuramoto_modal_bench.py
// `scripts/kuramoto_modal_bench.py` ships this file's *text* to a Modal H100
// container, compiles it with `torch.utils.cpp_extension.load_inline`, checks
// it against the reference math at small scale, then benchmarks it against
// the PyTorch baseline (`kuramoto_velocity` in scripts/kuramoto_coupling_op.py)
// at the full ImageNet-64 n16384 scale. Nothing here needs to compile locally.
//
// CONTRACT: define a host function with exactly this name and signature —
// the harness declares and pybind-binds it for you:
//
//   torch::Tensor kuramoto_forward(torch::Tensor theta, torch::Tensor omega, torch::Tensor K);
//
//   theta: (batch, n) fp32, CUDA.  omega: (1, n) fp32.  K: (n, n) fp32.
//   returns (batch, n) fp32 =
//       omega + cos(theta) * (sin(theta) @ K^T) - sin(theta) * (cos(theta) @ K^T)
//   (the algebraic expansion of dtheta_i/dt = omega_i + sum_j K_ij sin(theta_j - theta_i);
//   see kuramoto_velocity / kuramoto_velocity_reference in kuramoto_coupling_op.py
//   for the two equivalent forms and their derivation.)
//
// This is a forward-only comparison — the harness does not wire up a
// backward pass, so it can't replace kuramoto_velocity in training as-is.
//
// ThunderKittens headers are already on the include path (kittens.cuh,
// prototype.cuh), and the full TK repo is cloned into the image at
// /root/ThunderKittens — `modal shell scripts/kuramoto_modal_bench.py` drops
// you into that exact container to browse real tile-based kernels, e.g.
// /root/ThunderKittens/kernels/gemm/educational_h100/level_04.cu (a tiled
// matmul using rt_bf/st_bf register+shared tiles, gl global descriptors, and
// warp::load/store/mma_AB — the primitives to reach for once the naive
// version below is correct and you're ready to tile the two (batch,n)@(n,n)
// matmuls properly instead of doing an O(n^2) sin/cos-recompute per thread).

#include <cuda_runtime.h>
#include <torch/types.h>
#include "kittens.cuh"

__global__ void kuramoto_naive_kernel(
    const float* __restrict__ theta,
    const float* __restrict__ omega,
    const float* __restrict__ K,
    float* __restrict__ out,
    int batch,
    int n
) {
    int b = blockIdx.y;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batch || i >= n) {
        return;
    }

    const float* theta_row = theta + b * n;
    float sin_i = sinf(theta_row[i]);
    float cos_i = cosf(theta_row[i]);

    // Naive O(n) per thread, O(n^2) per row: recomputes sin/cos(theta_j) for
    // every output index i instead of computing them once per j and reusing
    // across i via a proper tiled matmul. Correct, not fast — the fusion
    // target described in kuramoto_coupling_op.py is exactly replacing this
    // loop with two tiled (batch,n)@(n,n) matmuls sharing loaded K tiles.
    float weighted_sin = 0.0f;
    float weighted_cos = 0.0f;
    const float* K_row = K + i * n;  // K[i, :]
    for (int j = 0; j < n; ++j) {
        float s, c;
        sincosf(theta_row[j], &s, &c);
        float k = K_row[j];
        weighted_sin += k * s;
        weighted_cos += k * c;
    }

    out[b * n + i] = omega[i] + cos_i * weighted_sin - sin_i * weighted_cos;
}

torch::Tensor kuramoto_forward(torch::Tensor theta, torch::Tensor omega, torch::Tensor K) {
    TORCH_CHECK(theta.is_cuda() && omega.is_cuda() && K.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(theta.dtype() == torch::kFloat32, "theta must be fp32");
    TORCH_CHECK(theta.dim() == 2, "theta must be (batch, n)");
    theta = theta.contiguous();
    omega = omega.contiguous();
    K = K.contiguous();

    int64_t batch = theta.size(0);
    int64_t n = theta.size(1);
    TORCH_CHECK(K.size(0) == n && K.size(1) == n, "K must be (n, n)");

    auto out = torch::empty({batch, n}, theta.options());

    constexpr int kThreads = 256;
    dim3 grid((n + kThreads - 1) / kThreads, batch);
    kuramoto_naive_kernel<<<grid, kThreads>>>(
        theta.data_ptr<float>(),
        omega.data_ptr<float>(),
        K.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(batch),
        static_cast<int>(n)
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kuramoto_naive_kernel launch failed: ", cudaGetErrorString(err));
    return out;
}
