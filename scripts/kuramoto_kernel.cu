// FlashKuramoto: fused tiled phase-velocity kernel, written with ThunderKittens.
//
// Loop: edit this file, then run
//   uv run --with modal modal run scripts/kuramoto_modal_bench.py
// (ships this file's text to a Modal GPU container, compiles it with
// load_inline, checks vs the reference math, benchmarks vs the PyTorch
// baseline in scripts/kuramoto_coupling_op.py).
//
// CONTRACT (declared + pybind-bound by the harness):
//   torch::Tensor kuramoto_forward(torch::Tensor theta, torch::Tensor omega, torch::Tensor K);
//   theta: (batch, n) fp32.  omega: (1, n) fp32.  K: (n, n) fp32.
//   returns (batch, n) fp32 =
//       omega + cos(theta) * (sin(theta) @ K^T) - sin(theta) * (cos(theta) @ K^T)
//
// ----------------------------------------------------------------------------
// Algorithm (FlashKuramoto, flash-attention-style tiling):
//
//   for batch block b (grid)                    x  receiver block r (grid):
//     acc_S = acc_C = 0                                  // fp32 tensor-core accumulators
//     for sender block s (inner loop):                   // streamed through SRAM
//       load theta[b, s], K[r, s]  (K tile loaded ONCE)
//       S_s, C_s = sin/cos(theta_s)                      // inline, registers
//       acc_S += S_s @ K_rs^T                            // tensor cores
//       acc_C += C_s @ K_rs^T                            //   (same K tile, used twice+)
//     epilogue (registers, single global write):
//       V[b, r] = omega_r + cos(theta_r) * acc_S - sin(theta_r) * acc_C
//
// Precision scheme — why the MMAs are bf16 "hi/lo" split:
//   The harness requires max |out - fp32 reference| < 1e-3.  A plain bf16 or
//   tf32 tensor-core MMA rounds the operands to 8/10 mantissa bits, and the
//   n-term reduction random-walks that to ~5e-3 even at n=64 (measured) — it
//   fails the check.  Instead each MMA operand x is split exactly as
//       x = hi + lo,   hi = bf16(x),   lo = bf16(x - hi)
//   and each product is computed with 3 MMAs (hi*hi + hi*lo + lo*hi), which
//   carries ~17 mantissa bits through the tensor cores: measured error ~1e-5,
//   comfortably inside the gate, while still running on the bf16 tensor
//   pipes (>= 2x fp32 FLOPs on L4, ~7x on H100).  The dropped lo*lo term is
//   O(2^-18) relative — negligible.
//   Net: 6 MMAs per K tile (3 per accumulator), and the K tile is loaded from
//   HBM/SRAM once for all 6 — the algorithm's "load once, multiply twice"
//   bandwidth win, amplified.
//
//   K's hi/lo split does not depend on theta, so it is precomputed once per
//   call by a tiny elementwise kernel (one read + one write of K) instead of
//   being redone inside every tile iteration.  theta's sin/cos + split IS
//   done inline in registers, per the algorithm.
//
// Tiling (per 256-thread block, 8 warps):
//   batch block  = 128 rows  (16 rows per warp; batch is zero-padded to 128)
//   receiver blk = BR = 64 columns of V  (= 64 rows of K)
//   sender blk   = BS = 32, streamed; K_hi/K_lo tiles staged in shared memory
//   Grid is 1D with a grouped swizzle (like Triton's grouped matmul) so K row
//   panels and theta stripes get L2 reuse across blocks.
// ----------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include "kittens.cuh"

using namespace kittens;

constexpr int BM = 16;                     // batch rows per warp
constexpr int NWARPS = 8;                  // warps per block
constexpr int BM_BLOCK = BM * NWARPS;      // batch rows per block = 128
constexpr int BR = 64;                     // receiver chunk (output cols per block)
constexpr int BS = 32;                     // sender chunk (reduction tile)
constexpr int GROUP_R = 8;                 // L2-reuse swizzle group size

using g_f32 = gl<float, 1, 1, -1, -1>;
using g_bf  = gl<bf16, 1, 1, -1, -1>;

struct fk_globals {
    g_f32 theta;   // (batch_padded, n)
    g_f32 omega;   // (1, n)
    g_bf  K_hi;    // (n, n) bf16 high part of K
    g_bf  K_lo;    // (n, n) bf16 low (residual) part of K
    g_f32 out;     // (batch_padded, n)
    int n;
    int batch_padded;
};

// --- exact 2-term bf16 split of an fp32 array: x = hi + lo (+ O(2^-18 |x|)) ---
__global__ void split_bf16_kernel(
    const float* __restrict__ x,
    bf16* __restrict__ hi,
    bf16* __restrict__ lo,
    long long size
) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        float v = x[i];
        bf16 h = __float2bfloat16(v);
        // v and float(h) agree to 8 bits, so this subtraction is exact.
        float r = v - __bfloat162float(h);
        hi[i] = h;
        lo[i] = __float2bfloat16(r);
    }
}

// Splits a fp32 register tile into bf16 hi/lo register tiles (mma operands).
template<int R, int C>
__device__ static inline void split_rt(
    rt_bf<R, C> &hi, rt_bf<R, C> &lo, const rt_fl<R, C> &src, rt_fl<R, C> &scratch
) {
    warp::copy(hi, src);          // fp32 -> bf16, round-to-nearest
    warp::copy(scratch, hi);      // exact widening back to fp32
    warp::sub(scratch, src, scratch);  // exact residual
    warp::copy(lo, scratch);
}

__global__ __launch_bounds__(NWARPS * WARP_THREADS)
void flash_kuramoto_kernel(const __grid_constant__ fk_globals g) {
    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<BR, BS> &Khi_s = al.allocate<st_bf<BR, BS>>();
    st_bf<BR, BS> &Klo_s = al.allocate<st_bf<BR, BS>>();

    using loader = kittens::group<NWARPS>;

    // Grouped 1D->2D block swizzle: sweep all batch blocks for a small group
    // of receiver blocks before moving on, so the group's K row panels stay
    // hot in L2 across the batch sweep and each theta stripe is shared by the
    // group's receiver blocks.
    const int num_r = g.n / BR;
    const int num_b = g.batch_padded / BM_BLOCK;
    const int pid = blockIdx.x;
    const int num_in_group = GROUP_R * num_b;
    const int first_r = (pid / num_in_group) * GROUP_R;
    const int group_size = min(num_r - first_r, GROUP_R);
    const int r_blk = first_r + (pid % group_size);
    const int b_blk = (pid % num_in_group) / group_size;

    // This warp's 16-row batch stripe, in 16-row units.
    const int warp_row = b_blk * NWARPS + kittens::warpid();

    // Tensor-core fp32 accumulators (algorithm lines 8-9), kept in registers
    // across the whole sender sweep.
    rt_fl<BM, BR> acc_S, acc_C;
    warp::zero(acc_S);
    warp::zero(acc_C);

    rt_bf<BM, BS> s_hi, s_lo, c_hi, c_lo;
    rt_bf<BR, BS> k_reg;

    // --- inner loop: stream sender chunks (algorithm lines 11-19) ---
    const int num_s = g.n / BS;
    for (int s = 0; s < num_s; s++) {
        // K_rs tiles: cooperative global -> shared load, once per block, then
        // consumed by all 8 warps x 6 MMAs.
        loader::load(Khi_s, g.K_hi, {0, 0, r_blk, s});
        loader::load(Klo_s, g.K_lo, {0, 0, r_blk, s});

        // Sender phases -> sin/cos -> bf16 hi/lo split, inline in registers.
        rt_fl<BM, BS> th, f, scratch;
        warp::load(th, g.theta, {0, 0, warp_row, s});
        warp::apply(f, th, [](int, int, float x) { return __sinf(x); });
        split_rt(s_hi, s_lo, f, scratch);
        warp::apply(f, th, [](int, int, float x) { return __cosf(x); });
        split_rt(c_hi, c_lo, f, scratch);

        __syncthreads();  // K tiles staged

        // Six tensor-core MMAs against ONE staged K tile pair:
        //   acc += hi@hi^T + lo@hi^T + hi@lo^T   (per accumulator)
        warp::load(k_reg, Khi_s);
        warp::mma_ABt(acc_S, s_hi, k_reg, acc_S);
        warp::mma_ABt(acc_S, s_lo, k_reg, acc_S);
        warp::mma_ABt(acc_C, c_hi, k_reg, acc_C);
        warp::mma_ABt(acc_C, c_lo, k_reg, acc_C);
        warp::load(k_reg, Klo_s);
        warp::mma_ABt(acc_S, s_hi, k_reg, acc_S);
        warp::mma_ABt(acc_C, c_hi, k_reg, acc_C);

        __syncthreads();  // done with K tiles; safe to overwrite next iter
    }

    // --- epilogue: fused recombination, single global write (lines 21-23).
    // (theta_r is loaded here rather than in a prologue purely to keep
    // register pressure down during the MMA loop — same values either way.)
    rt_fl<BM, BR> th_r, s_r, c_r;
    warp::load(th_r, g.theta, {0, 0, warp_row, r_blk});
    warp::apply(s_r, th_r, [](int, int, float x) { return __sinf(x); });
    warp::apply(c_r, th_r, [](int, int, float x) { return __cosf(x); });

    warp::mul(acc_S, acc_S, c_r);   // C_r . acc_S
    warp::mul(acc_C, acc_C, s_r);   // S_r . acc_C
    warp::sub(acc_S, acc_S, acc_C);

    typename rt_fl<BM, BR>::row_vec om;   // omega_r, broadcast down the batch rows
    warp::load(om, g.omega, {0, 0, 0, r_blk});
    warp::add_col(acc_S, acc_S, om);

    warp::store(g.out, acc_S, {0, 0, warp_row, r_blk});
}

torch::Tensor kuramoto_forward(torch::Tensor theta, torch::Tensor omega, torch::Tensor K) {
    TORCH_CHECK(theta.is_cuda() && omega.is_cuda() && K.is_cuda(), "all inputs must be CUDA tensors");
    TORCH_CHECK(theta.dtype() == torch::kFloat32 && omega.dtype() == torch::kFloat32 &&
                K.dtype() == torch::kFloat32, "all inputs must be fp32");
    TORCH_CHECK(theta.dim() == 2, "theta must be (batch, n)");
    theta = theta.contiguous();
    omega = omega.contiguous();
    K = K.contiguous();

    const int64_t batch = theta.size(0);
    const int64_t n = theta.size(1);
    TORCH_CHECK(K.size(0) == n && K.size(1) == n, "K must be (n, n)");
    TORCH_CHECK(omega.numel() == n, "omega must have n elements");
    TORCH_CHECK(n % BR == 0, "n must be a multiple of ", BR, " (got ", n, ")");

    auto stream = at::cuda::getCurrentCUDAStream();

    // Pad the batch up to a whole number of 128-row blocks (TK tile loads
    // don't bounds-check). Padded rows compute garbage that is sliced off.
    const int64_t batch_p = (batch + BM_BLOCK - 1) / BM_BLOCK * BM_BLOCK;
    torch::Tensor theta_p = theta;
    if (batch_p != batch) {
        theta_p = torch::zeros({batch_p, n}, theta.options());
        theta_p.narrow(0, 0, batch).copy_(theta);
    }

    // Precompute K's bf16 hi/lo split (theta-independent, one pass over K).
    auto bf_opts = theta.options().dtype(torch::kBFloat16);
    auto K_hi = torch::empty({n, n}, bf_opts);
    auto K_lo = torch::empty({n, n}, bf_opts);
    {
        const long long size = (long long)n * n;
        const int threads = 256;
        const int blocks = (int)std::min<long long>((size + threads - 1) / threads, 65535);
        split_bf16_kernel<<<blocks, threads, 0, stream>>>(
            K.data_ptr<float>(),
            reinterpret_cast<bf16*>(K_hi.data_ptr<at::BFloat16>()),
            reinterpret_cast<bf16*>(K_lo.data_ptr<at::BFloat16>()),
            size
        );
    }

    auto out = torch::empty({batch_p, n}, theta.options());

    fk_globals g{
        g_f32{theta_p.data_ptr<float>(), nullptr, nullptr, (int)batch_p, (int)n},
        g_f32{omega.data_ptr<float>(), nullptr, nullptr, 1, (int)n},
        g_bf{reinterpret_cast<bf16*>(K_hi.data_ptr<at::BFloat16>()), nullptr, nullptr, (int)n, (int)n},
        g_bf{reinterpret_cast<bf16*>(K_lo.data_ptr<at::BFloat16>()), nullptr, nullptr, (int)n, (int)n},
        g_f32{out.data_ptr<float>(), nullptr, nullptr, (int)batch_p, (int)n},
        (int)n,
        (int)batch_p
    };

    const int grid = (int)((n / BR) * (batch_p / BM_BLOCK));
    const int smem = 2 * sizeof(st_bf<BR, BS>) + 1024;  // + allocator alignment slack
    flash_kuramoto_kernel<<<grid, NWARPS * WARP_THREADS, smem, stream>>>(g);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "flash_kuramoto launch failed: ", cudaGetErrorString(err));

    return (batch_p == batch) ? out : out.narrow(0, 0, batch);
}
