"""Compile + benchmark a hand-written CUDA/ThunderKittens kernel for `kuramoto_velocity`.

Runs against the PyTorch baseline, on a Modal GPU (see the `GPU` constant below).

The workflow: edit `scripts/kuramoto_kernel.cu` locally (no local CUDA
toolchain required — this machine doesn't have an NVIDIA GPU), then run this
script. It ships the kernel source to a Modal container, JIT-compiles it with
`torch.utils.cpp_extension.load_inline` against a cloned ThunderKittens
checkout, checks it against `kuramoto_velocity_reference` at small scale, and
times it against `kuramoto_velocity` (the PyTorch baseline) at the scale you
pass in.

Usage:
    uv run --with modal modal run scripts/kuramoto_modal_bench.py
    uv run --with modal modal run scripts/kuramoto_modal_bench.py --n 16384 --batch 2048  # full ImageNet-64 n16384 scale
    uv run --with modal modal shell scripts/kuramoto_modal_bench.py                       # drop into the exact build container

First run: `modal setup` (or `modal token new`) once to authenticate.

This compares FORWARD PASS ONLY — the harness doesn't wire up a backward
pass for the custom kernel, unlike the fwd+bwd benchmark in
kuramoto_coupling_op.py's own __main__.
"""

from __future__ import annotations

import pathlib

import modal

app = modal.App("kuramoto-tk-bench")

SCRIPT_DIR = pathlib.Path(__file__).parent
TK_ROOT = "/root/ThunderKittens"

# ThunderKittens' own kernels/common.mk pins one of these per target GPU.
# H100 (SM90) needs the "a" (architecture-specific) variant to use
# Hopper-only instructions (wgmma, tma); A100 (SM80) has no such variant.
_GPU_ARCH = {
    "H100": ("SM90", "compute_90a", "sm_90a"),
    "A100": ("SM80", "compute_80", "sm_80"),
    # L4 is Ada (SM89); ThunderKittens has no dedicated SM89 case in
    # kernels/common.mk, but Ada is a strict superset of Ampere's PTX ISA
    # (cp.async etc.), so the SM80 codepath + an explicit sm_89 gencode is
    # the correct combination. T4 (Turing, SM75) does NOT work here — TK's
    # SM80 path uses cp.async, which doesn't exist on Turing hardware.
    "L4": ("SM80", "compute_89", "sm_89"),
}
# Modal gates H100/A100 behind a payment method on file even with free
# credits. Switch back to "H100" once billing is set up on the target
# workspace — nothing else in this file needs to change.
GPU = "L4"
_arch_name, _compute, _sm = _GPU_ARCH[GPU]
ARCH_DEFINE = f"-DKITTENS_{_arch_name}"
ARCH_GENCODE = f"-gencode=arch={_compute},code={_sm}"

cuda_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "torch==2.11.*",
        "ninja",
        "numpy",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .run_commands(
        f"git clone --depth 1 https://github.com/HazyResearch/ThunderKittens.git {TK_ROOT}"
    )
)

# Caches load_inline's ninja build directory across runs so editing the
# kernel and re-running doesn't recompile ThunderKittens' headers from
# scratch every time.
build_cache = modal.Volume.from_name("kuramoto-tk-build-cache", create_if_missing=True)


@app.function(
    image=cuda_image,
    gpu=GPU,
    timeout=20 * 60,
    volumes={"/root/.cache/torch_extensions": build_cache},
)
def benchmark_kernel(
    kernel_source: str,
    baseline_source: str,
    n: int,
    batch: int,
) -> dict:
    import time
    import types

    import torch
    from torch.utils.cpp_extension import load_inline

    baseline = types.ModuleType("kuramoto_baseline")
    exec(compile(baseline_source, "kuramoto_coupling_op.py", "exec"), baseline.__dict__)  # noqa: S102

    cpp_decl = (
        "torch::Tensor kuramoto_forward(torch::Tensor theta, torch::Tensor omega, torch::Tensor K);"
    )
    ext = load_inline(
        name="kuramoto_tk_kernel",
        cpp_sources=[cpp_decl],
        cuda_sources=[kernel_source],
        functions=["kuramoto_forward"],
        extra_include_paths=[f"{TK_ROOT}/include", f"{TK_ROOT}/prototype"],
        extra_cuda_cflags=[
            "-std=c++20",
            "-O3",
            "--expt-extended-lambda",
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            ARCH_DEFINE,
            ARCH_GENCODE,
            # Prevents ambiguous-operator errors between CUDA's native
            # half/bf16 operators and PyTorch's, when both kittens.cuh and
            # torch headers land in the same translation unit.
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-Xcompiler=-Wno-psabi",
            "-Xcompiler=-fno-strict-aliasing",
        ],
        verbose=True,
    )

    device = torch.device("cuda")
    dtype = torch.float32
    result: dict = {}

    # --- correctness, small scale (mirrors kuramoto_coupling_op.py's own check) ---
    torch.manual_seed(0)
    check_n = 64
    theta = (torch.rand(8, check_n, device=device, dtype=dtype) * 2 - 1) * torch.pi
    omega = torch.randn(1, check_n, device=device, dtype=dtype)
    K = torch.randn(check_n, check_n, device=device, dtype=dtype)

    ref = baseline.kuramoto_velocity_reference(theta, omega, K)
    cuda_out = ext.kuramoto_forward(theta, omega, K)
    max_err = (cuda_out - ref).abs().max().item()
    result["correctness_max_err"] = max_err
    result["correctness_n"] = check_n
    print(f"correctness: max |cuda - reference| = {max_err:.3e} (n={check_n})")

    if max_err > 1e-3:
        result["error"] = "correctness check failed (max_err > 1e-3); skipping benchmark"
        return result

    # --- benchmark, full scale ---
    theta_b = torch.randn(batch, n, device=device, dtype=dtype)
    omega_b = torch.randn(1, n, device=device, dtype=dtype)
    K_b = torch.randn(n, n, device=device, dtype=dtype)

    def bench(fn, iters: int = 20, warmup: int = 5) -> float:
        for i in range(warmup + iters):
            if i == warmup:
                torch.cuda.synchronize()
                start = time.perf_counter()
            fn()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - start) / iters

    baseline_ms = bench(lambda: baseline.kuramoto_velocity(theta_b, omega_b, K_b))
    cuda_ms = bench(lambda: ext.kuramoto_forward(theta_b, omega_b, K_b))

    result.update(
        {
            "n": n,
            "batch": batch,
            "baseline_fwd_ms": baseline_ms,
            "cuda_fwd_ms": cuda_ms,
            "speedup": baseline_ms / cuda_ms,
        }
    )
    print(
        f"benchmark: n={n}, batch={batch} -> baseline {baseline_ms:.3f} ms/call (fwd), "
        f"cuda {cuda_ms:.3f} ms/call (fwd), speedup {baseline_ms / cuda_ms:.2f}x"
    )
    return result


@app.local_entrypoint()
def main(
    kernel_path: str = str(SCRIPT_DIR / "kuramoto_kernel.cu"),
    n: int = 4096,
    batch: int = 512,
) -> None:
    kernel_source = pathlib.Path(kernel_path).read_text()
    baseline_source = (SCRIPT_DIR / "kuramoto_coupling_op.py").read_text()
    result = benchmark_kernel.remote(kernel_source, baseline_source, n, batch)
    print(result)
