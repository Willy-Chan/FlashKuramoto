"""Isolated Kuramoto oscillator coupling op — the model's sequence-mixing kernel.

Extracted from `un0/model.py` (`_kuramoto_velocity` and
`ConditionalKuramotoDynamics.forward`). This is the O(n^2) compute-bound
operation inside the model's ODE solver: at every solver substep, every
oscillator's phase velocity depends on every other oscillator's phase through
a learned dense coupling matrix `K`. It plays the same structural role as the
QK^T / softmax(...)@V matmuls in attention (batched dense mixing across a
"sequence" of n tokens/oscillators), except:

  * the mixing weights are a fixed, non-data-dependent learned matrix `K`
    (no query/key projection, no softmax), and
  * the nonlinearity is `sin`/`cos` of the phases rather than softmax of
    dot-product logits.

For a solver step count `num_steps`, the model calls this op
`num_steps` times (Euler) or `4 * num_steps` times (RK4) per generated batch,
each call doing 2-4 dense `(batch, n) @ (n, n)` matmuls at n up to 16384
(ImageNet-64 `n16384`) — this is the dominant cost of sampling/training and
the natural target for a fused CUDA kernel (fusing sin/cos + the matmuls +
the diagonal mask + the drive cross-term into one kernel, and ideally across
solver substeps).

Standalone: depends only on `torch`. No torchdiffeq, no model config, no
training code, so it can be dropped next to a CUDA/Triton implementation and
diffed against directly.

Usage:
    uv run python scripts/kuramoto_coupling_op.py
"""

from __future__ import annotations

import time

import torch
from torch import Tensor, nn

# ---------------------------------------------------------------------------
# 1. The op, in two forms: a readable O(n^2) reference spec, and the fast
#    O(n^2)-flops-but-matmul form actually used in the model (`un0/model.py:19`).
# ---------------------------------------------------------------------------


def kuramoto_velocity_reference(theta: Tensor, omega: Tensor, K: Tensor) -> Tensor:
    """Direct definition: dtheta_i/dt = omega_i + sum_j K_ij * sin(theta_j - theta_i).

    theta: (batch, n). omega: (1, n) or (n,), broadcasts over batch.
    K: (n, n); K[i, j] is the coupling weight from oscillator j onto i.

    Materializes a (batch, n, n) pairwise-phase-difference tensor, so this is
    the correctness spec for `kuramoto_velocity` below, not what the model
    runs (n=16384 makes the (batch, n, n) tensor too large). Use only at
    small n to validate a fused/CUDA implementation bit-for-bit against the
    math.
    """
    theta_i = theta.unsqueeze(-1)  # (batch, n, 1): row index i
    theta_j = theta.unsqueeze(-2)  # (batch, 1, n): col index j
    diff = theta_j - theta_i  # diff[b, i, j] = theta_j - theta_i
    coupling_term = torch.einsum("ij,bij->bi", K, torch.sin(diff))
    return omega + coupling_term


def kuramoto_velocity(theta: Tensor, omega: Tensor, K: Tensor) -> Tensor:
    """Production form of the op (verbatim `un0/model.py::_kuramoto_velocity`).

    Expands sin(theta_j - theta_i) = sin(theta_j)cos(theta_i) - cos(theta_j)sin(theta_i)
    so the O(n^2) coupling sum becomes two dense `(batch, n) @ (n, n)` matmuls
    instead of an explicit (batch, n, n) outer product:

        dtheta/dt = omega + cos(theta) * (sin(theta) @ K^T) - sin(theta) * (cos(theta) @ K^T)

    This is the fusion target: two matmuls plus four elementwise sin/cos/mul/add
    ops, evaluated back-to-back on the same operands, all memory-bandwidth
    bound at large n. A CUDA/Triton kernel should fuse sin_theta/cos_theta
    computation with both matmuls and the final combine into a single pass
    (analogous to a fused-attention kernel, but with sin/cos in place of the
    softmax and no online-softmax rescaling needed since there's no reduction
    normalization).
    """
    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    weighted_sin = sin_theta @ K.transpose(-1, -2)
    weighted_cos = cos_theta @ K.transpose(-1, -2)
    return omega + cos_theta * weighted_sin - sin_theta * weighted_cos


# ---------------------------------------------------------------------------
# 2. The op as actually invoked in the model: one dynamics step couples a
#    main oscillator block to itself, a small conditioning block to itself,
#    and drives the main block from the conditioning block. Verbatim logic
#    from `ConditionalKuramotoDynamics.forward` (`un0/model.py:125-155`),
#    with training-only bits (mup scaling, class dropout) kept since they
#    change the op's arithmetic, and unrelated bits (nn.Module bookkeeping)
#    trimmed.
# ---------------------------------------------------------------------------


class KuramotoCouplingStep(nn.Module):
    """One Kuramoto dynamics evaluation: the function called every ODE solver substep.

    This is the full unit of work per `odeint` callback in
    `ConditionalImplicitKuramotoGenerator.forward` (`un0/model.py:386-392`):
    given the current joint phase state (main + conditioning oscillators) and
    a per-sample class drive matrix, return the phase velocity for the whole
    state. A solver step calls this once (Euler) or four times (RK4); a full
    generation calls it `num_steps` or `4 * num_steps` times.
    """

    def __init__(self, *, n: int, n_cond: int, num_classes: int) -> None:
        super().__init__()
        self.n = int(n)
        self.n_cond = int(n_cond)
        self.num_classes = int(num_classes)

        self.omega = nn.Parameter(torch.randn(1, self.n))
        K_init = (self.n**-0.5) * torch.randn(self.n, self.n)
        K_init.fill_diagonal_(0.0)
        self.K = nn.Parameter(K_init)

        self.omega_cond = nn.Parameter(torch.randn(1, self.n_cond))
        K_cond_init = (self.n_cond**-0.5) * torch.randn(self.n_cond, self.n_cond)
        K_cond_init.fill_diagonal_(0.0)
        self.K_cond = nn.Parameter(K_cond_init)

        self.K_drive = nn.Parameter(
            (self.n_cond**-0.5) * torch.randn(self.num_classes, self.n, self.n_cond)
        )

    def forward(self, state: Tensor, drive: Tensor) -> Tensor:
        """state: (batch, n + n_cond). drive: (batch, n, n_cond) (`K_drive[class_id]`)."""
        theta_main = state[:, : self.n]
        theta_cond = state[:, self.n :]

        # Zero the diagonal every call so oscillators don't self-couple.
        K = self.K - torch.diag_embed(self.K.diagonal())
        K_cond = self.K_cond - torch.diag_embed(self.K_cond.diagonal())

        main_vel = kuramoto_velocity(theta_main, self.omega, K)
        cond_vel = kuramoto_velocity(theta_cond, self.omega_cond, K_cond)

        # Cross-block "drive" coupling: conditioning block -> main block only
        # (one-way; the main block does not feed back into the cond block).
        sin_c = torch.sin(theta_cond)
        cos_c = torch.cos(theta_cond)
        sin_m = torch.sin(theta_main)
        cos_m = torch.cos(theta_main)
        drive_sin = torch.einsum("bnm,bm->bn", drive, sin_c)
        drive_cos = torch.einsum("bnm,bm->bn", drive, cos_c)
        main_vel = main_vel + cos_m * drive_sin - sin_m * drive_cos

        return torch.cat([main_vel, cond_vel], dim=1)


# ---------------------------------------------------------------------------
# 3. Correctness check (fast form vs. O(n^2) reference) + a throughput
#    benchmark at a size matching the released ImageNet-64 n16384 checkpoint,
#    so a CUDA/Triton replacement has a concrete target to beat.
# ---------------------------------------------------------------------------


def _check_correctness(*, device: torch.device, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    batch, n = 8, 64
    theta = (torch.rand(batch, n, device=device, dtype=dtype) * 2 - 1) * torch.pi
    omega = torch.randn(1, n, device=device, dtype=dtype)
    K = torch.randn(n, n, device=device, dtype=dtype)

    fast = kuramoto_velocity(theta, omega, K)
    ref = kuramoto_velocity_reference(theta, omega, K)
    max_err = (fast - ref).abs().max().item()
    print(f"correctness: max |fast - reference| = {max_err:.3e} (n={n}, dtype={dtype})")
    assert max_err < 1e-3, "fast and reference forms disagree beyond floating-point tolerance"


def _benchmark(*, device: torch.device, dtype: torch.dtype) -> None:
    # Matches build_imagenet64_model's n16384 checkpoint: n_oscillators=16384,
    # n_conditional_oscillators=1, num_classes=1000, per-device batch 2048.
    n, n_cond, num_classes, batch = 16384, 1, 1000, 2048
    if device.type == "cpu":
        # CPU sizing would take minutes at full scale; shrink for a smoke run.
        n, batch = 2048, 256

    step = KuramotoCouplingStep(n=n, n_cond=n_cond, num_classes=num_classes).to(device, dtype)
    state = torch.randn(batch, n + n_cond, device=device, dtype=dtype, requires_grad=True)
    class_id = torch.randint(0, num_classes, (batch,), device=device)

    num_iters = 20
    warmup = 5
    for i in range(warmup + num_iters):
        if i == warmup:
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
        # Recomputed each iteration: reusing one `drive` graph node across
        # multiple independent `.backward()` calls double-frees its buffers.
        drive = step.K_drive[class_id]
        velocity = step(state, drive)
        velocity.sum().backward()
        state.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    per_call_ms = 1000 * elapsed / num_iters
    print(
        f"benchmark: n={n}, n_cond={n_cond}, batch={batch}, dtype={dtype}, device={device} -> "
        f"{per_call_ms:.3f} ms/call (fwd+bwd), {num_iters} iters"
    )
    print(
        "  a full ImageNet-64 generation calls this "
        f"num_steps (euler, default 10) or 4*num_steps (rk4) times: "
        f"~{10 * per_call_ms:.1f} ms (euler, num_steps=10) just for this op."
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    _check_correctness(device=device, dtype=dtype)
    _benchmark(device=device, dtype=dtype)
