#!/usr/bin/env python3
"""Peak memory and latency of the portable Mamba-3 scan.

The manuscript claims about scan cost must be backed by measurement rather than
asymptotics.  This script reports peak resident memory for a forward plus
backward pass of the rank-R MIMO scan, sweeping the blocking schedule.

Run one configuration per process so the peak is attributable:

    python benchmarks/benchmark_scan_memory.py --chunk none
    python benchmarks/benchmark_scan_memory.py --chunk 8

or sweep in-process (peaks then accumulate and only the maximum is meaningful):

    python benchmarks/benchmark_scan_memory.py --sweep
"""

from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import time

import torch

from mtace.mamba3 import mamba3_mimo_scan_parallel


def _rss_mib() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss is bytes on macOS and kibibytes on Linux.
    return peak / (1024**2 if sys.platform == "darwin" else 1024)


def run_once(chunk, atoms, shells, rank, heads, head_dim, state_dim):
    torch.manual_seed(0)
    shape = (atoms, shells, rank, heads, head_dim)
    key_shape = (atoms, shells, rank, heads, state_dim)
    x = torch.randn(*shape, requires_grad=True)
    k = torch.randn(*key_shape)
    q = torch.randn(*key_shape)
    alpha = torch.rand(atoms, shells, heads) * 0.5 + 0.4
    beta = torch.randn(atoms, shells, heads)
    gamma = torch.randn(atoms, shells, heads)
    skip = torch.randn(heads)
    gate = torch.randn(*shape)
    start = time.perf_counter()
    y = mamba3_mimo_scan_parallel(
        x, k, q, alpha, beta, gamma, skip, gate, chunk_size=chunk
    )
    y.square().sum().backward()
    elapsed = time.perf_counter() - start
    state_bytes = atoms * shells * heads * head_dim * state_dim * 4 / 1024**2
    return _rss_mib(), elapsed, state_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk", default="none")
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--shells", type=int, default=128)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--state-dim", type=int, default=16)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Re-exec one subprocess per schedule and tabulate the peaks",
    )
    arguments = parser.parse_args()

    common = [
        "--atoms", str(arguments.atoms),
        "--shells", str(arguments.shells),
        "--rank", str(arguments.rank),
        "--heads", str(arguments.heads),
        "--head-dim", str(arguments.head_dim),
        "--state-dim", str(arguments.state_dim),
    ]
    if arguments.sweep:
        print(
            f"MIMO scan, atoms={arguments.atoms} shells={arguments.shells} "
            f"rank={arguments.rank} heads={arguments.heads} "
            f"head_dim={arguments.head_dim} state_dim={arguments.state_dim} fp32"
        )
        print(f"{'schedule':>20} {'peak RSS (MiB)':>16} {'fwd+bwd (s)':>13}")
        for chunk in ("none", "32", "16", "8", "4", "2", "1"):
            output = subprocess.run(
                [sys.executable, __file__, "--chunk", chunk, *common],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip().splitlines()[-1]
            peak, elapsed = output.split()[-2:]
            label = "single pass" if chunk == "none" else f"blocked ({chunk})"
            print(f"{label:>20} {float(peak):16.1f} {float(elapsed):13.3f}")
        return

    chunk = None if arguments.chunk == "none" else int(arguments.chunk)
    peak, elapsed, state_bytes = run_once(
        chunk,
        arguments.atoms,
        arguments.shells,
        arguments.rank,
        arguments.heads,
        arguments.head_dim,
        arguments.state_dim,
    )
    print(f"# one recurrent state tensor is {state_bytes:.1f} MiB")
    print(f"{peak:.4f} {elapsed:.4f}")


if __name__ == "__main__":
    main()
