"""
GPU memory + cuBLAS consistency test (hardware/driver-level diagnostic).

Why: new CUDA processes on this machine fail intermittently with
"illegal memory access" / "CUBLAS_STATUS_INTERNAL_ERROR" inside plain sgemm
kernels, while long-running older processes are fine.  cuBLAS kernels have no
data-dependent addressing, so a fault inside them points at the driver or the
VRAM itself.  This script (1) fills as much free VRAM as possible with known
patterns and reads it back, (2) copies random host data to every chunk and
compares, (3) repeats an identical large matmul many times and checks that the
result is bitwise stable.  Any mismatch or CUDA error = hardware/driver fault.

Run from the repo root:   python3 -m journal_extension.diag_vram [--seconds 60] [--reserve-gb 2]
(safe to run while another experiment is running: it leaves --reserve-gb free)
"""
from __future__ import annotations
import sys, time, argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0, help="duration of the matmul stress")
    ap.add_argument("--reserve-gb", type=float, default=2.0, help="VRAM to leave free for other processes")
    ap.add_argument("--chunk-mb", type=int, default=256)
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device"); return 1
    dev = "cuda"
    print(f"device={torch.cuda.get_device_name(0)} torch={torch.__version__} cuda={torch.version.cuda}")
    free, total = torch.cuda.mem_get_info()
    print(f"free={free/2**30:.2f} GB total={total/2**30:.2f} GB")
    chunk = a.chunk_mb * 2**20
    n_max = int(max(0, free - a.reserve_gb * 2**30) // chunk)
    bufs = []
    for _ in range(n_max):
        try:
            bufs.append(torch.empty(chunk // 4, dtype=torch.float32, device=dev))
        except RuntimeError:
            break
    print(f"[1] allocated {len(bufs) * chunk / 2**30:.2f} GB in {len(bufs)} chunks of {a.chunk_mb} MB")
    errors = 0
    try:
        # ---- (1) constant patterns -------------------------------------------------
        for pattern in (0.0, -1.0, 3.14159, 1e30):
            for b in bufs:
                b.fill_(pattern)
            torch.cuda.synchronize()
            for i, b in enumerate(bufs):
                bad = int((b != pattern).sum().item())
                if bad:
                    errors += 1; print(f"    chunk {i:3d}: {bad} mismatching elements after fill({pattern})")
        print(f"[2] constant-pattern fill/readback: {'OK' if errors == 0 else 'ERRORS=' + str(errors)}")
        # ---- (2) random host data round trip -------------------------------------------
        g = torch.Generator().manual_seed(12345)
        for i, b in enumerate(bufs):
            h = torch.randint(-2**31, 2**31 - 1, (chunk // 4,), dtype=torch.int32, generator=g).view(torch.float32)
            b.copy_(h)
            back = b.cpu()
            bad = int((back.view(torch.int32) != h.view(torch.int32)).sum().item())
            if bad:
                errors += 1; print(f"    chunk {i:3d}: {bad} mismatching elements in host<->device round trip")
        print(f"[3] random-data H2D/D2H round trip over {len(bufs)} chunks: {'OK' if errors == 0 else 'ERRORS=' + str(errors)}")
        del bufs; torch.cuda.empty_cache()
        # ---- (3) cuBLAS stability: identical matmuls must be bitwise identical -----------
        torch.manual_seed(0)
        A = torch.randn(2048, 2048, device=dev); B = torch.randn(2048, 2048, device=dev)
        ref = (A @ B).clone(); torch.cuda.synchronize()
        t0 = time.time(); it = 0; bad_it = 0
        while time.time() - t0 < a.seconds:
            Cm = A @ B
            if not torch.equal(Cm, ref):
                bad_it += 1
                if bad_it <= 5:
                    print(f"    iteration {it}: matmul result differs from reference (max abs diff {(Cm - ref).abs().max().item():.3e})")
            it += 1
        torch.cuda.synchronize()
        print(f"[4] cuBLAS sgemm stability: {it} iterations in {a.seconds:.0f}s, {bad_it} mismatches -> "
              f"{'OK' if bad_it == 0 else 'ERRORS'}")
        errors += bad_it
        # ---- (4) small-model training stress (same kernels as the experiments) -----------
        import torch.nn as nn
        m = nn.Sequential(nn.Linear(39, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Linear(256, 128), nn.BatchNorm1d(128),
                          nn.ReLU(), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 8)).to(dev)
        opt = torch.optim.Adam(m.parameters(), 1e-3); crit = nn.CrossEntropyLoss()
        X = torch.randn(160000, 39); y = torch.randint(0, 8, (160000,))
        t0 = time.time(); steps = 0
        while time.time() - t0 < a.seconds:
            idx = torch.randint(0, 160000, (256,))
            bx, by = X[idx].to(dev), y[idx].to(dev)
            opt.zero_grad(); crit(m(bx), by).backward(); opt.step(); steps += 1
        torch.cuda.synchronize()
        print(f"[5] MLP training stress: {steps} steps in {a.seconds:.0f}s -> OK")
    except RuntimeError as ex:
        print(f"[CUDA ERROR] {str(ex).splitlines()[0]}")
        errors += 1
    print("RESULT:", "ALL OK (no memory or cuBLAS fault detected)" if errors == 0 else f"FAULTS DETECTED ({errors})")
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
