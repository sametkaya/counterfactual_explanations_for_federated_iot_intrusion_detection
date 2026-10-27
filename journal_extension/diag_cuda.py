"""
CUDA / cuBLAS diagnostic for the exp19 start-up failure.
Runs, in THIS process and shell environment, exactly the first training steps of
journal_extension.common.train_condition (IDSMLP with BatchNorm + dropout, Adam,
class-weighted cross-entropy, batch 256) on random data of the CICIoT2023 shape,
and reports the environment and the CUDA libraries actually loaded.

Run from the repo root:   python3 -m journal_extension.diag_cuda
Add --with-data to load the real CICIoT2023 sample first (as exp19 does) and
repeat the steps afterwards — tests whether the post-load memory state matters.
"""
from __future__ import annotations
import os, sys, time, platform

print("=== environment ===")
for k in sorted(os.environ):
    if any(t in k.upper() for t in ("CUDA", "CUBLAS", "CUDNN", "LD_LIBRARY", "NVIDIA", "OMP_", "MKL", "TORCH", "PYTORCH")):
        print(f"  {k}={os.environ[k]}")
print(f"  python={sys.version.split()[0]} platform={platform.platform()}")

import numpy as np
import torch, torch.nn as nn
print("=== torch ===")
print(f"  torch={torch.__version__} cuda={torch.version.cuda} cudnn={torch.backends.cudnn.version()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"  mem_get_info(GB)={[round(v/1e9, 2) for v in torch.cuda.mem_get_info()] if torch.cuda.is_available() else None}")

from journal_extension import common as C

def step(tag, fn):
    t = time.time()
    try:
        fn(); (torch.cuda.synchronize() if torch.cuda.is_available() else None); print(f"  [ok]   {tag} ({time.time()-t:.1f}s)")
        return True
    except Exception as ex:  # noqa: BLE001
        print(f"  [FAIL] {tag}: {type(ex).__name__}: {str(ex).splitlines()[0]}")
        return False

dev = "cuda" if torch.cuda.is_available() else "cpu"
WITH_DATA = "--with-data" in sys.argv
if WITH_DATA:
    print("=== loading real data first (as exp19 does) ===")
    ds = next((a for a in sys.argv if a in ("ciciot2023", "edge-iiot", "toniot")), "ciciot2023")
    Xtr, Xte, ytr, yte, info = C.load_data(ds, 200000)
    import resource
    print(f"  loaded {ds}: train={len(Xtr)} maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB")
    print(f"  mem_get_info(GB)={[round(v/1e9, 2) for v in torch.cuda.mem_get_info()] if torch.cuda.is_available() else None}")
else:
    info = {"num_features": 39, "num_classes": 8, "dataset": "ciciot2023"}
print("=== steps ===")
ok = step("plain matmul + backward (Linear)", lambda: nn.functional.cross_entropy(
    nn.Linear(39, 8).to(dev)(torch.randn(256, 39, device=dev)), torch.randint(0, 8, (256,), device=dev)).backward())

C.set_seed(42)
m = C.make_model("mlp", info).to(dev)
if WITH_DATA:
    X, y = Xtr[:256 * 20], ytr[:256 * 20]
    tl, _ = C.make_loaders(Xtr, ytr, Xte, yte, batch_size=256)      # the real loader, as in exp19
    cw = C.class_weights_tensor(ytr, info["num_classes"], dev)
else:
    X = torch.randn(256 * 20, 39); y = torch.randint(0, 8, (256 * 20,))
    tl, _ = C.make_loaders(X, y, X[:512], y[:512], batch_size=256)
    cw = C.class_weights_tensor(y, 8, dev)
crit = nn.CrossEntropyLoss(weight=cw)
K = info["num_classes"]

ok &= step("IDSMLP forward (train mode, BatchNorm+dropout)", lambda: m.train()(X[:256].to(dev)))
ok &= step("IDSMLP forward+backward, class-weighted CE", lambda: crit(m(X[:256].to(dev)), y[:256].to(dev)).backward())
ok &= step("common.train_condition A (1 epoch of the loader)", lambda: C.train_condition(
    m, tl, use_clipping=False, use_noise=False, noise_std=0.0, epochs=1, lr=1e-3, criterion=crit, device=dev))
ok &= step("common.train_condition B (clip+noise sigma=2.42)", lambda: C.train_condition(
    m, tl, use_clipping=True, use_noise=True, noise_std=2.4224, epochs=1, lr=1e-3, criterion=crit, device=dev))
# a larger, longer stress: 200 batches
if not WITH_DATA:
    ok &= step("stress: 200 batches A", lambda: [C.train_condition(m, tl, use_clipping=False, use_noise=False, noise_std=0.0,
                                                                    epochs=1, lr=1e-3, criterion=crit, device=dev) for _ in range(10)])

print("=== CUDA libraries mapped in this process ===")
seen = set()
with open("/proc/self/maps") as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 6 and any(k in parts[5] for k in ("cublas", "cudnn", "libcuda.so", "cudart", "nvrtc")):
            seen.add(parts[5])
for p in sorted(seen):
    print("  ", p)
print("RESULT:", "ALL OK" if ok else "FAILURES ABOVE")
