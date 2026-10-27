"""
Common utilities for journal-extension experiments.

Bu modul mevcut src/ kodunu YENIDEN KULLANIR; run_comprehensive_experiments.py
dosyasina dokunmaz. Boylece dergi deneyleri ana pipeline'dan bagimsiz calisir
ama ayni model / veri / DP-SGD kurulumunu paylasir (sonuclar karsilastirilabilir).
"""
from __future__ import annotations
import os, sys, json, math, time, random
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score

# --- repo koku + src/ sys.path'e ekle (paket disaridan da calissin) -----------
# NOT: src/ icindeki moduller birbirini "from counterfactual..." / "from models..."
# gibi (src bir sources-root'muskca) import ediyor; ana runner da satir 26'da
# src'yi yola ekliyor. Ayni davranisi burada tekrar ediyoruz ki CLI'dan calissin.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
_SRC = os.path.join(_ROOT, "src")
for _p in (_ROOT, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.models.ids_models import IDSMLP, IDSTransformer  # noqa: E402
from src.counterfactual.cf_generator import CounterfactualGenerator  # noqa: E402
from src.data import real_data_loader as rdl  # noqa: E402

# =============================================================================
# CONFIG (ana Config ile tutarli)
# =============================================================================
DATASETS = {
    "ciciot2023": {"path": "dataset/ciciot2023/", "max_samples": 500000},
    "edge-iiot":  {"path": "dataset/edge-iiot/",  "max_samples": 500000},
    "toniot":     {"path": "dataset/toniot/",     "max_samples": 500000},
}
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
DP_DELTA = 1e-5
DP_CLIP_NORM = 1.0
REFERENCE_EPSILON = 2.0
NUM_CFS = 5
CF_MAX_ITER = 300
RANDOM_SEED = 42
OUT_ROOT = "experiments_journal"   # dergi deneyleri buraya yazilir

# Loader siniflari (convenience fonksiyonu scaler'i dondurmuyor; dogrudan kullaniyoruz)
_LOADERS = {
    "ciciot2023": rdl.CICIoT2023Loader,
    "edge-iiot":  rdl.EdgeIIoTLoader,
    "edge_iiot":  rdl.EdgeIIoTLoader,
    "toniot":     rdl.ToNIoTLoader,
    "ton-iot":    rdl.ToNIoTLoader,
}


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> str:
    """'cuda' if available, else 'cpu'.  Override with the environment variable JE_DEVICE
    (e.g. JE_DEVICE=cpu python3 -m journal_extension.exp19_controls ...) when the GPU is
    unavailable or unreliable; results are identical up to floating-point order."""
    forced = os.environ.get("JE_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        if forced == "cuda" and not torch.cuda.is_available():
            print("[get_device] JE_DEVICE=cuda but CUDA is unavailable -> cpu")
            return "cpu"
        return forced
    return "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# DP NOISE (exp12/13 ile AYNI RDP formulu)
# =============================================================================
def sigma_from_epsilon(epsilon: float, delta: float = DP_DELTA) -> float:
    """sigma = sqrt(2 ln(1.25/delta)) / epsilon  (Gaussian mechanism)."""
    if epsilon == float("inf") or epsilon is None or epsilon <= 0:
        return 0.0
    return math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


# =============================================================================
# DATA
# =============================================================================
_DATA_CACHE: Dict[str, tuple] = {}
DISK_CACHE_DIR = os.path.join(_ROOT, "dataset", "_cache")   # dataset/ is git-ignored


def _loader_fingerprint() -> str:
    """Short hash of the loader source: a change in preprocessing invalidates the disk cache."""
    import hashlib
    try:
        with open(rdl.__file__, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()[:8]
    except Exception:  # noqa: BLE001
        return "nohash"


def load_data(dataset: str, max_samples: Optional[int] = None, cache: bool = True,
              disk_cache: bool = True):
    """
    Returns X_train, X_test, y_train, y_test, info
    info EK olarak 'scaler' ve 'feature_names' icerir (ham-uzay raporlamasi icin).

    Disk cache: the loaders read the full raw CSVs (CICIoT2023: 45M rows, ~7 min and
    tens of GB of RAM) and then take a SEEDED sample + stratified split (random_state=42),
    so the result is deterministic. The prepared tensors + info (scaler, encoders,
    feature names) are therefore stored once under dataset/_cache/ and reused; the file
    name carries the dataset, the sample size and a hash of the loader source, so any
    change in preprocessing produces a fresh cache. Set disk_cache=False (or env
    JE_NO_DISK_CACHE=1) to force a raw reload.
    """
    key = f"{dataset}:{max_samples}"
    if cache and key in _DATA_CACHE:
        return _DATA_CACHE[key]
    ds = dataset.lower()
    cfg = DATASETS.get(ds, DATASETS.get(ds.replace("_", "-"), None))
    ms = max_samples if max_samples is not None else (cfg["max_samples"] if cfg else 500000)
    path = cfg["path"] if cfg else f"dataset/{ds}/"
    use_disk = disk_cache and os.environ.get("JE_NO_DISK_CACHE", "0") != "1"
    cache_file = os.path.join(DISK_CACHE_DIR, f"{ds.replace('_', '-')}_{ms}_{_loader_fingerprint()}.pt")
    out = None
    if use_disk and os.path.exists(cache_file):
        try:
            blob = torch.load(cache_file, weights_only=False, map_location="cpu")
            out = (blob["Xtr"], blob["Xte"], blob["ytr"], blob["yte"], blob["info"])
            print(f"[load_data] cached sample loaded: {cache_file} "
                  f"(train={len(out[0])}, test={len(out[1])}, created {blob.get('created', '?')})")
        except Exception as ex:  # noqa: BLE001
            print(f"[load_data] cache unreadable ({ex}); reloading raw data")
            out = None
    if out is None:
        loader = _LOADERS[ds](data_path=path, sample_fraction=1.0, max_samples=ms)
        Xtr, Xte, ytr, yte, info = loader.load()
        info["scaler"] = getattr(loader, "scaler", None)   # StandardScaler (ham uzay icin)
        out = (Xtr, Xte, ytr, yte, info)
        if use_disk:
            try:
                os.makedirs(DISK_CACHE_DIR, exist_ok=True)
                tmp = cache_file + ".tmp"
                torch.save({"Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "info": info,
                            "dataset": ds, "max_samples": ms, "loader": type(loader).__name__,
                            "random_state": getattr(loader, "random_state", None),
                            "created": datetime.now().isoformat(timespec="seconds")}, tmp)
                os.replace(tmp, cache_file)
                print(f"[load_data] sample cached -> {cache_file}")
            except Exception as ex:  # noqa: BLE001
                print(f"[load_data] could not write cache ({ex}); continuing without it")
        import gc; gc.collect()
    if cache:
        _DATA_CACHE[key] = out
    return out


def class_weights_tensor(y, num_classes: int, device: str) -> torch.Tensor:
    y_np = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    counts = np.bincount(y_np, minlength=num_classes)
    w = len(y_np) / (num_classes * (counts + 1))
    return torch.tensor(w, dtype=torch.float32, device=device)


def make_loaders(Xtr, ytr, Xte, yte, batch_size: int = None):
    # NOT: BATCH_SIZE cagri aninda okunur; runner'lar C.BATCH_SIZE'i --batch-size ile
    # gunceller ve tum make_loaders cagrilarina otomatik yansir.
    from torch.utils.data import DataLoader, TensorDataset
    bs = batch_size if batch_size is not None else BATCH_SIZE
    tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=bs, shuffle=True)
    te = DataLoader(TensorDataset(Xte, yte), batch_size=bs, shuffle=False)
    return tr, te


def set_batch_size(bs):
    """Global egitim batch boyutunu ayarla (runner'lar --batch-size icin cagirir)."""
    global BATCH_SIZE
    if bs:
        BATCH_SIZE = int(bs)


# =============================================================================
# MODEL FACTORY  (exp17 icin cok mimarili)
# =============================================================================
class IDSCNN1D(nn.Module):
    """Ozellikleri 1xF sinyal gibi ele alan hafif 1D-CNN (paradox kapasite testi)."""
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)            # (B,1,F)
        return self.head(self.features(x))
    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=-1)
    def predict(self, x):
        return self.forward(x).argmax(-1)


def make_model(arch: str, info: dict, hidden_dims: Optional[List[int]] = None,
               dropout: float = 0.3) -> nn.Module:
    F = info["num_features"]; C = info["num_classes"]
    arch = arch.lower()
    if arch in ("mlp", "baseline"):
        return IDSMLP(input_dim=F, num_classes=C, hidden_dims=hidden_dims or [256, 128, 64], dropout=dropout)
    if arch in ("wide_mlp", "widemlp"):
        return IDSMLP(input_dim=F, num_classes=C, hidden_dims=[512, 256, 128], dropout=dropout)
    if arch in ("cnn", "cnn1d"):
        return IDSCNN1D(input_dim=F, num_classes=C, dropout=dropout)
    if arch in ("transformer", "tab_transformer"):
        return IDSTransformer(input_dim=F, num_classes=C)
    raise ValueError(f"Unknown arch: {arch}")


# =============================================================================
# DP-SGD CONDITION TRAINING  (exp12/13 ile birebir ayni egitim dongusu)
# =============================================================================
def train_condition(model, train_loader, *, use_clipping: bool, use_noise: bool,
                    noise_std: float, epochs: int, lr: float,
                    criterion, device: str,
                    clip_norm: float = DP_CLIP_NORM,
                    collect_grad_norms: bool = False) -> Dict:
    """
    A/B/C/D kosullarini destekleyen DP-SGD egitimi.
      A_Standard : clip=F noise=F ; D_Clip_Only: clip=T noise=F
      C_Noise    : clip=F noise=T ; B_DP_SGD   : clip=T noise=T
    collect_grad_norms=True ise egitim boyunca global gradient norm dagilimini toplar
    (mekanizma analizi icin — DP'nin gradyani yumusatma etkisi).
    """
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    grad_norms: List[float] = []
    for _ in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            if collect_grad_norms:
                tot = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        tot += float(p.grad.detach().norm(2).item() ** 2)
                grad_norms.append(math.sqrt(tot))
            if use_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if use_noise and noise_std > 0:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * noise_std)
            opt.step()
    model.eval()
    return {"grad_norms": grad_norms}


CONDITIONS = {
    # name -> (use_clipping, use_noise)
    "A_Standard":   (False, False),
    "B_DP_SGD":     (True,  True),
    "C_Noise_Only": (False, True),
    "D_Clip_Only":  (True,  False),
}


def evaluate_model(model, test_loader, device: str) -> Dict[str, float]:
    model.eval(); preds, labels = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            preds.extend(model(bx.to(device)).argmax(1).cpu().numpy())
            labels.extend(by.numpy())
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
    }


# =============================================================================
# CF EVALUATION  (exp2 ile ayni CF ureteci)
# =============================================================================
def get_attack_samples(model, X, y, n, device: str, benign_class: int = 0):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(len(X)):
            if model(X[i:i+1].to(device)).argmax(1).item() != benign_class:
                out.append(X[i])
                if len(out) >= n:
                    break
    return torch.stack(out) if out else None


def eval_cf(model, samples, info, device, *, target_class: int = 0,
            max_samples: int = 100, num_cfs: int = NUM_CFS,
            max_iter: int = CF_MAX_ITER) -> Dict:
    gen = CounterfactualGenerator(model=model, feature_count=info["num_features"],
                                  num_cfs=num_cfs, use_dp=False, max_iter=max_iter,
                                  device=device)
    val, prox, spar = [], [], []
    for i in range(min(max_samples, len(samples))):
        try:
            r = gen.generate(samples[i].to(device), target_class=target_class)
            val.append(r.validity); prox.append(r.proximity); spar.append(r.sparsity)
        except Exception:
            continue
    if not val:
        return {"validity": 0.0, "proximity": 0.0, "sparsity": 0.0, "n": 0,
                "validity_per_sample": []}
    return {"validity": float(np.mean(val)), "validity_std": float(np.std(val)),
            "proximity": float(np.mean(prox)), "sparsity": float(np.mean(spar)),
            "n": len(val), "validity_per_sample": val}


# =============================================================================
# OUTPUT / LOGGING
# =============================================================================
class Runlog:
    def __init__(self, exp_name: str, dataset: str, root: str = OUT_ROOT):
        self.exp_name = exp_name; self.dataset = dataset
        self.dir = os.path.join(root, f"{exp_name}_{dataset}")
        os.makedirs(self.dir, exist_ok=True)
        self.log_path = os.path.join(self.dir, "log.txt")
        self.t0 = time.time()
        self.log(f"=== {exp_name} | {dataset} | {datetime.now():%Y-%m-%d %H:%M} ===")

    def log(self, msg: str):
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def save_json(self, obj: dict, name: str = "results.json"):
        obj.setdefault("_meta", {})
        obj["_meta"].update({"exp": self.exp_name, "dataset": self.dataset,
                             "duration_sec": round(time.time() - self.t0, 1)})
        with open(os.path.join(self.dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str, ensure_ascii=False)
        self.log(f"saved -> {os.path.join(self.dir, name)}")


def mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.std(xs))
