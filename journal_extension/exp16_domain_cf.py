"""
EXP16 - DOMAIN-CONSTRAINED (feasible-by-construction) ACTIONABLE CF   [v2]
=========================================================================
Soru: "valid" raporlanan CF'lerin kaci, ozelliklerin GERCEK tiplerine/araliklarina
uyar (uygulanabilir)?  Kisit-altinda uretim (generator her iterasyonda hard-snap +
soft penalti) ile feasible-by-construction CF uretilir; actionability = valid AND
feasible.

v2 (4 Eyl 2026) — NEDEN DEGISTI
--------------------------------
v1 tipleri OZELLIK ADINDAN turetiyordu. Veri profili bunun iki veri setinde de
yanlis oldugunu gosterdi:
  * CICIoT2023: protokol sutunlari (TCP, UDP, HTTP, ...) ve *_flag_number sutunlari
    pencere ORANLARIDIR ([0,1] icinde kesirli; ayni anda TCP=1, IPv=1, LLC=1
    olabilir) -> "one-hot / binary" kisiti YANLISTI.
  * Edge-IIoTset: tcp.ack_raw (4e9) "_ack" ipucuyla binary sayildi; tcp.seq/checksum
    gibi kimlik-benzeri sutunlar "count" sayildi -> float32 duyarlik yuzunden
    feasibility hicbir zaman saglanamadi (0.000).
v2 tipleri EGITIM VERISINDEN cikarir (data-driven):
  1. egitimde sabit sutun                              -> immutable (dondurulur)
  2. kimlik ipucu (checksum/seq/ack_raw/stream/uid/mac/addr/_id) veya
     tamsayi & max >= 1e6 (sira/onay numarasi, ham zaman damgasi)  -> immutable
  3. label-encoded kategorik (loader'dan) : nunique > max_categorical -> immutable,
                                             aksi halde              -> categorical
  4. tamsayi & degerler {0,1}                            -> binary
  5. tamsayi & nunique <= max_categorical                -> categorical (gozlenen degerlere snap)
  6. tamsayi                                             -> count (yuvarla, [min,max] icinde)
  7. digerleri                                           -> continuous ([min,max] icine kirp;
                                                            negatif hiz/IAT/varyans engellenir)
  Protokol one-hot grubu YALNIZCA uyeler veride binary ve satir toplami 1 ise kurulur.
Tum snap/ihlal hesaplari raw uzayda float64 ile yapilir.

Ayrica v2 iki dedektor icin olcer: A_Standard (gurultusuz) ve B_DP_SGD (nominal
eps=2 gurultu) -> "gurultu, kisit altinda da validity'yi artiriyor mu?" sorusu
dogrudan yanitlanir.

Calistir:
    python3 -m journal_extension.exp16_domain_cf --dataset ciciot2023 --max-samples 200000 \
        --batch-size 256 --cf-max-iter 300 --model-conditions A,B
    # eski (ad-tabanli) tipleri yeniden uretmek icin: --semantics name
"""
from __future__ import annotations
import argparse, re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from journal_extension import common as C
# generator ile AYNI namespace (src sys.path'te) -> duck-typed constraint uyumu
from counterfactual.constraints import IoTConstraintManager, Constraint, ImmutableConstraint  # noqa: E402
from counterfactual.cf_generator import CounterfactualGenerator  # noqa: E402

PROTO_TOKENS = {"tcp", "udp", "icmp", "arp", "http", "https", "dns", "dhcp",
                "smtp", "ssh", "telnet", "ftp", "irc", "ntp", "mqtt", "modbus",
                "llc", "ipv", "igmp", "ssl"}
COUNT_HINTS = ("count", "_cnt", "pkts", "packets", "len", "length", "size",
               "bytes", "duration", "trans_id", "seq", "ttl", "time_to_live",
               "header_length", "number", "num", "port", "_id")
TRUE_BINARY_HINTS = ("is_", "has_", "_rejected", "_aa", "_rd", "_ra",
                     "established", "resumed", "_syn", "_ack", "_rst", "_fin",
                     "conn.rst", "conn.syn", "flags.ack")
IMMUTABLE_HINTS = ("uid", "mac", "src_ip", "dst_ip", "ip.addr", "addr", "stream",
                   "ip_bytes")
# v2: kimlik-benzeri, eyleme donusturulemez sutunlar (raw sira/onay no, checksum, ...)
IDENTIFIER_HINTS = ("checksum", "ack_raw", ".seq", "_seq", "seq_le", "stream", "uid",
                    "mac", "addr", "trans_id", "_id", "timestamp", "frame.time")

MODEL_COND_ALIASES = {"A": "A_Standard", "B": "B_DP_SGD", "C": "C_Noise_Only", "D": "D_Clip_Only"}


def _is_count(name: str) -> bool:
    return name.endswith("_flag_number") or any(h in name for h in COUNT_HINTS)


def _proto_index(feature_names) -> List[int]:
    out = []
    for i, raw in enumerate(feature_names):
        name = str(raw).lower().strip(); base = re.sub(r"[^a-z0-9]", "", name)
        if any(name == t or base == t or name.endswith("." + t) or name.endswith("_" + t)
               for t in PROTO_TOKENS):
            out.append(i)
    return out


# =============================================================================
# v1 (legacy) — ad-tabanli tipler; --semantics name ile erisilir
# =============================================================================
def derive_semantics(feature_names):
    """[v1, legacy] Tip etiketi + one-hot gruplari ozellik ADINDAN (CICIoT'ta yanlis cikti)."""
    n = len(feature_names)
    ftype = ["continuous"] * n
    immutable = [False] * n
    onehot_groups = {}
    for i, raw in enumerate(feature_names):
        name = str(raw).lower().strip(); base = re.sub(r"[^a-z0-9]", "", name)
        mp = next((t for t in PROTO_TOKENS if name == t or base == t
                   or name.endswith("." + t) or name.endswith("_" + t)), None)
        if mp:
            onehot_groups.setdefault("protocol", []).append(i); ftype[i] = "binary"; continue
        if any(h in name for h in IMMUTABLE_HINTS):
            immutable[i] = True; ftype[i] = "immutable"; continue
        if _is_count(name):
            ftype[i] = "count"; continue
        if any(h in name for h in TRUE_BINARY_HINTS):
            ftype[i] = "binary"; continue
    lo = np.full(n, -np.inf); hi = np.full(n, np.inf)
    return {"ftype": ftype, "immutable": immutable, "onehot_groups": onehot_groups,
            "lo": lo, "hi": hi, "categories": {}, "mode": "name"}


# =============================================================================
# v2 — VERIDEN tip cikarimi
# =============================================================================
def infer_semantics_data(Xtr_z: torch.Tensor, scaler, feature_names, info: dict,
                         max_categorical: int = 32, id_threshold: float = 1e6,
                         n_probe: int = 100000) -> Dict:
    """Egitim verisinin RAW degerlerinden ozellik tiplerini ve araliklarini cikarir."""
    X = Xtr_z[:n_probe].numpy().astype(np.float64)
    raw = scaler.inverse_transform(X) if scaler is not None else X
    n = raw.shape[1]
    names = [str(x) for x in feature_names]
    encoded = set()
    le = info.get("label_encoders", None)
    if isinstance(le, dict):
        encoded = {str(k) for k in le.keys()}
    ftype = ["continuous"] * n
    immutable = [False] * n
    categories: Dict[int, List[float]] = {}
    lo = raw.min(axis=0); hi = raw.max(axis=0)
    reasons = {}
    for j in range(n):
        col = raw[:, j]
        rounded = np.round(col)
        # float32 z -> float64 raw gidis-donusu ~|x|*6e-8 hata birakir; goreli tolerans
        is_int = bool(np.all(np.abs(col - rounded) <= 1e-4 * np.maximum(1.0, np.abs(col))))
        uniq = np.unique(rounded if is_int else np.round(col, 6))
        nu = len(uniq)
        name = names[j].lower()
        if nu <= 1:
            ftype[j] = "immutable"; immutable[j] = True; reasons[names[j]] = "constant"; continue
        if any(h in name for h in IDENTIFIER_HINTS) or (is_int and hi[j] >= id_threshold):
            ftype[j] = "immutable"; immutable[j] = True; reasons[names[j]] = "identifier"; continue
        if names[j] in encoded:
            if nu > max_categorical:
                ftype[j] = "immutable"; immutable[j] = True
                reasons[names[j]] = "high-cardinality encoded string"; continue
            ftype[j] = "categorical"; categories[j] = [float(v) for v in uniq]
            reasons[names[j]] = "encoded categorical"; continue
        if is_int and set(uniq.tolist()) <= {0.0, 1.0}:
            ftype[j] = "binary"; reasons[names[j]] = "integer {0,1}"; continue
        if is_int and nu <= max_categorical:
            ftype[j] = "categorical"; categories[j] = [float(v) for v in uniq]
            reasons[names[j]] = f"integer, {nu} values"; continue
        if is_int:
            ftype[j] = "count"; reasons[names[j]] = "integer"; continue
        ftype[j] = "continuous"; reasons[names[j]] = "real-valued"
    # protokol one-hot grubu: yalnizca uyeler binary VE satir toplami 1 ise
    onehot_groups = {}
    proto = [i for i in _proto_index(names) if ftype[i] == "binary"]
    if len(proto) >= 2:
        s = raw[:, proto].sum(axis=1)
        if np.all(np.abs(s - 1.0) < 1e-6):
            onehot_groups["protocol"] = proto
    return {"ftype": ftype, "immutable": immutable, "onehot_groups": onehot_groups,
            "lo": lo, "hi": hi, "categories": categories, "reasons": reasons, "mode": "data"}


# =============================================================================
# DomainConstraint (v2) — z-uzayinda calisir, raw tip/aralik kurallarini float64 ile uygular
# =============================================================================
class DomainConstraint(Constraint):
    """
    Standardize (z) uzayda tanimli; raw = z*std+mean (float64) ile:
      binary      : raw -> {0,1}
      categorical : raw -> en yakin gozlenen deger
      count       : raw -> round, [lo,hi] icinde
      continuous  : raw -> [lo,hi] icine kirp  (v2: negatif hiz/IAT/varyans engellenir)
      one-hot grup: tam olarak bir uye "1"
    apply(): hard-snap. compute_violation(): feasible izgaraya raw-uzay uzakligi (soft).
    """
    def __init__(self, mean, std, sem: Dict, name="domain", snap_every: int = 10):
        # snap_every: hard-snap her k. apply() cagrisinda uygulanir (lazy projection).
        # Her adimda yuvarlamak, std'si kucuk tamsayi ozellikleri Adam adimi (0.01 z)
        # 0.5 raw birimin altinda kaldigi icin DONDURUR ("rounding trap"); k adimda bir
        # snap bunu onler. Araliga kirpma (continuous/count [lo,hi]) her adimda yapilir.
        # Son CF her durumda tam projeksiyondan gecirilir (_evaluate_model'de).
        self.snap_every = max(1, int(snap_every)); self._calls = 0
        self.mean = torch.tensor(np.asarray(mean, dtype=np.float64))
        std = np.asarray(std, dtype=np.float64).copy()
        std[np.abs(std) < 1e-12] = 1.0
        self.std = torch.tensor(std)
        ft = sem["ftype"]
        onehot_flat = {i for g in sem["onehot_groups"].values() for i in g}
        self.binary_idx = [i for i, t in enumerate(ft) if t == "binary" and i not in onehot_flat]
        self.count_idx = [i for i, t in enumerate(ft) if t == "count"]
        self.cont_idx = [i for i, t in enumerate(ft) if t == "continuous"]
        self.cat = {i: torch.tensor(sorted(v), dtype=torch.float64) for i, v in sem.get("categories", {}).items()}
        self.onehot_groups = [list(g) for g in sem["onehot_groups"].values()]
        lo = np.asarray(sem.get("lo", np.full(len(ft), -np.inf)), dtype=np.float64)
        hi = np.asarray(sem.get("hi", np.full(len(ft), np.inf)), dtype=np.float64)
        self.lo = torch.tensor(lo); self.hi = torch.tensor(hi)
        self.name = name

    def _raw(self, cf):
        d = cf.device
        return cf.double() * self.std.to(d) + self.mean.to(d), self.mean.to(d), self.std.to(d)

    def _snap_raw(self, raw: torch.Tensor) -> torch.Tensor:
        d = raw.device; lo = self.lo.to(d); hi = self.hi.to(d)
        out = raw.clone()
        for j in self.binary_idx:
            out[..., j] = torch.round(torch.clamp(out[..., j], 0.0, 1.0))
        for j, vals in self.cat.items():
            v = vals.to(d)
            idx = torch.argmin(torch.abs(out[..., j].unsqueeze(-1) - v), dim=-1)
            out[..., j] = v[idx]
        for j in self.count_idx:
            out[..., j] = torch.clamp(torch.round(out[..., j]), min=float(lo[j]), max=float(hi[j]))
        for j in self.cont_idx:
            out[..., j] = torch.clamp(out[..., j], min=float(lo[j]), max=float(hi[j]))
        for grp in self.onehot_groups:
            if len(grp) < 2:
                continue
            raws = torch.stack([out[..., j] for j in grp])
            win = int(torch.argmax(raws, dim=0).item()) if raws.dim() == 1 else 0
            for k, j in enumerate(grp):
                out[..., j] = 1.0 if k == win else 0.0
        return out

    def _clip_raw(self, raw: torch.Tensor) -> torch.Tensor:
        d = raw.device; lo = self.lo.to(d); hi = self.hi.to(d)
        out = raw.clone()
        for j in self.count_idx + self.cont_idx:
            out[..., j] = torch.clamp(out[..., j], min=float(lo[j]), max=float(hi[j]))
        for j in self.binary_idx:
            out[..., j] = torch.clamp(out[..., j], 0.0, 1.0)
        return out

    def apply(self, original: torch.Tensor, cf: torch.Tensor, force: bool = False) -> torch.Tensor:
        self._calls += 1
        raw, mean, std = self._raw(cf)
        if force or (self._calls % self.snap_every == 0):
            out = self._snap_raw(raw)
        else:
            out = self._clip_raw(raw)
        return ((out - mean) / std).to(cf.dtype)

    def project(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Tam (zorunlu) projeksiyon — son CF icin."""
        return self.apply(original, cf, force=True)

    def compute_violation(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        raw, _, _ = self._raw(cf)
        snapped = self._snap_raw(raw.detach())
        # normalize by std so that large-scale features do not dominate the penalty
        v = torch.abs(raw - snapped) / self.std.to(raw.device)
        return v.sum().to(cf.dtype)


def build_domain_manager(info, sem, feature_count, snap_every: int = 10):
    """Range+Sparsity (generic) + DomainConstraint (+immutable) iceren manager."""
    mgr = IoTConstraintManager(feature_count=feature_count, dataset_name=info.get("dataset", "generic"))
    scaler = info.get("scaler", None)
    mean = scaler.mean_ if scaler is not None else np.zeros(feature_count)
    std = scaler.scale_ if scaler is not None else np.ones(feature_count)
    imm_idx = [i for i, b in enumerate(sem["immutable"]) if b]
    if imm_idx:
        mgr.add_constraint(ImmutableConstraint(feature_indices=imm_idx))
    dc = DomainConstraint(mean, std, sem, snap_every=snap_every)
    mgr.domain_constraint = dc          # son projeksiyon icin erisim
    mgr.add_constraint(dc)
    onehot_flat = sorted({i for g in sem["onehot_groups"].values() for i in g})
    return mgr, {"binary": dc.binary_idx, "categorical": sorted(dc.cat.keys()),
                 "count": dc.count_idx, "continuous": dc.cont_idx,
                 "onehot_features": onehot_flat, "immutable": imm_idx}


def _feasible(cf_z, orig_z, sem, scaler, tol=0.05, rel_tol=0.01) -> bool:
    """CF raw uzayda tip- ve aralik-tutarli mi? (float64)"""
    cf64 = np.asarray(cf_z, dtype=np.float64).reshape(1, -1)
    raw = scaler.inverse_transform(cf64)[0] if scaler is not None else cf64[0]
    ftype = sem["ftype"]; lo = sem.get("lo"); hi = sem.get("hi"); cats = sem.get("categories", {})
    for j, t in enumerate(ftype):
        r = raw[j]
        if t == "immutable":
            if abs(float(cf_z[j]) - float(orig_z[j])) > 1e-5:
                return False
        elif t == "binary":
            if min(abs(r), abs(r - 1)) > tol:
                return False
        elif t == "categorical":
            allowed = cats.get(j)
            dist = min(abs(r - v) for v in allowed) if allowed else abs(r - round(r))
            if dist > tol:
                return False
        elif t == "count":
            if abs(r - round(r)) > tol:
                return False
            if lo is not None and np.isfinite(lo[j]) and (r < lo[j] - tol or r > hi[j] + tol):
                return False
        else:  # continuous
            if lo is not None and np.isfinite(lo[j]):
                span = max(hi[j] - lo[j], 1e-12)
                if r < lo[j] - rel_tol * span or r > hi[j] + rel_tol * span:
                    return False
    for grp in sem["onehot_groups"].values():
        if len(grp) >= 2 and abs(sum(raw[j] for j in grp) - 1.0) > tol:
            return False
    return True


def _best_cf(result, feature_count):
    cfs = getattr(result, "counterfactuals", None)
    if not cfs:
        return None
    c = cfs[0]
    return (c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c)).reshape(-1)


def _pred(model, cf_z, device):
    with torch.no_grad():
        return model(torch.tensor(np.asarray(cf_z), dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item()


def _raw_example(orig_z, cf_z, sem, fnames, scaler):
    if scaler is not None:
        o = scaler.inverse_transform(np.asarray(orig_z, dtype=np.float64).reshape(1, -1))[0]
        c = scaler.inverse_transform(np.asarray(cf_z, dtype=np.float64).reshape(1, -1))[0]
    else:
        o, c = orig_z, cf_z
    changed = np.where(np.abs(np.asarray(cf_z) - np.asarray(orig_z)) > 0.01)[0]
    return [{"feature": str(fnames[j]), "type": sem["ftype"][j],
             "original": round(float(o[j]), 4), "cf": round(float(c[j]), 4)} for j in changed]


def _semantics_summary(sem, fnames) -> Dict:
    ft = sem["ftype"]
    by_type: Dict[str, List[str]] = {}
    for j, t in enumerate(ft):
        by_type.setdefault(t, []).append(str(fnames[j]))
    return {"mode": sem.get("mode", "?"),
            "n_binary": ft.count("binary"), "n_categorical": ft.count("categorical"),
            "n_count": ft.count("count"), "n_continuous": ft.count("continuous"),
            "n_immutable": ft.count("immutable"),
            "onehot_groups": {k: len(v) for k, v in sem["onehot_groups"].items()},
            "features_by_type": by_type, "reasons": sem.get("reasons", {})}


def _attack_samples_permuted(model, X, y, n, device, rng: np.random.Generator, benign_class: int = 0):
    """Like C.get_attack_samples (first n instances predicted non-benign) but scans the test set in a
    seed-dependent random order, so that different runs use different query sets."""
    model.eval(); out = []
    order = rng.permutation(len(X))
    with torch.no_grad():
        for i in order:
            if model(X[i:i+1].to(device)).argmax(1).item() != benign_class:
                out.append(X[i])
                if len(out) >= n:
                    break
    return torch.stack(out) if out else None


def _evaluate_model(model, tag, Xtr, Xte, ytr, yte, info, sem, scaler, fnames, F,
                    n_samples, cf_max_iter, device, log, snap_every: int = 10,
                    query_rng: Optional[np.random.Generator] = None) -> Dict:
    attacks = (C.get_attack_samples(model, Xte, yte, n_samples, device) if query_rng is None
               else _attack_samples_permuted(model, Xte, yte, n_samples, device, query_rng))
    if attacks is None:
        log.log(f"[{tag}] attack sample yok"); return {"error": "no attacks"}
    acc = C.evaluate_model(model, C.make_loaders(Xtr, ytr, Xte, yte)[1], device)
    generic_mgr = IoTConstraintManager(feature_count=F, dataset_name=info.get("dataset", "generic"))
    domain_mgr, idx_info = build_domain_manager(info, sem, F, snap_every=snap_every)

    def gen_for(mgr, lam_feas):
        return CounterfactualGenerator(model=model, feature_count=F, num_cfs=C.NUM_CFS,
                                       use_dp=False, max_iter=cf_max_iter, device=device,
                                       constraint_manager=mgr, lambda_feas=lam_feas)
    g_generic = gen_for(generic_mgr, 1.0)
    g_domain = gen_for(domain_mgr, 2.0)

    res = {"generic": {"validity": [], "feasible": [], "sparsity": []},
           "domain": {"validity": [], "feasible": [], "sparsity": []},
           "examples": []}
    for i in range(len(attacks)):
        x = attacks[i]; xn = x.cpu().numpy()
        try:
            rg = g_generic.generate(x.to(device), target_class=0)
            cfg = _best_cf(rg, F)
            if cfg is not None:
                res["generic"]["validity"].append(1.0 if _pred(model, cfg, device) == 0 else 0.0)
                res["generic"]["feasible"].append(1.0 if _feasible(cfg, xn, sem, scaler) else 0.0)
                res["generic"]["sparsity"].append(int(np.sum(np.abs(cfg - xn) > 0.01)))
        except Exception as ex:  # noqa: BLE001
            log.log(f"[{tag}] generic gen error #{i}: {ex}")
        try:
            rd = g_domain.generate(x.to(device), target_class=0)
            cfd = _best_cf(rd, F)
            if cfd is not None:
                # son (zorunlu) projeksiyon: immutable geri yukle + tam snap; validity SONRA olculur
                cft = torch.tensor(cfd, dtype=torch.float32, device=device)
                cft = domain_mgr.apply_all(x.to(device), cft)
                cft = domain_mgr.domain_constraint.project(x.to(device), cft)
                cfd = cft.detach().cpu().numpy().reshape(-1)
                valid = _pred(model, cfd, device) == 0
                feas = _feasible(cfd, xn, sem, scaler)
                res["domain"]["validity"].append(1.0 if valid else 0.0)
                res["domain"]["feasible"].append(1.0 if feas else 0.0)
                res["domain"]["sparsity"].append(int(np.sum(np.abs(cfd - xn) > 0.01)))
                if valid and feas and len(res["examples"]) < 5:
                    res["examples"].append(_raw_example(xn, cfd, sem, fnames, scaler))
        except Exception as ex:  # noqa: BLE001
            log.log(f"[{tag}] domain gen error #{i}: {ex}")

    def ms(a): return (float(np.mean(a)) if a else 0.0, float(np.std(a)) if a else 0.0)
    g_act = [1.0 if (v == 1.0 and f == 1.0) else 0.0
             for v, f in zip(res["generic"]["validity"], res["generic"]["feasible"])]
    d_act = [1.0 if (v == 1.0 and f == 1.0) else 0.0
             for v, f in zip(res["domain"]["validity"], res["domain"]["feasible"])]
    out = {
        "accuracy": acc["accuracy"], "f1_macro": acc["f1_macro"],
        "generic": {"validity": ms(res["generic"]["validity"])[0],
                    "feasibility_rate": ms(res["generic"]["feasible"])[0],
                    "actionability": float(np.mean(g_act)) if g_act else 0.0,
                    "mean_sparsity": ms(res["generic"]["sparsity"])[0],
                    "n": len(res["generic"]["validity"])},
        "domain": {"validity": ms(res["domain"]["validity"])[0],
                   "feasibility_rate": ms(res["domain"]["feasible"])[0],
                   "actionability": float(np.mean(d_act)) if d_act else 0.0,
                   "mean_sparsity": ms(res["domain"]["sparsity"])[0],
                   "n": len(res["domain"]["validity"])},
        "raw_space_examples": res["examples"],
        "constraint_index_counts": {k: len(v) for k, v in idx_info.items()},
    }
    log.log(f"[{tag}] acc={acc['accuracy']:.3f}  GENERIC: validity={out['generic']['validity']:.3f} "
            f"feasibility={out['generic']['feasibility_rate']:.3f} actionability={out['generic']['actionability']:.3f} "
            f"sparsity={out['generic']['mean_sparsity']:.1f}  |  DOMAIN: validity={out['domain']['validity']:.3f} "
            f"feasibility={out['domain']['feasibility_rate']:.3f} actionability={out['domain']['actionability']:.3f} "
            f"sparsity={out['domain']['mean_sparsity']:.1f}  (n={out['domain']['n']})")
    return out


_AGG_KEYS = ("validity", "feasibility_rate", "actionability", "mean_sparsity")


def _aggregate_runs(per_run: List[Dict]) -> Dict:
    """Aggregate the per-run dicts returned by _evaluate_model (same condition, different seeds).

    The returned dict keeps the v2 layout (generic/domain -> validity, feasibility_rate,
    actionability, mean_sparsity, n) with MEANS over runs, and adds `<key>_std`, `<key>_raw`,
    `n_runs` and the list of per-run dicts under `runs`, so single-run consumers keep working."""
    ok = [r for r in per_run if "generic" in r]
    if not ok:
        return {"error": "no successful runs", "runs": per_run}
    out: Dict = {"accuracy": float(np.mean([r["accuracy"] for r in ok])),
                 "accuracy_std": float(np.std([r["accuracy"] for r in ok])),
                 "f1_macro": float(np.mean([r["f1_macro"] for r in ok])),
                 "f1_macro_std": float(np.std([r["f1_macro"] for r in ok])),
                 "n_runs": len(ok), "runs": per_run}
    for gen in ("generic", "domain"):
        block: Dict = {}
        for k in _AGG_KEYS:
            vals = [float(r[gen][k]) for r in ok]
            block[k] = float(np.mean(vals)); block[k + "_std"] = float(np.std(vals)); block[k + "_raw"] = vals
        block["n"] = int(sum(r[gen]["n"] for r in ok))          # pooled query count
        block["n_per_run"] = [int(r[gen]["n"]) for r in ok]
        # pooled (query-level) rates, weighting runs by their query counts
        for k in ("validity", "feasibility_rate", "actionability"):
            num = sum(float(r[gen][k]) * r[gen]["n"] for r in ok); den = max(block["n"], 1)
            block[k + "_pooled"] = num / den
        out[gen] = block
    out["raw_space_examples"] = [ex for r in ok for ex in r.get("raw_space_examples", [])][:5]
    out["constraint_index_counts"] = ok[0].get("constraint_index_counts", {})
    return out


def run(dataset: str, n_samples: int = 100, max_samples=None, epochs: int = C.EPOCHS,
        quick: bool = False, batch_size=None, cf_max_iter: int = C.CF_MAX_ITER,
        semantics: str = "data", model_conditions: str = "A,B",
        max_categorical: int = 32, snap_every: int = 10, n_runs: int = 1, out_suffix: str = ""):
    log = C.Runlog("exp16_domain_cf" + out_suffix, dataset)
    device = C.get_device()
    C.set_batch_size(batch_size)
    if quick:
        max_samples = max_samples or 40000; epochs = 3; n_samples = 25
    Xtr, Xte, ytr, yte, info = C.load_data(dataset, max_samples)
    scaler = info.get("scaler", None)
    fnames = info.get("feature_names", [f"f{i}" for i in range(info["num_features"])])
    F = info["num_features"]
    sem = (derive_semantics(fnames) if semantics == "name"
           else infer_semantics_data(Xtr, scaler, fnames, info, max_categorical=max_categorical))
    ssum = _semantics_summary(sem, fnames)
    log.log(f"semantics[{ssum['mode']}]: binary={ssum['n_binary']} categorical={ssum['n_categorical']} "
            f"count={ssum['n_count']} continuous={ssum['n_continuous']} immutable={ssum['n_immutable']} "
            f"onehot_groups={ssum['onehot_groups']} cf_max_iter={cf_max_iter} n_samples={n_samples} "
            f"snap_every={snap_every} n_runs={n_runs}")
    for t, names in ssum["features_by_type"].items():
        log.log(f"   {t:11s}: {names}")

    cw = C.class_weights_tensor(ytr, info["num_classes"], device)
    criterion = nn.CrossEntropyLoss(weight=cw)
    tr_loader, _ = C.make_loaders(Xtr, ytr, Xte, yte)
    noise_std = C.sigma_from_epsilon(C.REFERENCE_EPSILON) * C.DP_CLIP_NORM

    conds = [MODEL_COND_ALIASES.get(c.strip().upper(), c.strip()) for c in model_conditions.split(",")]
    per_run: Dict[str, List[Dict]] = {c: [] for c in conds}
    for run_i in range(max(1, n_runs)):
        seed = C.RANDOM_SEED + run_i
        # single run: identical to v2 (seed 42, first-n queries); multi-run: seed-dependent model init,
        # batch order and query set (random scan order of the test set)
        query_rng = None if n_runs <= 1 else np.random.default_rng(seed)
        if n_runs > 1:
            log.log(f"\n=== run {run_i+1}/{n_runs} (seed {seed}) ===")
        for cond in conds:
            clip, noise = C.CONDITIONS[cond]
            C.set_seed(seed)
            model = C.make_model("mlp", info).to(device)
            C.train_condition(model, tr_loader, use_clipping=clip, use_noise=noise,
                              noise_std=(noise_std if noise else 0.0), epochs=epochs,
                              lr=C.LEARNING_RATE, criterion=criterion, device=device)
            log.log(f"\n--- detector {cond} (clip={clip}, noise={noise}, seed={seed}) ---")
            r = _evaluate_model(model, cond, Xtr, Xte, ytr, yte, info, sem, scaler,
                                fnames, F, n_samples, cf_max_iter, device, log,
                                snap_every=snap_every, query_rng=query_rng)
            r["seed"] = seed
            per_run[cond].append(r)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    if n_runs <= 1:
        results = {c: per_run[c][0] for c in conds}
    else:
        results = {c: _aggregate_runs(per_run[c]) for c in conds}
        log.log("\n=== multi-run summary (mean +/- std over runs; pooled = query-level rate) ===")
        for c in conds:
            r = results[c]
            if "generic" not in r:
                continue
            for gen in ("generic", "domain"):
                b = r[gen]
                log.log(f"  {c:10s} {gen:7s}: validity={b['validity']:.3f}+/-{b['validity_std']:.3f} "
                        f"feasibility={b['feasibility_rate']:.3f}+/-{b['feasibility_rate_std']:.3f} "
                        f"actionability={b['actionability']:.3f}+/-{b['actionability_std']:.3f} "
                        f"(pooled {b['actionability_pooled']:.3f}, n={b['n']}) sparsity={b['mean_sparsity']:.1f}")

    summary = {"version": 2, "reference_epsilon": C.REFERENCE_EPSILON, "cf_max_iter": cf_max_iter,
               "n_samples": n_samples, "snap_every": snap_every, "n_runs": n_runs,
               "seeds": [C.RANDOM_SEED + i for i in range(max(1, n_runs))],
               "semantics": ssum, "models": results}
    # geriye-donuk anahtarlar (ilk kosulun / B varsa B'nin ozeti)
    ref = results.get("B_DP_SGD") or results.get(conds[0])
    if ref and "generic" in ref:
        summary["generic_baseline"] = ref["generic"]; summary["domain_constrained"] = ref["domain"]
    log.save_json(summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="EXP16 domain-constrained actionable CF (v2, data-driven semantics)")
    ap.add_argument("--dataset", default="ciciot2023",
                    choices=["ciciot2023", "edge-iiot", "toniot", "all"])
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--cf-max-iter", type=int, default=C.CF_MAX_ITER, help="CF optimizasyon iterasyonu (varsayilan 300)")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None, help="egitim batch boyutu (varsayilan 64)")
    ap.add_argument("--semantics", default="data", choices=["data", "name"],
                    help="tip cikarimi: data (v2, veriden) | name (v1, addan; eski sonuclari uretir)")
    ap.add_argument("--model-conditions", default="A,B", help="degerlendirilecek dedektorler: A,B (vars.) veya A,B,C,D")
    ap.add_argument("--max-categorical", type=int, default=32)
    ap.add_argument("--snap-every", type=int, default=10,
                    help="domain hard-snap periyodu (iterasyon); 1 = her adimda (v1 davranisi)")
    ap.add_argument("--n-runs", type=int, default=1,
                    help="tekrar sayisi (seed 42..42+n-1; >1 ise her kosuda farkli model init + farkli sorgu kumesi)")
    ap.add_argument("--out-suffix", default="", help="output folder suffix, e.g. _abcd -> exp16_domain_cf_abcd_<dataset>")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dsets = ["ciciot2023", "edge-iiot", "toniot"] if a.dataset == "all" else [a.dataset]
    for d in dsets:
        run(d, n_samples=a.n_samples, epochs=a.epochs, max_samples=a.max_samples,
            quick=a.quick, batch_size=a.batch_size, cf_max_iter=a.cf_max_iter,
            semantics=a.semantics, model_conditions=a.model_conditions,
            max_categorical=a.max_categorical, snap_every=a.snap_every, n_runs=a.n_runs,
            out_suffix=a.out_suffix)


if __name__ == "__main__":
    main()
