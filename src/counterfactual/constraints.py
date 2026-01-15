"""
IoT-Specific Constraints
========================
Counterfactual generation için IoT domain kısıtlamaları
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class FeatureInfo:
    """Feature metadata"""
    name: str
    index: int
    dtype: str  # 'continuous', 'categorical', 'binary'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    allowed_values: Optional[List] = None
    is_immutable: bool = False
    is_actionable: bool = True


class Constraint(ABC):
    """Base constraint class"""
    
    @abstractmethod
    def apply(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Constraint'i uygula ve düzeltilmiş CF döndür"""
        pass
    
    @abstractmethod
    def compute_violation(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Constraint ihlal miktarını hesapla (loss için)"""
        pass


class ImmutableConstraint(Constraint):
    """
    Değiştirilemez feature'lar için constraint
    Örn: Source IP, MAC Address, Device Type
    """
    
    def __init__(self, feature_indices: List[int], name: str = "immutable"):
        self.feature_indices = feature_indices
        self.name = name
    
    def apply(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Immutable feature'ları original'e geri döndür"""
        cf_constrained = cf.clone()
        cf_constrained[..., self.feature_indices] = original[..., self.feature_indices]
        return cf_constrained
    
    def compute_violation(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Immutable feature değişim miktarı"""
        diff = torch.abs(cf[..., self.feature_indices] - original[..., self.feature_indices])
        return torch.sum(diff)


class RangeConstraint(Constraint):
    """
    Feature değer aralığı constraint'i
    Örn: Port [0, 65535], Packet Size [64, 1500]
    """
    
    def __init__(
        self, 
        feature_indices: List[int],
        min_vals: torch.Tensor,
        max_vals: torch.Tensor,
        name: str = "range"
    ):
        self.feature_indices = feature_indices
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.name = name
    
    def apply(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Değerleri aralığa clip et"""
        cf_constrained = cf.clone()
        for i, idx in enumerate(self.feature_indices):
            cf_constrained[..., idx] = torch.clamp(
                cf[..., idx], 
                self.min_vals[i].item(), 
                self.max_vals[i].item()
            )
        return cf_constrained
    
    def compute_violation(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Aralık dışı değerlerin toplamı"""
        violation = torch.tensor(0.0, device=cf.device)
        for i, idx in enumerate(self.feature_indices):
            below = torch.relu(self.min_vals[i] - cf[..., idx])
            above = torch.relu(cf[..., idx] - self.max_vals[i])
            violation = violation + torch.sum(below + above)
        return violation


class CausalConstraint(Constraint):
    """
    Feature'lar arası nedensel ilişki constraint'i
    Örn: Total Bytes = Packet Count × Avg Packet Size
    """
    
    def __init__(
        self,
        source_indices: List[int],
        target_index: int,
        relation_fn: Callable,
        name: str = "causal"
    ):
        self.source_indices = source_indices
        self.target_index = target_index
        self.relation_fn = relation_fn
        self.name = name
    
    def apply(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Target'ı source'lardan hesapla"""
        cf_constrained = cf.clone()
        source_values = cf[..., self.source_indices]
        cf_constrained[..., self.target_index] = self.relation_fn(source_values)
        return cf_constrained
    
    def compute_violation(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Nedensel ilişki ihlali"""
        source_values = cf[..., self.source_indices]
        expected = self.relation_fn(source_values)
        actual = cf[..., self.target_index]
        return torch.sum(torch.abs(expected - actual))


class BinaryConstraint(Constraint):
    """
    Binary feature constraint (0 veya 1)
    Örn: TCP flags
    """
    
    def __init__(self, feature_indices: List[int], name: str = "binary"):
        self.feature_indices = feature_indices
        self.name = name
    
    def apply(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Binary değerlere yuvarla"""
        cf_constrained = cf.clone()
        cf_constrained[..., self.feature_indices] = torch.round(
            torch.clamp(cf[..., self.feature_indices], 0, 1)
        )
        return cf_constrained
    
    def compute_violation(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """0 veya 1'den sapma"""
        vals = cf[..., self.feature_indices]
        # 0'a veya 1'e en yakın değerden uzaklık
        dist_to_binary = torch.min(torch.abs(vals), torch.abs(vals - 1))
        return torch.sum(dist_to_binary)


class SparsityConstraint(Constraint):
    """
    Değişen feature sayısını sınırla
    Daha az feature değişikliği = daha açıklanabilir
    """
    
    def __init__(self, max_changes: int, threshold: float = 0.01, name: str = "sparsity"):
        self.max_changes = max_changes
        self.threshold = threshold
        self.name = name
    
    def apply(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """En önemli max_changes feature'ı değiştir, diğerlerini geri al"""
        cf_constrained = cf.clone()
        diff = torch.abs(cf - original)
        
        # Top-k değişikliği bul
        if diff.dim() == 1:
            _, top_indices = torch.topk(diff, min(self.max_changes, diff.size(0)))
            mask = torch.zeros_like(diff, dtype=torch.bool)
            mask[top_indices] = True
            cf_constrained[~mask] = original[~mask]
        else:
            # Batch için
            for i in range(cf.size(0)):
                _, top_indices = torch.topk(diff[i], min(self.max_changes, diff.size(1)))
                mask = torch.zeros_like(diff[i], dtype=torch.bool)
                mask[top_indices] = True
                cf_constrained[i, ~mask] = original[i, ~mask]
        
        return cf_constrained
    
    def compute_violation(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """max_changes'den fazla değişiklik varsa penalty"""
        diff = torch.abs(cf - original)
        num_changes = torch.sum(diff > self.threshold, dim=-1).float()
        excess = torch.relu(num_changes - self.max_changes)
        return torch.sum(excess)


class IoTConstraintManager:
    """
    Tüm IoT constraint'lerini yöneten sınıf
    
    Generic constraints kullanarak tüm dataset'ler için adil karşılaştırma sağlar.
    Dataset-specific constraints yerine normalized data üzerinde çalışan
    evrensel constraint'ler uygulanır.
    """
    
    def __init__(self, feature_count: int = 47, dataset_name: str = "generic"):
        self.feature_count = feature_count
        self.dataset_name = dataset_name
        self.constraints: List[Constraint] = []
        self._setup_generic_constraints()
    
    def _setup_generic_constraints(self):
        """
        Generic IoT Constraints - Tüm dataset'ler için adil karşılaştırma
        
        Bu constraint'ler StandardScaler sonrası data üzerinde çalışır.
        StandardScaler: mean=0, std=1, typical range [-3, 3]
        
        Constraints:
        1. Reasonable Range: Feature'lar [-5, 5] aralığında (StandardScaler uyumlu)
        2. Sparsity: Maksimum 10 feature değişikliği (actionability)
        3. Proximity: Değişiklikler orijinalden çok uzaklaşamaz (implicit)
        """
        
        all_indices = list(range(self.feature_count))
        
        # 1. Reasonable Range Constraint [-5, 5]
        # StandardScaler sonrası değerler genellikle [-3, 3] aralığında
        # [-5, 5] hem realitik hem de CF generation'a izin veriyor
        self.add_constraint(RangeConstraint(
            feature_indices=all_indices,
            min_vals=torch.ones(self.feature_count) * -5.0,
            max_vals=torch.ones(self.feature_count) * 5.0,
            name="reasonable_range"
        ))
        
        # 2. Sparsity Constraint
        # Maksimum 10 feature değişikliği - actionability için kritik
        # Güvenlik analistleri 10'dan fazla değişikliği yorumlayamaz
        self.add_constraint(SparsityConstraint(
            max_changes=10,
            threshold=0.01,  # %1'den küçük değişiklikler yok sayılır
            name="sparsity"
        ))
        
        # NOT: Non-negativity constraint kaldırıldı çünkü
        # StandardScaler sonrası negatif değerler normal
    
    def add_constraint(self, constraint: Constraint):
        """Yeni constraint ekle"""
        self.constraints.append(constraint)
    
    def apply_all(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Tüm constraint'leri sırayla uygula"""
        cf_constrained = cf.clone()
        for constraint in self.constraints:
            cf_constrained = constraint.apply(original, cf_constrained)
        return cf_constrained
    
    def compute_total_violation(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Tüm constraint ihlallerinin toplamı"""
        total_violation = torch.tensor(0.0, device=cf.device)
        for constraint in self.constraints:
            total_violation = total_violation + constraint.compute_violation(original, cf)
        return total_violation
    
    def get_violation_breakdown(
        self, 
        original: torch.Tensor, 
        cf: torch.Tensor
    ) -> Dict[str, float]:
        """Her constraint için ayrı ihlal miktarı"""
        breakdown = {}
        for constraint in self.constraints:
            violation = constraint.compute_violation(original, cf)
            breakdown[constraint.name] = violation.item()
        return breakdown
    
    def is_feasible(self, original: torch.Tensor, cf: torch.Tensor, threshold: float = 0.01) -> bool:
        """CF'nin tüm constraint'leri sağlayıp sağlamadığını kontrol et"""
        total_violation = self.compute_total_violation(original, cf)
        return total_violation.item() < threshold


# Test
if __name__ == "__main__":
    print("Testing Generic IoT Constraints...")
    print("="*60)
    
    # Test farklı feature sayıları ile (farklı dataset'ler için)
    test_configs = [
        ("CICIoT2023", 39),
        ("Edge-IIoT", 42),
        ("ToN-IoT", 44),
    ]
    
    for dataset_name, feature_count in test_configs:
        print(f"\n{dataset_name} (features={feature_count})")
        print("-"*40)
        
        batch_size = 4
        
        # Normalized dummy data [0, 1] aralığında
        original = torch.rand(batch_size, feature_count)
        cf = original + torch.randn(batch_size, feature_count) * 0.2
        
        # Constraint manager
        manager = IoTConstraintManager(feature_count=feature_count, dataset_name=dataset_name)
        
        # Apply constraints
        cf_constrained = manager.apply_all(original, cf)
        
        # Violations
        violation_before = manager.compute_total_violation(original, cf)
        violation_after = manager.compute_total_violation(original, cf_constrained)
        
        print(f"  Violation before: {violation_before.item():.4f}")
        print(f"  Violation after:  {violation_after.item():.4f}")
        
        # Check if CF values are in valid range
        cf_min = cf_constrained.min().item()
        cf_max = cf_constrained.max().item()
        print(f"  CF range: [{cf_min:.4f}, {cf_max:.4f}]")
        
        # Feasibility check
        is_feasible = manager.is_feasible(original, cf_constrained)
        print(f"  Feasible: {is_feasible}")
    
    print("\n" + "="*60)
    print("Generic Constraints test completed!")
    print("All datasets use the same constraints for fair comparison.")
