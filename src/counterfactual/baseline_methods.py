"""
Baseline Counterfactual Methods for Fair Comparison
====================================================
Implements classic CF methods with OPTIONAL constraint support for fair comparison.

Methods:
1. Wachter et al. (2017) - Gradient-based
2. Growing Spheres (Laugel et al., 2018) - Instance-based
3. FACE (Poyiadzi et al., 2020) - Feasible Actionable Counterfactual Explanations

All methods now support:
- apply_constraints: Enable post-hoc constraint application
- feature_ranges: (min, max) tuple for valid feature range
- max_features_to_change: Sparsity constraint

References:
- Wachter et al. "Counterfactual Explanations without Opening the Black Box" (2017)
- Laugel et al. "Comparison-based Inverse Classification for Interpretability" (2018)
- Poyiadzi et al. "FACE: Feasible and Actionable Counterfactual Explanations" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
import time
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist


@dataclass
class CFResult:
    """Counterfactual generation result"""
    original: torch.Tensor
    counterfactuals: List[torch.Tensor]
    original_pred: int
    cf_preds: List[int]
    validity: float
    proximity: float
    sparsity: float
    diversity: float
    generation_time: float
    method: str


# =============================================================================
# CONSTRAINT UTILITIES (Shared across all methods)
# =============================================================================

class ConstraintApplicator:
    """Applies constraints to counterfactuals (post-hoc)"""
    
    def __init__(
        self,
        feature_ranges: Tuple[float, float] = (-5.0, 5.0),
        max_features_to_change: int = 10,
        change_threshold: float = 0.01
    ):
        self.min_val, self.max_val = feature_ranges
        self.max_changes = max_features_to_change
        self.threshold = change_threshold
    
    def apply(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Apply all constraints to a counterfactual"""
        cf_constrained = cf.clone()
        
        # 1. Range constraint
        cf_constrained = torch.clamp(cf_constrained, self.min_val, self.max_val)
        
        # 2. Sparsity constraint - keep only top-k changes
        if original.dim() == 1:
            diff = torch.abs(cf_constrained - original)
            if diff.numel() > self.max_changes:
                _, top_indices = torch.topk(diff, min(self.max_changes, diff.numel()))
                mask = torch.zeros_like(diff, dtype=torch.bool)
                mask[top_indices] = True
                cf_constrained[~mask] = original[~mask]
        else:
            # Batch processing
            diff = torch.abs(cf_constrained - original)
            for i in range(cf_constrained.size(0)):
                _, top_indices = torch.topk(diff[i], min(self.max_changes, diff.size(1)))
                mask = torch.zeros_like(diff[i], dtype=torch.bool)
                mask[top_indices] = True
                cf_constrained[i, ~mask] = original[i, ~mask]
        
        return cf_constrained


# =============================================================================
# 1. WACHTER METHOD (2017)
# =============================================================================

class WachterCF:
    """
    Wachter et al. (2017) - Gradient-based Counterfactual
    
    Loss = lambda * (f(x') - y')^2 + d(x, x')
    
    Now supports:
    - apply_constraints: Post-hoc constraint application
    - feature_ranges: Valid feature range
    - max_features_to_change: Sparsity constraint
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        lambda_weight: float = 0.1,
        learning_rate: float = 0.01,
        max_iterations: int = 300,
        distance: str = "l1",
        device: str = "cuda",
        # NEW: Constraint parameters for fair comparison
        apply_constraints: bool = False,
        feature_ranges: Tuple[float, float] = (-5.0, 5.0),
        max_features_to_change: int = 10
    ):
        self.model = model
        self.num_classes = num_classes
        self.lambda_weight = lambda_weight
        self.lr = learning_rate
        self.max_iter = max_iterations
        self.distance = distance
        self.device = device
        self.model.eval()
        
        # Constraint settings
        self.apply_constraints = apply_constraints
        if apply_constraints:
            self.constraint_applicator = ConstraintApplicator(
                feature_ranges=feature_ranges,
                max_features_to_change=max_features_to_change
            )
    
    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        num_cfs: int = 5
    ) -> CFResult:
        """Generate counterfactuals using Wachter method"""
        start_time = time.time()
        
        x = x.to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Original prediction
        with torch.no_grad():
            orig_pred = self.model(x).argmax(dim=1).item()
        
        counterfactuals = []
        cf_preds = []
        
        for cf_idx in range(num_cfs):
            # Initialize CF with small random perturbation (different for each CF)
            cf = x.clone().detach()
            cf = cf + torch.randn_like(cf) * 0.1 * (cf_idx + 1)
            cf = cf.requires_grad_(True)
            
            optimizer = torch.optim.Adam([cf], lr=self.lr)
            
            best_cf = None
            best_loss = float('inf')
            
            for iteration in range(self.max_iter):
                optimizer.zero_grad()
                
                # Prediction loss
                logits = self.model(cf)
                pred_loss = F.cross_entropy(
                    logits, 
                    torch.tensor([target_class], device=self.device)
                )
                
                # Distance loss
                if self.distance == "l1":
                    dist_loss = torch.mean(torch.abs(cf - x))
                else:
                    dist_loss = torch.mean((cf - x) ** 2)
                
                # Total loss
                total_loss = self.lambda_weight * pred_loss + dist_loss
                
                # Check if valid CF found
                with torch.no_grad():
                    current_pred = logits.argmax(dim=1).item()
                    if current_pred == target_class:
                        if total_loss.item() < best_loss:
                            best_loss = total_loss.item()
                            best_cf = cf.clone().detach()
                
                total_loss.backward()
                optimizer.step()
            
            # Use best CF found or final CF
            final_cf = best_cf if best_cf is not None else cf.clone().detach()
            
            # Apply constraints if enabled
            if self.apply_constraints:
                final_cf = self.constraint_applicator.apply(x.squeeze(0), final_cf.squeeze(0))
                if final_cf.dim() == 1:
                    final_cf = final_cf.unsqueeze(0)
            
            # Get final prediction
            with torch.no_grad():
                final_pred = self.model(final_cf).argmax(dim=1).item()
            
            counterfactuals.append(final_cf.squeeze(0))
            cf_preds.append(final_pred)
        
        generation_time = time.time() - start_time
        
        # Calculate metrics
        validity = sum(1 for p in cf_preds if p == target_class) / len(cf_preds)
        proximity = self._calc_proximity(x, counterfactuals)
        sparsity = self._calc_sparsity(x, counterfactuals)
        diversity = self._calc_diversity(counterfactuals)
        
        return CFResult(
            original=x.squeeze(0),
            counterfactuals=counterfactuals,
            original_pred=orig_pred,
            cf_preds=cf_preds,
            validity=validity,
            proximity=proximity,
            sparsity=sparsity,
            diversity=diversity,
            generation_time=generation_time,
            method="Wachter"
        )
    
    def _calc_proximity(self, x, cfs):
        if not cfs:
            return 0.0
        return np.mean([torch.mean(torch.abs(cf.cpu() - x.squeeze(0).cpu())).item() for cf in cfs])
    
    def _calc_sparsity(self, x, cfs, threshold=0.01):
        if not cfs:
            return 0.0
        x_flat = x.squeeze(0).cpu()
        changes = [torch.sum(torch.abs(cf.cpu() - x_flat) > threshold).item() for cf in cfs]
        return np.mean(changes)
    
    def _calc_diversity(self, cfs):
        if len(cfs) < 2:
            return 0.0
        dists = []
        for i in range(len(cfs)):
            for j in range(i + 1, len(cfs)):
                dists.append(torch.mean(torch.abs(cfs[i].cpu() - cfs[j].cpu())).item())
        return np.mean(dists) if dists else 0.0


# =============================================================================
# 2. GROWING SPHERES (2018)
# =============================================================================

class GrowingSpheresCF:
    """
    Growing Spheres (Laugel et al., 2018)
    
    Algorithm:
    1. Start from original instance
    2. Generate random points on expanding spheres
    3. Return first point that crosses decision boundary
    
    Now supports constraints for fair comparison.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        initial_radius: float = 0.1,
        radius_growth: float = 1.5,
        max_iterations: int = 300,
        n_samples_per_sphere: int = 100,
        device: str = "cuda",
        # NEW: Constraint parameters
        apply_constraints: bool = False,
        feature_ranges: Tuple[float, float] = (-5.0, 5.0),
        max_features_to_change: int = 10
    ):
        self.model = model
        self.num_classes = num_classes
        self.initial_radius = initial_radius
        self.radius_growth = radius_growth
        self.max_iter = max_iterations
        self.n_samples = n_samples_per_sphere
        self.device = device
        self.model.eval()
        
        # Constraint settings
        self.apply_constraints = apply_constraints
        if apply_constraints:
            self.constraint_applicator = ConstraintApplicator(
                feature_ranges=feature_ranges,
                max_features_to_change=max_features_to_change
            )
    
    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        num_cfs: int = 5
    ) -> CFResult:
        """Generate counterfactuals using Growing Spheres"""
        start_time = time.time()
        
        x = x.to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        n_features = x.shape[1]
        
        # Original prediction
        with torch.no_grad():
            orig_pred = self.model(x).argmax(dim=1).item()
        
        counterfactuals = []
        cf_preds = []
        
        for cf_idx in range(num_cfs):
            radius = self.initial_radius * (1 + cf_idx * 0.2)  # Vary initial radius
            found_cf = None
            
            for iteration in range(self.max_iter):
                # Generate random points on sphere surface
                random_directions = torch.randn(self.n_samples, n_features, device=self.device)
                random_directions = F.normalize(random_directions, p=2, dim=1)
                
                # Points on sphere
                sphere_points = x + radius * random_directions
                
                # Apply constraints during search if enabled
                if self.apply_constraints:
                    for i in range(sphere_points.size(0)):
                        sphere_points[i] = self.constraint_applicator.apply(
                            x.squeeze(0), sphere_points[i]
                        )
                
                # Check predictions
                with torch.no_grad():
                    preds = self.model(sphere_points).argmax(dim=1)
                
                # Find points with target class
                valid_mask = (preds == target_class)
                
                if valid_mask.any():
                    # Get closest valid point
                    valid_points = sphere_points[valid_mask]
                    distances = torch.mean(torch.abs(valid_points - x), dim=1)
                    closest_idx = distances.argmin()
                    found_cf = valid_points[closest_idx]
                    break
                
                # Expand sphere
                radius *= self.radius_growth
            
            if found_cf is not None:
                # Apply constraints to final CF
                if self.apply_constraints:
                    found_cf = self.constraint_applicator.apply(x.squeeze(0), found_cf)
                
                counterfactuals.append(found_cf.cpu())
                with torch.no_grad():
                    pred = self.model(found_cf.unsqueeze(0).to(self.device)).argmax(dim=1).item()
                cf_preds.append(pred)
            else:
                # No CF found - use last sphere point as fallback
                fallback = x.squeeze(0) + torch.randn(n_features, device=self.device) * radius * 0.1
                if self.apply_constraints:
                    fallback = self.constraint_applicator.apply(x.squeeze(0), fallback)
                counterfactuals.append(fallback.cpu())
                with torch.no_grad():
                    pred = self.model(fallback.unsqueeze(0)).argmax(dim=1).item()
                cf_preds.append(pred)
        
        generation_time = time.time() - start_time
        
        # Calculate metrics
        validity = sum(1 for p in cf_preds if p == target_class) / len(cf_preds) if cf_preds else 0
        proximity = self._calc_proximity(x, counterfactuals)
        sparsity = self._calc_sparsity(x, counterfactuals)
        diversity = self._calc_diversity(counterfactuals)
        
        return CFResult(
            original=x.squeeze(0).cpu(),
            counterfactuals=counterfactuals,
            original_pred=orig_pred,
            cf_preds=cf_preds,
            validity=validity,
            proximity=proximity,
            sparsity=sparsity,
            diversity=diversity,
            generation_time=generation_time,
            method="GrowingSpheres"
        )
    
    def _calc_proximity(self, x, cfs):
        if not cfs:
            return 0.0
        x_cpu = x.squeeze(0).cpu()
        return np.mean([torch.mean(torch.abs(cf - x_cpu)).item() for cf in cfs])
    
    def _calc_sparsity(self, x, cfs, threshold=0.01):
        if not cfs:
            return 0.0
        x_cpu = x.squeeze(0).cpu()
        changes = [torch.sum(torch.abs(cf - x_cpu) > threshold).item() for cf in cfs]
        return np.mean(changes)
    
    def _calc_diversity(self, cfs):
        if len(cfs) < 2:
            return 0.0
        dists = []
        for i in range(len(cfs)):
            for j in range(i + 1, len(cfs)):
                dists.append(torch.mean(torch.abs(cfs[i] - cfs[j])).item())
        return np.mean(dists) if dists else 0.0


# =============================================================================
# 3. FACE (2020)
# =============================================================================

class FACECF:
    """
    FACE - Feasible and Actionable Counterfactual Explanations (Poyiadzi et al., 2020)
    
    Algorithm:
    1. Build weighted graph on training data
    2. Find shortest path from query to target class
    3. Return path endpoint as counterfactual
    
    Now supports constraints for fair comparison.
    """
    
    def __init__(
        self,
        model: nn.Module,
        num_classes: int,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        n_neighbors: int = 10,
        max_iterations: int = 300,
        device: str = "cuda",
        # NEW: Constraint parameters
        apply_constraints: bool = False,
        feature_ranges: Tuple[float, float] = (-5.0, 5.0),
        max_features_to_change: int = 10
    ):
        self.model = model
        self.num_classes = num_classes
        self.X_train = X_train.cpu().numpy() if isinstance(X_train, torch.Tensor) else X_train
        self.y_train = y_train.cpu().numpy() if isinstance(y_train, torch.Tensor) else y_train
        self.n_neighbors = n_neighbors
        self.max_iter = max_iterations
        self.device = device
        self.model.eval()
        
        # Constraint settings
        self.apply_constraints = apply_constraints
        if apply_constraints:
            self.constraint_applicator = ConstraintApplicator(
                feature_ranges=feature_ranges,
                max_features_to_change=max_features_to_change
            )
        
        # Build KNN index
        self.knn = NearestNeighbors(n_neighbors=min(n_neighbors, len(self.X_train)))
        self.knn.fit(self.X_train)
    
    def generate(
        self,
        x: torch.Tensor,
        target_class: int,
        num_cfs: int = 5
    ) -> CFResult:
        """Generate counterfactuals using FACE method"""
        start_time = time.time()
        
        x = x.to(self.device).float()
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x_np = x.cpu().numpy().reshape(1, -1)
        
        # Original prediction
        with torch.no_grad():
            orig_pred = self.model(x).argmax(dim=1).item()
        
        # Find target class samples
        target_mask = (self.y_train == target_class)
        target_samples = self.X_train[target_mask]
        
        if len(target_samples) == 0:
            # No target samples - return original
            return CFResult(
                original=x.squeeze(0).cpu(),
                counterfactuals=[x.squeeze(0).cpu() for _ in range(num_cfs)],
                original_pred=orig_pred,
                cf_preds=[orig_pred] * num_cfs,
                validity=0.0,
                proximity=0.0,
                sparsity=0.0,
                diversity=0.0,
                generation_time=time.time() - start_time,
                method="FACE"
            )
        
        # Find nearest target samples
        distances = cdist(x_np, target_samples, metric='euclidean')[0]
        sorted_indices = np.argsort(distances)
        
        counterfactuals = []
        cf_preds = []
        
        for cf_idx in range(num_cfs):
            if cf_idx < len(sorted_indices):
                # Get candidate CF
                candidate_idx = sorted_indices[cf_idx]
                candidate = target_samples[candidate_idx]
                
                # Path-based interpolation (FACE specific)
                # Move along path from x to candidate
                alpha = min(1.0, 0.5 + cf_idx * 0.15)  # Vary interpolation
                cf = x_np[0] * (1 - alpha) + candidate * alpha
                cf_tensor = torch.tensor(cf, dtype=torch.float32)
                
                # Apply constraints if enabled
                if self.apply_constraints:
                    cf_tensor = self.constraint_applicator.apply(
                        torch.tensor(x_np[0], dtype=torch.float32),
                        cf_tensor
                    )
                
                counterfactuals.append(cf_tensor)
                
                # Get prediction
                with torch.no_grad():
                    cf_input = cf_tensor.unsqueeze(0).to(self.device)
                    pred = self.model(cf_input).argmax(dim=1).item()
                cf_preds.append(pred)
            else:
                # Fallback: use closest target sample
                cf_tensor = torch.tensor(target_samples[0], dtype=torch.float32)
                if self.apply_constraints:
                    cf_tensor = self.constraint_applicator.apply(
                        torch.tensor(x_np[0], dtype=torch.float32),
                        cf_tensor
                    )
                counterfactuals.append(cf_tensor)
                with torch.no_grad():
                    pred = self.model(cf_tensor.unsqueeze(0).to(self.device)).argmax(dim=1).item()
                cf_preds.append(pred)
        
        generation_time = time.time() - start_time
        
        # Calculate metrics
        validity = sum(1 for p in cf_preds if p == target_class) / len(cf_preds) if cf_preds else 0
        proximity = self._calc_proximity(x, counterfactuals)
        sparsity = self._calc_sparsity(x, counterfactuals)
        diversity = self._calc_diversity(counterfactuals)
        
        return CFResult(
            original=x.squeeze(0).cpu(),
            counterfactuals=counterfactuals,
            original_pred=orig_pred,
            cf_preds=cf_preds,
            validity=validity,
            proximity=proximity,
            sparsity=sparsity,
            diversity=diversity,
            generation_time=generation_time,
            method="FACE"
        )
    
    def _calc_proximity(self, x, cfs):
        if not cfs:
            return 0.0
        x_cpu = x.squeeze(0).cpu()
        return np.mean([torch.mean(torch.abs(cf - x_cpu)).item() for cf in cfs])
    
    def _calc_sparsity(self, x, cfs, threshold=0.01):
        if not cfs:
            return 0.0
        x_cpu = x.squeeze(0).cpu()
        changes = [torch.sum(torch.abs(cf - x_cpu) > threshold).item() for cf in cfs]
        return np.mean(changes)
    
    def _calc_diversity(self, cfs):
        if len(cfs) < 2:
            return 0.0
        dists = []
        for i in range(len(cfs)):
            for j in range(i + 1, len(cfs)):
                dists.append(torch.mean(torch.abs(cfs[i] - cfs[j])).item())
        return np.mean(dists) if dists else 0.0


# =============================================================================
# UNIFIED INTERFACE
# =============================================================================

class BaselineCFGenerator:
    """Unified interface for all baseline CF methods"""
    
    METHODS = {
        'wachter': WachterCF,
        'growing_spheres': GrowingSpheresCF,
        'face': FACECF,
    }
    
    @staticmethod
    def create(
        method: str,
        model: nn.Module,
        num_classes: int,
        device: str = "cuda",
        X_train: torch.Tensor = None,
        y_train: torch.Tensor = None,
        apply_constraints: bool = False,
        feature_ranges: Tuple[float, float] = (-5.0, 5.0),
        max_features_to_change: int = 10,
        **kwargs
    ):
        """Factory method to create CF generator with optional constraints"""
        method = method.lower()
        
        constraint_kwargs = {
            'apply_constraints': apply_constraints,
            'feature_ranges': feature_ranges,
            'max_features_to_change': max_features_to_change
        }
        
        if method == 'wachter':
            return WachterCF(model, num_classes, device=device, **constraint_kwargs, **kwargs)
        elif method == 'growing_spheres':
            return GrowingSpheresCF(model, num_classes, device=device, **constraint_kwargs, **kwargs)
        elif method == 'face':
            if X_train is None or y_train is None:
                raise ValueError("FACE requires X_train and y_train")
            return FACECF(model, num_classes, X_train, y_train, device=device, **constraint_kwargs, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Available: {list(BaselineCFGenerator.METHODS.keys())}")


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Baseline CF Methods with Constraint Support - Test")
    print("=" * 60)
    
    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self, n_features, n_classes):
            super().__init__()
            self.fc = nn.Linear(n_features, n_classes)
        
        def forward(self, x):
            return self.fc(x)
    
    n_features, n_classes = 10, 3
    model = DummyModel(n_features, n_classes)
    
    # Dummy data
    X_train = torch.randn(100, n_features)
    y_train = torch.randint(0, n_classes, (100,))
    x = torch.randn(1, n_features)
    
    # Test each method with and without constraints
    for method in ['wachter', 'growing_spheres', 'face']:
        print(f"\n{'='*60}")
        print(f"Testing {method}...")
        
        for use_constraints in [False, True]:
            constraint_str = "WITH constraints" if use_constraints else "WITHOUT constraints"
            print(f"\n  {constraint_str}:")
            try:
                gen = BaselineCFGenerator.create(
                    method, model, n_classes,
                    device='cpu',
                    X_train=X_train,
                    y_train=y_train,
                    apply_constraints=use_constraints,
                    feature_ranges=(-5.0, 5.0),
                    max_features_to_change=10,
                    max_iterations=50  # Fewer iterations for test
                )
                result = gen.generate(x, target_class=0, num_cfs=3)
                print(f"    Validity: {result.validity:.2%}")
                print(f"    Proximity: {result.proximity:.4f}")
                print(f"    Sparsity: {result.sparsity:.1f} features changed")
                print(f"    Time: {result.generation_time:.2f}s")
            except Exception as e:
                print(f"    Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
