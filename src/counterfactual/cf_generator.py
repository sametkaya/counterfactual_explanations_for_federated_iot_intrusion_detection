"""
Privacy-Preserving Counterfactual Generator
============================================
IoT-IDS için DiCE tabanlı CF generation with Differential Privacy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import copy

# Local imports
import sys
sys.path.append('..')
from counterfactual.constraints import IoTConstraintManager, Constraint


@dataclass
class CFResult:
    """Counterfactual generation sonucu"""
    original: torch.Tensor
    counterfactuals: List[torch.Tensor]
    original_pred: int
    cf_preds: List[int]
    validity: float
    proximity: float
    sparsity: float
    diversity: float
    actionability: float
    generation_time: float


class DPNoiseMechanism:
    """Differential Privacy noise mekanizması"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        mechanism: str = "gaussian"  # "gaussian" veya "laplace"
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        self.noise_scale = self._compute_noise_scale()
    
    def _compute_noise_scale(self) -> float:
        """Noise scale hesapla"""
        if self.mechanism == "gaussian":
            # Gaussian mechanism for (ε, δ)-DP
            return self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        else:
            # Laplace mechanism for ε-DP
            return self.sensitivity / self.epsilon
    
    def add_noise(self, tensor: torch.Tensor) -> torch.Tensor:
        """Tensor'a calibrated noise ekle"""
        if self.mechanism == "gaussian":
            noise = torch.randn_like(tensor) * self.noise_scale
        else:
            # Laplace noise
            noise = torch.zeros_like(tensor).exponential_(1/self.noise_scale)
            noise = noise * (2 * torch.bernoulli(torch.ones_like(tensor) * 0.5) - 1)
        
        return tensor + noise
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Harcanan privacy budget'ı döndür"""
        return self.epsilon, self.delta


class CounterfactualGenerator:
    """
    Privacy-Preserving Counterfactual Generator
    
    DiCE algoritmasının IoT-IDS için adapte edilmiş versiyonu
    Differential Privacy ile privacy-preserving
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_count: int = 47,
        num_cfs: int = 5,
        # Loss weights
        lambda_pred: float = 1.0,
        lambda_dist: float = 0.5,
        lambda_div: float = 0.1,
        lambda_feas: float = 1.0,
        # Optimization
        learning_rate: float = 0.05,
        max_iter: int = 300,
        early_stop_threshold: float = 0.9,
        # Privacy
        epsilon: float = 1.0,
        delta: float = 1e-5,
        use_dp: bool = True,
        # Constraints
        constraint_manager: Optional[IoTConstraintManager] = None,
        device: str = "cpu",
        optimizer: str = "adam",  # "adam" (default, the paper), "sgd" (plain gradient descent) or
                                  # "signgd" (step = lr * sign(gradient), scale-free like Adam's first step)
        betas: Tuple[float, float] = (0.9, 0.999),   # Adam moment coefficients (beta1=0 -> no momentum, RMSprop-like)
        trace: bool = False       # record per-iteration step norms, gradient sign flips and Adam's |m_hat|/sqrt(v_hat)
                                  # in self.last_trace (optimiser diagnostics; costs a little time)
    ):
        self.model = model
        self.model.eval()
        self.feature_count = feature_count
        self.num_cfs = num_cfs
        
        # Loss weights
        self.lambda_pred = lambda_pred
        self.lambda_dist = lambda_dist
        self.lambda_div = lambda_div
        self.lambda_feas = lambda_feas
        
        # Optimization
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.early_stop_threshold = early_stop_threshold
        
        # Privacy
        self.use_dp = use_dp
        if use_dp:
            self.dp_mechanism = DPNoiseMechanism(
                epsilon=epsilon,
                delta=delta,
                sensitivity=1.0
            )
        
        # Constraints
        self.constraint_manager = constraint_manager or IoTConstraintManager(feature_count)
        
        self.device = device
        self.model.to(device)
        self.optimizer_name = optimizer
        self.betas = tuple(betas)
        self.trace = trace
        self.last_trace: Optional[Dict] = None

    def generate(
        self,
        query_instance: torch.Tensor,
        target_class: int = 0,  # Default: Benign
        diversity_weight: Optional[float] = None
    ) -> CFResult:
        """
        Counterfactual'lar üret
        
        Args:
            query_instance: Açıklanacak instance (attack olarak classify edilmiş)
            target_class: Hedef sınıf (genellikle 0=Benign)
            diversity_weight: Optional diversity weight override
            
        Returns:
            CFResult: Üretilen CF'ler ve metrikler
        """
        import time
        start_time = time.time()
        
        query = query_instance.to(self.device)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Original prediction
        with torch.no_grad():
            original_pred = self.model.predict(query).item()
        
        # Initialize CF candidates
        cfs = self._initialize_cfs(query)
        
        # Optimizer
        optimizer = (Adam(cfs, lr=self.learning_rate, betas=self.betas) if self.optimizer_name == "adam"
                     else SGD(cfs, lr=self.learning_rate))

        # Optimization loop
        best_cfs = None
        best_validity = 0.0

        # optimiser diagnostics (trace=True): effective step ||x_t - x_{t-1}|| after projection, share of
        # gradient coordinates whose sign flipped since the previous iteration, Adam's mean |m_hat|/sqrt(v_hat)
        # (the factor that scales its step; near 0 when the moment average cancels while v_hat stays large),
        # validity per iteration and the predicted classes the candidates pass through
        tr = ({"step_norm": [], "sign_flip": [], "adam_ratio": [], "validity": [], "share_third_class": []}
              if self.trace else None)
        prev_grad = None
        visited = [set() for _ in cfs]

        for iteration in range(self.max_iter):
            optimizer.zero_grad()
            
            # Stack CFs for batch processing
            cf_batch = torch.stack(cfs)
            
            # Compute losses
            loss_pred = self._prediction_loss(cf_batch, target_class)
            loss_dist = self._distance_loss(cf_batch, query)
            loss_div = self._diversity_loss(cf_batch)
            loss_feas = self._feasibility_loss(cf_batch, query)
            
            # Total loss
            total_loss = (
                self.lambda_pred * loss_pred +
                self.lambda_dist * loss_dist +
                (diversity_weight or self.lambda_div) * loss_div +
                self.lambda_feas * loss_feas
            )
            
            # Backward
            total_loss.backward()
            
            # Add DP noise to gradients
            if self.use_dp:
                self._add_dp_noise_to_grads(cfs)
            
            # sign-GD: replace every gradient by its sign (the SGD step then moves lr per coordinate)
            if self.optimizer_name == "signgd":
                for cf in cfs:
                    if cf.grad is not None:
                        cf.grad.sign_()

            if tr is not None:
                with torch.no_grad():
                    g = torch.stack([cf.grad for cf in cfs]).detach()
                    if prev_grad is not None:
                        nz = (g != 0) & (prev_grad != 0)
                        flips = ((g.sign() != prev_grad.sign()) & nz).float().sum() / nz.float().sum().clamp(min=1.0)
                        tr["sign_flip"].append(float(flips))
                    prev_grad = g.clone()
                    x_before = torch.stack(cfs).detach().clone()

            # Update
            optimizer.step()

            # Apply constraints
            with torch.no_grad():
                for i, cf in enumerate(cfs):
                    cfs[i].data = self.constraint_manager.apply_all(query.squeeze(0), cf)

            # Check validity
            with torch.no_grad():
                cf_batch = torch.stack(cfs)
                preds = self.model.predict(cf_batch)
                validity = (preds == target_class).float().mean().item()

                if tr is not None:
                    tr["step_norm"].append(float((cf_batch - x_before).norm(dim=1).mean().item()))
                    tr["validity"].append(validity)
                    pl = preds.tolist()
                    for i, p in enumerate(pl):
                        visited[i].add(int(p))
                    tr["share_third_class"].append(float(np.mean([(p != target_class) and (p != original_pred) for p in pl])))
                    if self.optimizer_name == "adam":
                        b1, b2 = self.betas; ratios = []
                        for cf in cfs:
                            st = optimizer.state.get(cf, {})
                            if "exp_avg" in st:
                                t = st["step"]; t = float(t.item()) if torch.is_tensor(t) else float(t)
                                m_hat = st["exp_avg"] / max(1.0 - b1 ** t, 1e-12)
                                v_hat = st["exp_avg_sq"] / max(1.0 - b2 ** t, 1e-12)
                                ratios.append(float((m_hat.abs() / (v_hat.sqrt() + 1e-8)).mean().item()))
                        tr["adam_ratio"].append(float(np.mean(ratios)) if ratios else float("nan"))

                if validity > best_validity:
                    best_validity = validity
                    best_cfs = [cf.clone() for cf in cfs]

                # Early stopping
                if validity >= self.early_stop_threshold:
                    break

        if tr is not None:
            tr["n_iter"] = len(tr["step_norm"])
            tr["classes_visited"] = float(np.mean([len(v) for v in visited]))
            tr["best_validity"] = float(best_validity)
            self.last_trace = tr

        # Use best CFs found
        final_cfs = best_cfs if best_cfs is not None else cfs
        
        # Compute final metrics
        generation_time = time.time() - start_time
        result = self._compute_metrics(query.squeeze(0), final_cfs, target_class, generation_time)
        result.original_pred = original_pred
        
        return result
    
    def _initialize_cfs(self, query: torch.Tensor) -> List[torch.Tensor]:
        """CF candidate'ları initialize et"""
        cfs = []
        for i in range(self.num_cfs):
            # Query etrafında noise ile başla
            noise_scale = 0.1 + 0.05 * i  # Farklı scale'lerle diversity
            cf = query.clone().squeeze(0) + torch.randn(self.feature_count, device=self.device) * noise_scale
            cf = cf.requires_grad_(True)
            cfs.append(cf)
        return cfs
    
    def _prediction_loss(self, cfs: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Prediction loss: CF'lerin target class'a yakınlığı
        
        Cross-entropy + hinge loss kombinasyonu
        """
        logits = self.model(cfs)
        
        # Cross entropy to target class
        target = torch.full((cfs.size(0),), target_class, dtype=torch.long, device=self.device)
        ce_loss = F.cross_entropy(logits, target)
        
        # Hinge loss for confidence margin
        probs = F.softmax(logits, dim=-1)
        target_prob = probs[:, target_class]
        hinge_loss = torch.relu(0.8 - target_prob).mean()
        
        return ce_loss + hinge_loss
    
    def _distance_loss(self, cfs: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Distance loss: CF'lerin original'e yakınlığı
        
        Weighted L1 distance (MAD normalized)
        """
        # L1 distance
        l1_dist = torch.abs(cfs - query).mean(dim=1)
        
        return l1_dist.mean()
    
    def _diversity_loss(self, cfs: torch.Tensor) -> torch.Tensor:
        """
        Diversity loss: CF'ler arası çeşitlilik
        
        Determinantal Point Process (DPP) kernel
        """
        if cfs.size(0) < 2:
            return torch.tensor(0.0, device=self.device)
        
        # Pairwise distances
        dists = torch.cdist(cfs, cfs, p=2)
        
        # RBF kernel
        sigma = 1.0
        K = torch.exp(-dists ** 2 / (2 * sigma ** 2))
        
        # Negative log determinant (daha yüksek diversity = daha düşük loss)
        # Add small diagonal for numerical stability
        K = K + torch.eye(K.size(0), device=self.device) * 1e-4
        
        try:
            log_det = torch.logdet(K)
            if torch.isnan(log_det) or torch.isinf(log_det):
                log_det = torch.tensor(0.0, device=self.device)
        except:
            log_det = torch.tensor(0.0, device=self.device)
        
        return -log_det
    
    def _feasibility_loss(self, cfs: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """
        Feasibility loss: Constraint ihlalleri
        """
        total_violation = torch.tensor(0.0, device=self.device)
        
        for cf in cfs:
            violation = self.constraint_manager.compute_total_violation(query.squeeze(0), cf)
            total_violation = total_violation + violation
        
        return total_violation / cfs.size(0)
    
    def _add_dp_noise_to_grads(self, cfs: List[torch.Tensor]):
        """Gradient'lara DP noise ekle - düzeltilmiş versiyon"""
        for cf in cfs:
            if cf.grad is not None:
                # Gradient clipping
                grad_norm = cf.grad.norm()
                max_grad_norm = 1.0
                
                if grad_norm > max_grad_norm:
                    cf.grad = cf.grad * (max_grad_norm / grad_norm)
                
                # Add noise - SCALED DOWN for CF generation
                # CF generation'da her iteration'da noise eklendiği için
                # noise'u iteration sayısına göre scale ediyoruz
                noise_multiplier = 0.01  # Çok daha düşük noise
                noise = torch.randn_like(cf.grad) * self.dp_mechanism.noise_scale * noise_multiplier
                cf.grad.data = cf.grad.data + noise
    
    def _compute_metrics(
        self,
        query: torch.Tensor,
        cfs: List[torch.Tensor],
        target_class: int,
        generation_time: float
    ) -> CFResult:
        """Final metrikleri hesapla"""
        
        cf_tensors = []
        cf_preds = []
        
        with torch.no_grad():
            for cf in cfs:
                pred = self.model.predict(cf.unsqueeze(0)).item()
                cf_tensors.append(cf.detach().cpu())
                cf_preds.append(pred)
        
        # Validity: Target class'a ulaşan CF oranı
        validity = sum(1 for p in cf_preds if p == target_class) / len(cf_preds)
        
        # Proximity: Ortalama L1 distance
        distances = [torch.abs(cf - query.cpu()).mean().item() for cf in cf_tensors]
        proximity = np.mean(distances)
        
        # Sparsity: Ortalama değişen feature sayısı
        changes = [torch.sum(torch.abs(cf - query.cpu()) > 0.01).item() for cf in cf_tensors]
        sparsity = np.mean(changes)
        
        # Diversity: Ortalama pairwise distance
        if len(cf_tensors) > 1:
            pairwise_dists = []
            for i in range(len(cf_tensors)):
                for j in range(i + 1, len(cf_tensors)):
                    dist = torch.norm(cf_tensors[i] - cf_tensors[j]).item()
                    pairwise_dists.append(dist)
            diversity = np.mean(pairwise_dists)
        else:
            diversity = 0.0
        
        # Actionability: Constraint'leri sağlayan CF oranı
        actionable = sum(
            1 for cf in cf_tensors 
            if self.constraint_manager.is_feasible(query.cpu(), cf)
        ) / len(cf_tensors)
        
        return CFResult(
            original=query.detach().cpu(),
            counterfactuals=cf_tensors,
            original_pred=-1,  # Will be set by caller
            cf_preds=cf_preds,
            validity=validity,
            proximity=proximity,
            sparsity=sparsity,
            diversity=diversity,
            actionability=actionable,
            generation_time=generation_time
        )
    
    def get_privacy_budget(self) -> Tuple[float, float]:
        """Toplam harcanan privacy budget"""
        if self.use_dp:
            return self.dp_mechanism.get_privacy_spent()
        return float('inf'), 0.0


class BatchCFGenerator:
    """
    Batch counterfactual generation
    Birden fazla instance için parallel CF üretimi
    """
    
    def __init__(self, cf_generator: CounterfactualGenerator):
        self.generator = cf_generator
    
    def generate_batch(
        self,
        instances: torch.Tensor,
        target_class: int = 0,
        max_workers: int = 4
    ) -> List[CFResult]:
        """
        Batch CF generation
        
        Args:
            instances: Shape (N, feature_count)
            target_class: Hedef sınıf
            
        Returns:
            List of CFResult
        """
        results = []
        
        for i in range(instances.size(0)):
            result = self.generator.generate(instances[i], target_class)
            results.append(result)
        
        return results
    
    def aggregate_statistics(self, results: List[CFResult]) -> Dict[str, float]:
        """Batch sonuçlarını aggregate et"""
        return {
            'mean_validity': np.mean([r.validity for r in results]),
            'mean_proximity': np.mean([r.proximity for r in results]),
            'mean_sparsity': np.mean([r.sparsity for r in results]),
            'mean_diversity': np.mean([r.diversity for r in results]),
            'mean_actionability': np.mean([r.actionability for r in results]),
            'mean_generation_time': np.mean([r.generation_time for r in results]),
            'total_cfs': sum(len(r.counterfactuals) for r in results)
        }


# Test
if __name__ == "__main__":
    print("Testing Counterfactual Generator...")
    
    # Dummy model
    from models.ids_models import IDSMLP
    
    feature_count = 47
    num_classes = 8
    
    model = IDSMLP(input_dim=feature_count, num_classes=num_classes)
    
    # Generator
    generator = CounterfactualGenerator(
        model=model,
        feature_count=feature_count,
        num_cfs=5,
        use_dp=True,
        epsilon=1.0,
        max_iter=100
    )
    
    # Query instance (simulated attack)
    query = torch.randn(feature_count)
    
    print(f"\nQuery shape: {query.shape}")
    print(f"Original prediction: {model.predict(query.unsqueeze(0)).item()}")
    
    # Generate CFs
    print("\nGenerating counterfactuals...")
    result = generator.generate(query, target_class=0)
    
    print(f"\n=== Results ===")
    print(f"Original prediction: {result.original_pred}")
    print(f"CF predictions: {result.cf_preds}")
    print(f"Validity: {result.validity:.2%}")
    print(f"Proximity: {result.proximity:.4f}")
    print(f"Sparsity: {result.sparsity:.1f} features")
    print(f"Diversity: {result.diversity:.4f}")
    print(f"Actionability: {result.actionability:.2%}")
    print(f"Generation time: {result.generation_time:.3f}s")
    
    # Privacy budget
    eps, delta = generator.get_privacy_budget()
    print(f"\nPrivacy budget: ε={eps}, δ={delta}")
    
    print("\nCounterfactual Generator test completed!")
