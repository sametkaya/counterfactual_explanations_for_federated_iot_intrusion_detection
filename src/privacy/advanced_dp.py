"""
Advanced Privacy Mechanisms
============================
Rényi Differential Privacy ve Privacy Accounting

Referanslar:
- Mironov (2017): "Rényi Differential Privacy"
- Abadi et al. (2016): "Deep Learning with Differential Privacy"
- Google DP Library konseptleri
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import math


@dataclass
class PrivacySpent:
    """Harcanan privacy budget"""
    epsilon: float
    delta: float
    rdp_alphas: Optional[Dict[float, float]] = None  # alpha -> RDP epsilon mapping
    num_compositions: int = 0


class RenyiDifferentialPrivacy:
    """
    Rényi Differential Privacy (RDP) Mechanism
    
    RDP, composition için daha tight bound sağlar.
    ε-DP'ye göre avantajı: Çoklu composition'da daha az privacy kaybı.
    
    RDP Definition:
    D_α(M(D) || M(D')) ≤ ε
    where D_α is the Rényi divergence of order α
    """
    
    def __init__(
        self,
        noise_multiplier: float = 1.0,
        delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        alphas: Optional[List[float]] = None
    ):
        """
        Args:
            noise_multiplier: σ/sensitivity ratio (higher = more privacy)
            delta: Target delta for (ε,δ)-DP conversion
            max_grad_norm: Clipping bound C
            alphas: RDP orders to track (default: [1.5, 2, 5, 10, 25, 50, 100])
        """
        self.noise_multiplier = noise_multiplier
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.alphas = alphas or [1.5, 2, 5, 10, 25, 50, 100]
        
        # Privacy accounting
        self.rdp_epsilon = {alpha: 0.0 for alpha in self.alphas}
        self.num_compositions = 0
    
    def compute_rdp_single_step(self, sampling_rate: float = 1.0) -> Dict[float, float]:
        """
        Compute RDP epsilon for a single step
        
        For Gaussian mechanism with sampling:
        ε_α ≤ α / (2σ²) for α > 1
        
        With subsampling amplification:
        ε_α ≈ log(1 + q²(exp(α/σ²) - 1)) / (α-1)
        """
        rdp_eps = {}
        
        for alpha in self.alphas:
            if sampling_rate >= 1.0:
                # No subsampling
                eps = alpha / (2 * self.noise_multiplier ** 2)
            else:
                # Subsampled Gaussian mechanism (privacy amplification)
                # Using tight bound from "Rényi Differential Privacy of the Sampled Gaussian Mechanism"
                q = sampling_rate
                sigma = self.noise_multiplier
                
                # Approximate bound
                if alpha <= 1:
                    eps = 0
                else:
                    inner = q ** 2 * (math.exp(alpha / (sigma ** 2)) - 1)
                    eps = math.log1p(inner) / (alpha - 1)
            
            rdp_eps[alpha] = eps
        
        return rdp_eps
    
    def accumulate_privacy(self, sampling_rate: float = 1.0):
        """
        Accumulate privacy spent for one iteration
        
        RDP composes linearly: ε_total = Σ ε_i
        """
        step_rdp = self.compute_rdp_single_step(sampling_rate)
        
        for alpha in self.alphas:
            self.rdp_epsilon[alpha] += step_rdp[alpha]
        
        self.num_compositions += 1
    
    def get_privacy_spent(self) -> PrivacySpent:
        """
        Convert accumulated RDP to (ε, δ)-DP
        
        Conversion: ε = min_α (ε_α + log(1/δ)/(α-1))
        """
        min_epsilon = float('inf')
        
        for alpha in self.alphas:
            if alpha > 1:
                eps = self.rdp_epsilon[alpha] + math.log(1 / self.delta) / (alpha - 1)
                min_epsilon = min(min_epsilon, eps)
        
        return PrivacySpent(
            epsilon=min_epsilon,
            delta=self.delta,
            rdp_alphas=self.rdp_epsilon.copy(),
            num_compositions=self.num_compositions
        )
    
    def reset_accounting(self):
        """Reset privacy accountant"""
        self.rdp_epsilon = {alpha: 0.0 for alpha in self.alphas}
        self.num_compositions = 0


class AdvancedDPMechanism:
    """
    Advanced DP Mechanism with multiple noise types and adaptive clipping
    """
    
    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        mechanism: str = "gaussian",
        adaptive_clipping: bool = True,
        noise_decay: float = 1.0
    ):
        """
        Args:
            target_epsilon: Target privacy budget
            target_delta: Target delta
            max_grad_norm: Initial clipping bound
            mechanism: "gaussian", "laplace", or "analytic_gaussian"
            adaptive_clipping: Whether to use adaptive clipping
            noise_decay: Decay factor for noise over iterations (1.0 = no decay)
        """
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.max_grad_norm = max_grad_norm
        self.mechanism = mechanism
        self.adaptive_clipping = adaptive_clipping
        self.noise_decay = noise_decay
        
        # Compute noise multiplier from target epsilon
        self.noise_multiplier = self._compute_noise_multiplier()
        
        # RDP accountant
        self.rdp_accountant = RenyiDifferentialPrivacy(
            noise_multiplier=self.noise_multiplier,
            delta=target_delta,
            max_grad_norm=max_grad_norm
        )
        
        # Adaptive clipping state
        self.grad_norm_history = []
        self.current_clip_bound = max_grad_norm
        self.iteration = 0
    
    def _compute_noise_multiplier(self) -> float:
        """
        Compute noise multiplier from target epsilon
        
        For single query: σ ≥ √(2 ln(1.25/δ)) / ε
        """
        if self.mechanism == "gaussian" or self.mechanism == "analytic_gaussian":
            return math.sqrt(2 * math.log(1.25 / self.target_delta)) / self.target_epsilon
        else:  # Laplace
            return 1.0 / self.target_epsilon
    
    def clip_gradients(self, gradients: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Clip gradients to bound sensitivity
        
        Returns:
            clipped_gradients, clip_factor
        """
        grad_norm = gradients.norm(2)
        
        # Track for adaptive clipping
        self.grad_norm_history.append(grad_norm.item())
        
        # Compute clip bound
        if self.adaptive_clipping and len(self.grad_norm_history) > 10:
            # Use median of recent norms as clip bound
            recent_norms = self.grad_norm_history[-100:]
            adaptive_bound = np.percentile(recent_norms, 75)
            self.current_clip_bound = min(self.max_grad_norm, adaptive_bound * 1.5)
        
        # Clip
        clip_factor = min(1.0, self.current_clip_bound / (grad_norm + 1e-8))
        clipped = gradients * clip_factor
        
        return clipped, clip_factor.item() if isinstance(clip_factor, torch.Tensor) else clip_factor
    
    def add_noise(self, tensor: torch.Tensor, sampling_rate: float = 1.0) -> torch.Tensor:
        """
        Add calibrated DP noise to tensor
        """
        # Decay noise over iterations (optional warmup)
        effective_noise = self.noise_multiplier * (self.noise_decay ** self.iteration)
        
        if self.mechanism == "gaussian":
            noise = torch.randn_like(tensor) * self.current_clip_bound * effective_noise
        elif self.mechanism == "laplace":
            scale = self.current_clip_bound * effective_noise
            noise = torch.zeros_like(tensor)
            noise.exponential_(1.0 / scale)
            noise *= (2 * torch.bernoulli(torch.ones_like(tensor) * 0.5) - 1)
        elif self.mechanism == "analytic_gaussian":
            # Analytic Gaussian mechanism (tighter for small epsilon)
            noise = torch.randn_like(tensor) * self.current_clip_bound * effective_noise
        else:
            raise ValueError(f"Unknown mechanism: {self.mechanism}")
        
        # Update privacy accounting
        self.rdp_accountant.accumulate_privacy(sampling_rate)
        self.iteration += 1
        
        return tensor + noise
    
    def privatize_gradients(
        self,
        gradients: torch.Tensor,
        sampling_rate: float = 1.0
    ) -> torch.Tensor:
        """
        Full DP gradient pipeline: clip + noise
        """
        clipped, _ = self.clip_gradients(gradients)
        noisy = self.add_noise(clipped, sampling_rate)
        return noisy
    
    def get_privacy_spent(self) -> PrivacySpent:
        """Get current privacy expenditure"""
        return self.rdp_accountant.get_privacy_spent()
    
    def reset(self):
        """Reset mechanism state"""
        self.rdp_accountant.reset_accounting()
        self.grad_norm_history = []
        self.current_clip_bound = self.max_grad_norm
        self.iteration = 0


class PrivacyAccountant:
    """
    Global Privacy Accountant for tracking privacy across multiple operations
    
    Supports:
    - Multiple mechanisms
    - Composition theorems
    - Budget management
    """
    
    def __init__(
        self,
        total_epsilon: float = 10.0,
        total_delta: float = 1e-5,
        composition: str = "rdp"  # "basic", "advanced", "rdp"
    ):
        """
        Args:
            total_epsilon: Total privacy budget
            total_delta: Total delta budget
            composition: Composition theorem to use
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.composition = composition
        
        # Track spent privacy
        self.operations: List[PrivacySpent] = []
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
    
    def add_operation(self, privacy_spent: PrivacySpent):
        """Record a privacy-spending operation"""
        self.operations.append(privacy_spent)
        
        if self.composition == "basic":
            # Basic composition: ε_total = Σ ε_i
            self.spent_epsilon += privacy_spent.epsilon
            self.spent_delta += privacy_spent.delta
        elif self.composition == "advanced":
            # Advanced composition: ε_total = √(2k ln(1/δ)) * ε + k*ε²
            k = len(self.operations)
            eps = privacy_spent.epsilon
            self.spent_epsilon = math.sqrt(2 * k * math.log(1 / self.total_delta)) * eps + k * eps * (math.exp(eps) - 1)
            self.spent_delta = k * privacy_spent.delta
        elif self.composition == "rdp":
            # RDP composition (handled by RenyiDifferentialPrivacy)
            self.spent_epsilon = privacy_spent.epsilon
            self.spent_delta = privacy_spent.delta
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        return (
            max(0, self.total_epsilon - self.spent_epsilon),
            max(0, self.total_delta - self.spent_delta)
        )
    
    def is_budget_exceeded(self) -> bool:
        """Check if privacy budget is exceeded"""
        return self.spent_epsilon > self.total_epsilon or self.spent_delta > self.total_delta
    
    def get_summary(self) -> Dict:
        """Get privacy summary"""
        remaining_eps, remaining_delta = self.get_remaining_budget()
        return {
            'total_budget': {'epsilon': self.total_epsilon, 'delta': self.total_delta},
            'spent': {'epsilon': self.spent_epsilon, 'delta': self.spent_delta},
            'remaining': {'epsilon': remaining_eps, 'delta': remaining_delta},
            'num_operations': len(self.operations),
            'budget_exceeded': self.is_budget_exceeded()
        }


def compute_dp_sgd_privacy(
    num_examples: int,
    batch_size: int,
    epochs: int,
    noise_multiplier: float,
    delta: float
) -> float:
    """
    Compute privacy spent for DP-SGD training
    
    Args:
        num_examples: Training set size
        batch_size: Batch size
        epochs: Number of epochs
        noise_multiplier: Noise multiplier σ
        delta: Target delta
    
    Returns:
        epsilon: Privacy spent
    """
    # Sampling rate
    q = batch_size / num_examples
    
    # Number of steps
    steps = int(epochs * num_examples / batch_size)
    
    # RDP accountant
    rdp = RenyiDifferentialPrivacy(
        noise_multiplier=noise_multiplier,
        delta=delta
    )
    
    # Accumulate privacy for all steps
    for _ in range(steps):
        rdp.accumulate_privacy(q)
    
    return rdp.get_privacy_spent().epsilon


def auto_tune_noise_multiplier(
    target_epsilon: float,
    delta: float,
    num_examples: int,
    batch_size: int,
    epochs: int,
    tolerance: float = 0.1
) -> float:
    """
    Auto-tune noise multiplier to achieve target epsilon
    
    Binary search to find optimal σ
    """
    low, high = 0.1, 100.0
    
    while high - low > tolerance:
        mid = (low + high) / 2
        eps = compute_dp_sgd_privacy(
            num_examples, batch_size, epochs, mid, delta
        )
        
        if eps > target_epsilon:
            low = mid  # Need more noise
        else:
            high = mid  # Can use less noise
    
    return high  # Return conservative estimate
