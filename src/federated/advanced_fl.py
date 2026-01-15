"""
Advanced Federated Learning Aggregation Algorithms
===================================================

FedAvg'in ötesinde: FedProx, FedNova, Scaffold, FedOpt

Referanslar:
- FedProx: Li et al. (2020) "Federated Optimization in Heterogeneous Networks"
- FedNova: Wang et al. (2020) "Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization"
- Scaffold: Karimireddy et al. (2020) "SCAFFOLD: Stochastic Controlled Averaging"
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import copy
import numpy as np


@dataclass
class ClientUpdate:
    """Client update with metadata"""
    client_id: int
    weights: Dict[str, torch.Tensor]
    num_samples: int
    local_steps: int  # Number of local SGD steps
    metrics: Dict[str, float]
    # For FedNova
    tau: float = 1.0  # Normalized gradient scaling factor
    # For Scaffold
    control_variate: Optional[Dict[str, torch.Tensor]] = None


class FedAvgAggregator:
    """
    Standard Federated Averaging
    
    w_global = Σ (n_k / n) * w_k
    """
    
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model
    
    def aggregate(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """Aggregate client updates using weighted average"""
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = {}
        for key in updates[0].weights.keys():
            weighted_sum = torch.zeros_like(updates[0].weights[key], dtype=torch.float32)
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.weights[key].float()
            aggregated[key] = weighted_sum
        
        return aggregated


class FedProxAggregator:
    """
    FedProx: Federated Optimization in Heterogeneous Networks
    
    Proximal term eklenerek client drift azaltılır:
    min_w F_k(w) + (μ/2) ||w - w_global||²
    
    Aggregation FedAvg ile aynı, fark local training'de.
    Bu sınıf aggregation + proximal hesaplama sağlar.
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        mu: float = 0.01  # Proximal term coefficient
    ):
        self.global_model = global_model
        self.mu = mu
        self._global_weights = None
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global weights (for proximal term calculation)"""
        self._global_weights = {
            name: param.clone().detach()
            for name, param in self.global_model.state_dict().items()
        }
        return self._global_weights
    
    def compute_proximal_term(
        self,
        local_weights: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute proximal term: (μ/2) ||w - w_global||²
        
        Bu değer local loss'a eklenir.
        """
        if self._global_weights is None:
            return torch.tensor(0.0)
        
        prox_term = torch.tensor(0.0)
        for key in local_weights.keys():
            if key in self._global_weights:
                diff = local_weights[key] - self._global_weights[key]
                prox_term += torch.sum(diff ** 2)
        
        return (self.mu / 2) * prox_term
    
    def aggregate(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """Aggregate (same as FedAvg)"""
        total_samples = sum(u.num_samples for u in updates)
        
        aggregated = {}
        for key in updates[0].weights.keys():
            weighted_sum = torch.zeros_like(updates[0].weights[key], dtype=torch.float32)
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.weights[key].float()
            aggregated[key] = weighted_sum
        
        return aggregated


class FedNovaAggregator:
    """
    FedNova: Tackling the Objective Inconsistency Problem
    
    Problem: FedAvg'de farklı local step sayıları objective inconsistency yaratır.
    Çözüm: Gradient'ları normalize et.
    
    Normalized aggregation:
    d_k = (w_k - w_global) / τ_k  (normalized pseudo-gradient)
    w_new = w_global + τ_eff * Σ (p_k * d_k)
    
    where τ_eff = Σ (p_k * τ_k)
    """
    
    def __init__(self, global_model: nn.Module):
        self.global_model = global_model
        self._global_weights = None
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global weights"""
        self._global_weights = {
            name: param.clone().detach()
            for name, param in self.global_model.state_dict().items()
        }
        return self._global_weights
    
    def aggregate(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """
        FedNova normalized aggregation
        """
        if self._global_weights is None:
            self._global_weights = self.get_global_weights()
        
        total_samples = sum(u.num_samples for u in updates)
        
        # Compute tau_eff (effective number of local steps)
        tau_eff = sum(
            (u.num_samples / total_samples) * u.local_steps
            for u in updates
        )
        
        # Compute normalized pseudo-gradients and aggregate
        aggregated = {}
        for key in updates[0].weights.keys():
            # Weighted sum of normalized gradients
            grad_sum = torch.zeros_like(updates[0].weights[key], dtype=torch.float32)
            
            for update in updates:
                weight = update.num_samples / total_samples
                tau_k = update.local_steps
                
                # Pseudo-gradient: (w_k - w_global)
                pseudo_grad = update.weights[key].float() - self._global_weights[key].float()
                
                # Normalized pseudo-gradient: d_k = pseudo_grad / τ_k
                normalized_grad = pseudo_grad / tau_k
                
                grad_sum += weight * normalized_grad
            
            # New weights: w_global + τ_eff * aggregated_grad
            aggregated[key] = self._global_weights[key].float() + tau_eff * grad_sum
        
        return aggregated


class ScaffoldAggregator:
    """
    SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
    
    Control variates kullanarak variance azaltma.
    Client drift'i kontrol etmek için server ve client control variates.
    
    Client update:
    c_i^+ = c_i - c + (1/(K*η)) * (w_global - w_i)
    
    Server update:
    c = c + (1/N) * Σ (c_i^+ - c_i)
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        num_clients: int,
        learning_rate: float = 0.01
    ):
        self.global_model = global_model
        self.num_clients = num_clients
        self.learning_rate = learning_rate
        
        # Server control variate
        self.server_control = None
        
        # Client control variates
        self.client_controls: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def initialize_controls(self):
        """Initialize control variates to zero"""
        self.server_control = {
            name: torch.zeros_like(param)
            for name, param in self.global_model.state_dict().items()
        }
        
        for i in range(self.num_clients):
            self.client_controls[i] = {
                name: torch.zeros_like(param)
                for name, param in self.global_model.state_dict().items()
            }
    
    def get_client_control(self, client_id: int) -> Dict[str, torch.Tensor]:
        """Get control variate for a client"""
        if client_id not in self.client_controls:
            self.client_controls[client_id] = {
                name: torch.zeros_like(param)
                for name, param in self.global_model.state_dict().items()
            }
        return self.client_controls[client_id]
    
    def get_server_control(self) -> Dict[str, torch.Tensor]:
        """Get server control variate"""
        if self.server_control is None:
            self.initialize_controls()
        return self.server_control
    
    def aggregate(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """
        SCAFFOLD aggregation with control variate updates
        """
        if self.server_control is None:
            self.initialize_controls()
        
        total_samples = sum(u.num_samples for u in updates)
        num_participating = len(updates)
        
        # Aggregate model weights (like FedAvg)
        aggregated = {}
        for key in updates[0].weights.keys():
            weighted_sum = torch.zeros_like(updates[0].weights[key], dtype=torch.float32)
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.weights[key].float()
            aggregated[key] = weighted_sum
        
        # Update server control variate
        if updates[0].control_variate is not None:
            for key in self.server_control.keys():
                control_delta = torch.zeros_like(self.server_control[key])
                for update in updates:
                    if update.control_variate is not None:
                        old_control = self.client_controls.get(update.client_id, {}).get(key, torch.zeros_like(control_delta))
                        new_control = update.control_variate.get(key, old_control)
                        control_delta += (new_control - old_control)
                
                # Update server control
                self.server_control[key] += control_delta / self.num_clients
                
                # Update client controls
                for update in updates:
                    if update.control_variate is not None:
                        self.client_controls[update.client_id][key] = update.control_variate.get(
                            key, self.client_controls[update.client_id].get(key, torch.zeros_like(control_delta))
                        )
        
        return aggregated


class FedOptAggregator:
    """
    FedOpt: Adaptive Federated Optimization
    
    Server-side optimizer (Adam, SGD with momentum, etc.) kullanarak
    aggregated update'i uygula.
    
    Varyantlar:
    - FedAdam: Server-side Adam
    - FedYogi: Server-side Yogi
    - FedAdagrad: Server-side Adagrad
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        server_optimizer: str = "adam",  # "adam", "sgd", "yogi", "adagrad"
        server_lr: float = 1.0,
        beta1: float = 0.9,
        beta2: float = 0.99,
        tau: float = 1e-3  # For Yogi
    ):
        self.global_model = global_model
        self.server_optimizer = server_optimizer
        self.server_lr = server_lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        
        # Optimizer states
        self.m = None  # First moment
        self.v = None  # Second moment
        self.t = 0     # Timestep
        
        self._global_weights = None
    
    def _initialize_optimizer_states(self, keys: List[str], shapes: Dict[str, torch.Size]):
        """Initialize optimizer states"""
        self.m = {key: torch.zeros(shapes[key]) for key in keys}
        self.v = {key: torch.zeros(shapes[key]) for key in keys}
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global weights"""
        self._global_weights = {
            name: param.clone().detach()
            for name, param in self.global_model.state_dict().items()
        }
        return self._global_weights
    
    def aggregate(self, updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """
        FedOpt aggregation with server-side optimization
        """
        if self._global_weights is None:
            self._global_weights = self.get_global_weights()
        
        total_samples = sum(u.num_samples for u in updates)
        
        # Compute pseudo-gradient (FedAvg style)
        pseudo_grad = {}
        for key in updates[0].weights.keys():
            weighted_sum = torch.zeros_like(updates[0].weights[key], dtype=torch.float32)
            for update in updates:
                weight = update.num_samples / total_samples
                weighted_sum += weight * update.weights[key].float()
            
            # Pseudo-gradient = new_weights - old_weights
            pseudo_grad[key] = weighted_sum - self._global_weights[key].float()
        
        # Initialize optimizer states if needed
        if self.m is None:
            shapes = {key: grad.shape for key, grad in pseudo_grad.items()}
            self._initialize_optimizer_states(list(pseudo_grad.keys()), shapes)
        
        self.t += 1
        
        # Apply server optimizer
        aggregated = {}
        for key in pseudo_grad.keys():
            g = pseudo_grad[key]
            
            if self.server_optimizer == "sgd":
                # SGD with momentum
                self.m[key] = self.beta1 * self.m[key] + g
                update = self.server_lr * self.m[key]
            
            elif self.server_optimizer == "adam":
                # Adam
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g ** 2)
                
                # Bias correction
                m_hat = self.m[key] / (1 - self.beta1 ** self.t)
                v_hat = self.v[key] / (1 - self.beta2 ** self.t)
                
                update = self.server_lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
            
            elif self.server_optimizer == "yogi":
                # Yogi (more robust than Adam)
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
                
                # Yogi update for v
                v_sign = torch.sign(g ** 2 - self.v[key])
                self.v[key] = self.v[key] + (1 - self.beta2) * v_sign * (g ** 2)
                
                update = self.server_lr * self.m[key] / (torch.sqrt(self.v[key]) + self.tau)
            
            elif self.server_optimizer == "adagrad":
                # Adagrad
                self.v[key] = self.v[key] + g ** 2
                update = self.server_lr * g / (torch.sqrt(self.v[key]) + 1e-8)
            
            else:
                raise ValueError(f"Unknown optimizer: {self.server_optimizer}")
            
            aggregated[key] = self._global_weights[key].float() + update
        
        return aggregated


class AdvancedFederatedServer:
    """
    Advanced Federated Server supporting multiple aggregation algorithms
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        aggregation: str = "fedavg",  # "fedavg", "fedprox", "fednova", "scaffold", "fedopt"
        device: str = "cpu",
        **kwargs
    ):
        self.global_model = copy.deepcopy(global_model).to(device)
        self.device = device
        self.aggregation = aggregation
        
        # Initialize aggregator
        if aggregation == "fedavg":
            self.aggregator = FedAvgAggregator(self.global_model)
        elif aggregation == "fedprox":
            mu = kwargs.get('mu', 0.01)
            self.aggregator = FedProxAggregator(self.global_model, mu=mu)
        elif aggregation == "fednova":
            self.aggregator = FedNovaAggregator(self.global_model)
        elif aggregation == "scaffold":
            num_clients = kwargs.get('num_clients', 10)
            lr = kwargs.get('learning_rate', 0.01)
            self.aggregator = ScaffoldAggregator(self.global_model, num_clients, lr)
        elif aggregation == "fedopt":
            server_opt = kwargs.get('server_optimizer', 'adam')
            server_lr = kwargs.get('server_lr', 1.0)
            self.aggregator = FedOptAggregator(self.global_model, server_opt, server_lr)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        self.round_num = 0
        self.history = []
    
    def get_global_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global weights"""
        if hasattr(self.aggregator, 'get_global_weights'):
            return self.aggregator.get_global_weights()
        return {
            name: param.clone().detach()
            for name, param in self.global_model.state_dict().items()
        }
    
    def aggregate_and_update(self, updates: List[ClientUpdate]) -> Dict[str, float]:
        """
        Aggregate client updates and update global model
        
        Returns:
            metrics: Round metrics
        """
        # Aggregate
        new_weights = self.aggregator.aggregate(updates)
        
        # Update global model
        self.global_model.load_state_dict(new_weights)
        
        # Record metrics
        metrics = {
            'round': self.round_num,
            'num_clients': len(updates),
            'total_samples': sum(u.num_samples for u in updates),
            'aggregation': self.aggregation
        }
        
        # Average client metrics
        for key in updates[0].metrics.keys():
            metrics[f'avg_{key}'] = np.mean([u.metrics.get(key, 0) for u in updates])
        
        self.history.append(metrics)
        self.round_num += 1
        
        return metrics
    
    def get_aggregator_state(self) -> Dict:
        """Get aggregator-specific state (for Scaffold, FedOpt)"""
        state = {'aggregation': self.aggregation}
        
        if isinstance(self.aggregator, ScaffoldAggregator):
            state['server_control'] = self.aggregator.server_control
            state['client_controls'] = self.aggregator.client_controls
        elif isinstance(self.aggregator, FedOptAggregator):
            state['m'] = self.aggregator.m
            state['v'] = self.aggregator.v
            state['t'] = self.aggregator.t
        elif isinstance(self.aggregator, FedProxAggregator):
            state['mu'] = self.aggregator.mu
        
        return state
