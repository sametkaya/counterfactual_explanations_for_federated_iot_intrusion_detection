"""
Learnable IoT Constraints
==========================
Veri-driven constraint learning for IoT-IDS

Sabit constraint'ler yerine veriden öğrenilen constraint'ler:
1. Feature correlation constraints
2. Learned valid ranges
3. Causal relationship discovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class LearnedConstraint:
    """Öğrenilmiş constraint"""
    name: str
    constraint_type: str
    parameters: Dict
    confidence: float
    violation_penalty: float = 1.0


class ConstraintLearner(nn.Module):
    """
    Veriden constraint öğrenen neural network
    
    Autoencoder tabanlı: Valid data manifold'unu öğren
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32, 16],
        latent_dim: int = 8
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_recon, _ = self.forward(x)
        return F.mse_loss(x_recon, x, reduction='none').sum(dim=1)


class FeatureCorrelationLearner:
    """
    Feature korelasyonlarını öğren ve constraint olarak kullan
    
    Örnek: packet_count ve byte_count arasında güçlü korelasyon
    → byte_count ≈ α * packet_count + β
    """
    
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names
        self.correlations: Dict[Tuple[int, int], Dict] = {}
        self.learned_relationships: List[LearnedConstraint] = []
    
    def fit(
        self,
        X: torch.Tensor,
        correlation_threshold: float = 0.8
    ):
        """Önemli korelasyonları öğren"""
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        
        # Correlation matrix
        corr_matrix = np.corrcoef(X_np.T)
        n_features = corr_matrix.shape[0]
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = corr_matrix[i, j]
                
                if abs(corr) >= correlation_threshold:
                    # Fit linear relationship
                    x_i, x_j = X_np[:, i], X_np[:, j]
                    
                    # Simple linear regression
                    slope = np.cov(x_i, x_j)[0, 1] / (np.var(x_i) + 1e-8)
                    intercept = np.mean(x_j) - slope * np.mean(x_i)
                    
                    # Residual std for tolerance
                    predicted = slope * x_i + intercept
                    residual_std = np.std(x_j - predicted)
                    
                    self.correlations[(i, j)] = {
                        'correlation': corr,
                        'slope': slope,
                        'intercept': intercept,
                        'tolerance': 2 * residual_std  # 2-sigma tolerance
                    }
                    
                    # Create constraint
                    name = f"corr_{i}_{j}"
                    if self.feature_names:
                        name = f"corr_{self.feature_names[i]}_{self.feature_names[j]}"
                    
                    self.learned_relationships.append(LearnedConstraint(
                        name=name,
                        constraint_type='linear_relationship',
                        parameters={
                            'feature_i': i,
                            'feature_j': j,
                            'slope': float(slope),
                            'intercept': float(intercept),
                            'tolerance': float(residual_std * 2)
                        },
                        confidence=abs(corr)
                    ))
        
        print(f"Learned {len(self.learned_relationships)} correlation constraints")
        return self
    
    def compute_violation(self, x: torch.Tensor) -> torch.Tensor:
        """Tüm öğrenilmiş korelasyon constraint'lerinin violation'ını hesapla"""
        total_violation = torch.tensor(0.0)
        
        for constraint in self.learned_relationships:
            params = constraint.parameters
            i, j = params['feature_i'], params['feature_j']
            
            expected_j = params['slope'] * x[i] + params['intercept']
            actual_j = x[j]
            
            deviation = torch.abs(actual_j - expected_j)
            violation = torch.relu(deviation - params['tolerance'])
            
            total_violation = total_violation + constraint.confidence * violation
        
        return total_violation


class RangeLearner:
    """
    Valid feature range'lerini veriden öğren
    
    Outlier-robust: Percentile-based ranges
    """
    
    def __init__(
        self,
        percentile_low: float = 1.0,
        percentile_high: float = 99.0
    ):
        self.percentile_low = percentile_low
        self.percentile_high = percentile_high
        self.ranges: Dict[int, Tuple[float, float]] = {}
        self.learned_constraints: List[LearnedConstraint] = []
    
    def fit(self, X: torch.Tensor):
        """Her feature için valid range öğren"""
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        n_features = X_np.shape[1]
        
        for i in range(n_features):
            low = np.percentile(X_np[:, i], self.percentile_low)
            high = np.percentile(X_np[:, i], self.percentile_high)
            
            # Add some margin
            margin = (high - low) * 0.1
            self.ranges[i] = (low - margin, high + margin)
            
            self.learned_constraints.append(LearnedConstraint(
                name=f"range_feature_{i}",
                constraint_type='range',
                parameters={
                    'feature_idx': i,
                    'min_val': float(low - margin),
                    'max_val': float(high + margin)
                },
                confidence=1.0
            ))
        
        print(f"Learned ranges for {n_features} features")
        return self
    
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply learned ranges"""
        x_clamped = x.clone()
        for i, (low, high) in self.ranges.items():
            x_clamped[i] = torch.clamp(x_clamped[i], low, high)
        return x_clamped
    
    def compute_violation(self, x: torch.Tensor) -> torch.Tensor:
        """Range violation hesapla"""
        total_violation = torch.tensor(0.0)
        
        for i, (low, high) in self.ranges.items():
            below = torch.relu(low - x[i])
            above = torch.relu(x[i] - high)
            total_violation = total_violation + below + above
        
        return total_violation


class CausalConstraintLearner:
    """
    Causal relationship discovery for IoT features
    
    Basit yaklaşım: Conditional independence tests
    Gelişmiş: PC algorithm veya causal forests kullanılabilir
    """
    
    def __init__(self):
        self.causal_edges: List[Tuple[int, int]] = []
        self.causal_functions: Dict[int, callable] = {}
    
    def fit(
        self,
        X: torch.Tensor,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.7
    ):
        """
        Basit causal discovery: Partial correlations ile
        """
        X_np = X.numpy() if isinstance(X, torch.Tensor) else X
        n_features = X_np.shape[1]
        
        # Compute partial correlations (simplified)
        # Full implementation would use PC algorithm
        corr_matrix = np.corrcoef(X_np.T)
        
        # Find strong directional relationships
        # (Simplified: just use correlation strength)
        for i in range(n_features):
            for j in range(n_features):
                if i != j and abs(corr_matrix[i, j]) > threshold:
                    # Assume i -> j if var(i) > var(j) (heuristic)
                    if np.var(X_np[:, i]) > np.var(X_np[:, j]):
                        self.causal_edges.append((i, j))
        
        print(f"Discovered {len(self.causal_edges)} potential causal edges")
        return self


class LearnableConstraintManager:
    """
    Tüm learnable constraint'leri yöneten manager
    """
    
    def __init__(
        self,
        feature_count: int,
        feature_names: Optional[List[str]] = None
    ):
        self.feature_count = feature_count
        self.feature_names = feature_names
        
        # Learners
        self.manifold_learner: Optional[ConstraintLearner] = None
        self.correlation_learner = FeatureCorrelationLearner(feature_names)
        self.range_learner = RangeLearner()
        self.causal_learner = CausalConstraintLearner()
        
        # Combined constraints
        self.learned_constraints: List[LearnedConstraint] = []
        
        # Weights for different constraint types
        self.weights = {
            'manifold': 1.0,
            'correlation': 0.5,
            'range': 0.3,
            'causal': 0.2
        }
    
    def fit(
        self,
        X_train: torch.Tensor,
        y_train: Optional[torch.Tensor] = None,
        train_manifold: bool = True
    ):
        """Tüm constraint'leri veriden öğren"""
        print("Learning constraints from data...")
        
        # 1. Manifold (autoencoder)
        if train_manifold:
            print("  Training manifold learner...")
            self.manifold_learner = ConstraintLearner(self.feature_count)
            self._train_manifold(X_train)
        
        # 2. Correlations
        print("  Learning correlations...")
        self.correlation_learner.fit(X_train)
        
        # 3. Ranges
        print("  Learning ranges...")
        self.range_learner.fit(X_train)
        
        # 4. Causal relationships
        print("  Discovering causal relationships...")
        self.causal_learner.fit(X_train, self.feature_names)
        
        # Combine all constraints
        self.learned_constraints = (
            self.correlation_learner.learned_relationships +
            self.range_learner.learned_constraints
        )
        
        print(f"Total learned constraints: {len(self.learned_constraints)}")
        return self
    
    def _train_manifold(
        self,
        X_train: torch.Tensor,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 1e-3
    ):
        """Train autoencoder for manifold learning"""
        self.manifold_learner.train()
        optimizer = torch.optim.Adam(self.manifold_learner.parameters(), lr=lr)
        
        dataset = torch.utils.data.TensorDataset(X_train)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for (batch_x,) in dataloader:
                x_recon, z = self.manifold_learner(batch_x)
                loss = F.mse_loss(x_recon, batch_x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"    Epoch {epoch+1}/{epochs}: Loss={total_loss/len(dataloader):.4f}")
        
        self.manifold_learner.eval()
    
    def compute_violation(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Toplam constraint violation hesapla"""
        total_violation = torch.tensor(0.0)
        
        # Manifold violation (reconstruction error)
        if self.manifold_learner is not None:
            recon_loss = self.manifold_learner.reconstruction_loss(cf.unsqueeze(0))
            total_violation = total_violation + self.weights['manifold'] * recon_loss.squeeze()
        
        # Correlation violations
        corr_violation = self.correlation_learner.compute_violation(cf)
        total_violation = total_violation + self.weights['correlation'] * corr_violation
        
        # Range violations
        range_violation = self.range_learner.compute_violation(cf)
        total_violation = total_violation + self.weights['range'] * range_violation
        
        return total_violation
    
    def apply_constraints(self, original: torch.Tensor, cf: torch.Tensor) -> torch.Tensor:
        """Soft constraint application - gradient'ları yönlendir"""
        # Range constraints (hard)
        cf_constrained = self.range_learner.apply(cf)
        return cf_constrained
    
    def is_feasible(self, original: torch.Tensor, cf: torch.Tensor, threshold: float = 0.1) -> bool:
        """Check if CF satisfies learned constraints"""
        violation = self.compute_violation(original, cf)
        return violation.item() < threshold
    
    def get_constraint_summary(self) -> Dict:
        """Get summary of learned constraints"""
        return {
            'num_correlation_constraints': len(self.correlation_learner.learned_relationships),
            'num_range_constraints': len(self.range_learner.learned_constraints),
            'num_causal_edges': len(self.causal_learner.causal_edges),
            'has_manifold_learner': self.manifold_learner is not None,
            'constraint_weights': self.weights
        }
