"""
Diffusion-based Counterfactual Generator
=========================================
DDPM tabanlı counterfactual explanation üretimi

Referans: "Diffusion Models for Counterfactual Explanations" (2023)

Avantajları:
- Daha realistic CF'ler (manifold üzerinde)
- Daha iyi diversity
- Gradient-free (bazı modeller için)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class DiffusionCFResult:
    """Diffusion CF sonucu"""
    original: torch.Tensor
    counterfactuals: List[torch.Tensor]
    original_pred: int
    cf_preds: List[int]
    validity: float
    proximity: float
    sparsity: float
    diversity: float
    realism_score: float  # Diffusion-specific: manifold üzerinde olma skoru
    generation_time: float


class DiffusionSchedule:
    """Noise schedule for diffusion process"""
    
    def __init__(
        self,
        num_timesteps: int = 100,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "linear"
    ):
        self.num_timesteps = num_timesteps
        
        if schedule_type == "linear":
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            self.betas = self._cosine_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1})
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
    
    def _cosine_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)


class TabularDenoiser(nn.Module):
    """
    Denoising network for tabular data
    Time-conditional MLP
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 512, 256],
        time_embed_dim: int = 64,
        num_classes: int = 8,
        class_conditional: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.class_conditional = class_conditional
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        
        # Class embedding (for conditional generation)
        if class_conditional:
            self.class_embed = nn.Embedding(num_classes, time_embed_dim)
        
        # Main network
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i] + time_embed_dim, hidden_dims[i+1]),
                nn.LayerNorm(hidden_dims[i+1]),
                nn.SiLU(),
                nn.Dropout(0.1)
            ))
        
        self.output_proj = nn.Linear(hidden_dims[-1] + time_embed_dim, input_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        target_class: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise for denoising
        
        Args:
            x: Noisy input [batch, input_dim]
            t: Timestep [batch] (normalized to [0, 1])
            target_class: Target class for conditional generation [batch]
        """
        # Time embedding
        t_embed = self.time_embed(t.unsqueeze(-1).float())
        
        # Add class embedding if conditional
        if self.class_conditional and target_class is not None:
            c_embed = self.class_embed(target_class)
            t_embed = t_embed + c_embed
        
        # Forward pass
        h = self.input_proj(x)
        
        for layer in self.layers:
            h = layer(torch.cat([h, t_embed], dim=-1))
        
        output = self.output_proj(torch.cat([h, t_embed], dim=-1))
        
        return output


class DiffusionCFGenerator:
    """
    Diffusion-based Counterfactual Generator
    
    İki modda çalışır:
    1. Pre-trained denoiser ile (daha iyi kalite)
    2. Classifier guidance ile (denoiser gerektirmez)
    """
    
    def __init__(
        self,
        classifier: nn.Module,
        feature_count: int,
        num_classes: int = 8,
        num_timesteps: int = 100,
        num_cfs: int = 5,
        guidance_scale: float = 2.0,
        denoiser: Optional[TabularDenoiser] = None,
        device: str = "cpu"
    ):
        self.classifier = classifier
        self.classifier.eval()
        self.feature_count = feature_count
        self.num_classes = num_classes
        self.num_cfs = num_cfs
        self.guidance_scale = guidance_scale
        self.device = device
        
        # Diffusion schedule
        self.schedule = DiffusionSchedule(num_timesteps=num_timesteps)
        self.num_timesteps = num_timesteps
        
        # Denoiser (optional)
        self.denoiser = denoiser
        if denoiser is not None:
            self.denoiser.to(device)
            self.denoiser.eval()
        
        # Move schedule tensors to device
        self._move_schedule_to_device()
    
    def _move_schedule_to_device(self):
        """Move schedule tensors to device"""
        self.schedule.betas = self.schedule.betas.to(self.device)
        self.schedule.alphas = self.schedule.alphas.to(self.device)
        self.schedule.alphas_cumprod = self.schedule.alphas_cumprod.to(self.device)
        self.schedule.sqrt_alphas_cumprod = self.schedule.sqrt_alphas_cumprod.to(self.device)
        self.schedule.sqrt_one_minus_alphas_cumprod = self.schedule.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.schedule.posterior_variance = self.schedule.posterior_variance.to(self.device)
    
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0)
        Add noise to x_0 at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t].view(-1, 1)
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t].view(-1, 1)
        
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
    
    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        target_class: int
    ) -> torch.Tensor:
        """
        Reverse diffusion step: p(x_{t-1} | x_t)
        """
        batch_size = x_t.size(0)
        t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
        
        # Predict noise
        if self.denoiser is not None:
            # Use denoiser network
            t_normalized = t_tensor.float() / self.num_timesteps
            target_tensor = torch.full((batch_size,), target_class, device=self.device, dtype=torch.long)
            predicted_noise = self.denoiser(x_t, t_normalized, target_tensor)
        else:
            # Use classifier guidance (gradient-based)
            predicted_noise = self._classifier_guided_noise(x_t, t, target_class)
        
        # Compute x_{t-1}
        alpha = self.schedule.alphas[t]
        alpha_cumprod = self.schedule.alphas_cumprod[t]
        beta = self.schedule.betas[t]
        
        # Mean of p(x_{t-1} | x_t)
        coef1 = 1.0 / torch.sqrt(alpha)
        coef2 = beta / self.schedule.sqrt_one_minus_alphas_cumprod[t]
        mean = coef1 * (x_t - coef2 * predicted_noise)
        
        # Add noise (except for t=0)
        if t > 0:
            noise = torch.randn_like(x_t)
            variance = self.schedule.posterior_variance[t]
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    def _classifier_guided_noise(
        self,
        x_t: torch.Tensor,
        t: int,
        target_class: int
    ) -> torch.Tensor:
        """
        Classifier guidance: Use classifier gradients to guide diffusion
        """
        x_t = x_t.clone().requires_grad_(True)
        
        # Get classifier prediction
        logits = self.classifier(x_t)
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_prob = log_probs[:, target_class].sum()
        
        # Compute gradient
        grad = torch.autograd.grad(target_log_prob, x_t)[0]
        
        # Guided noise prediction (simple approximation)
        # In practice, this guides towards target class
        noise_estimate = -self.guidance_scale * grad
        
        return noise_estimate
    
    @torch.no_grad()
    def generate(
        self,
        query_instance: torch.Tensor,
        target_class: int = 0
    ) -> DiffusionCFResult:
        """
        Generate counterfactuals using diffusion
        
        Process:
        1. Add noise to query (forward diffusion)
        2. Denoise with target class conditioning (reverse diffusion)
        """
        import time
        start_time = time.time()
        
        query = query_instance.to(self.device)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        # Original prediction
        original_pred = self.classifier(query).argmax(1).item()
        
        # Generate multiple CFs
        counterfactuals = []
        cf_preds = []
        
        for i in range(self.num_cfs):
            # Start from query with some noise
            noise_level = 0.3 + 0.1 * i  # Varying noise for diversity
            start_t = int(self.num_timesteps * noise_level)
            
            # Forward diffusion to start_t
            x_t = self.q_sample(
                query.expand(1, -1),
                torch.tensor([start_t], device=self.device)
            )
            
            # Reverse diffusion with target class guidance
            for t in reversed(range(start_t)):
                x_t = self.p_sample(x_t, t, target_class)
            
            cf = x_t.squeeze(0)
            pred = self.classifier(cf.unsqueeze(0)).argmax(1).item()
            
            counterfactuals.append(cf.cpu())
            cf_preds.append(pred)
        
        generation_time = time.time() - start_time
        
        # Compute metrics
        result = self._compute_metrics(
            query.squeeze(0).cpu(),
            counterfactuals,
            cf_preds,
            target_class,
            original_pred,
            generation_time
        )
        
        return result
    
    def _compute_metrics(
        self,
        query: torch.Tensor,
        cfs: List[torch.Tensor],
        cf_preds: List[int],
        target_class: int,
        original_pred: int,
        generation_time: float
    ) -> DiffusionCFResult:
        """Compute evaluation metrics"""
        
        # Validity
        validity = sum(1 for p in cf_preds if p == target_class) / len(cf_preds)
        
        # Proximity
        distances = [torch.abs(cf - query).mean().item() for cf in cfs]
        proximity = np.mean(distances)
        
        # Sparsity
        changes = [torch.sum(torch.abs(cf - query) > 0.01).item() for cf in cfs]
        sparsity = np.mean(changes)
        
        # Diversity
        if len(cfs) > 1:
            pairwise_dists = []
            for i in range(len(cfs)):
                for j in range(i + 1, len(cfs)):
                    dist = torch.norm(cfs[i] - cfs[j]).item()
                    pairwise_dists.append(dist)
            diversity = np.mean(pairwise_dists)
        else:
            diversity = 0.0
        
        # Realism score (placeholder - in practice would use density estimation)
        # Lower distance = more realistic (closer to data manifold)
        realism_score = 1.0 / (1.0 + proximity)
        
        return DiffusionCFResult(
            original=query,
            counterfactuals=cfs,
            original_pred=original_pred,
            cf_preds=cf_preds,
            validity=validity,
            proximity=proximity,
            sparsity=sparsity,
            diversity=diversity,
            realism_score=realism_score,
            generation_time=generation_time
        )
    
    def train_denoiser(
        self,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-3
    ):
        """
        Train the denoiser network on training data
        
        Args:
            train_data: Training samples [N, feature_count]
            train_labels: Labels [N]
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
        """
        if self.denoiser is None:
            self.denoiser = TabularDenoiser(
                input_dim=self.feature_count,
                num_classes=self.num_classes,
                class_conditional=True
            ).to(self.device)
        
        self.denoiser.train()
        optimizer = torch.optim.Adam(self.denoiser.parameters(), lr=lr)
        
        dataset = torch.utils.data.TensorDataset(train_data, train_labels)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Sample random timesteps
                t = torch.randint(0, self.num_timesteps, (batch_x.size(0),), device=self.device)
                t_normalized = t.float() / self.num_timesteps
                
                # Sample noise
                noise = torch.randn_like(batch_x)
                
                # Get noisy samples
                x_t = self.q_sample(batch_x, t, noise)
                
                # Predict noise
                predicted_noise = self.denoiser(x_t, t_normalized, batch_y)
                
                # MSE loss
                loss = F.mse_loss(predicted_noise, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0:
                print(f"  Denoiser Epoch {epoch+1}/{epochs}: Loss={total_loss/len(dataloader):.4f}")
        
        self.denoiser.eval()


class GrowingSpheresGenerator:
    """
    Growing Spheres CF Generator
    
    Basit ama etkili: Orijinal noktadan başlayarak
    giderek büyüyen küreler içinde CF ara
    
    Referans: Laugel et al. "Inverse Classification for Comparison-based Interpretability" (2018)
    """
    
    def __init__(
        self,
        classifier: nn.Module,
        feature_count: int,
        num_cfs: int = 5,
        initial_radius: float = 0.1,
        radius_growth: float = 1.5,
        max_radius: float = 5.0,
        samples_per_sphere: int = 100,
        device: str = "cpu"
    ):
        self.classifier = classifier
        self.classifier.eval()
        self.feature_count = feature_count
        self.num_cfs = num_cfs
        self.initial_radius = initial_radius
        self.radius_growth = radius_growth
        self.max_radius = max_radius
        self.samples_per_sphere = samples_per_sphere
        self.device = device
    
    @torch.no_grad()
    def generate(
        self,
        query_instance: torch.Tensor,
        target_class: int = 0
    ) -> DiffusionCFResult:
        """Generate CFs using growing spheres"""
        import time
        start_time = time.time()
        
        query = query_instance.to(self.device)
        if query.dim() == 1:
            query = query.unsqueeze(0)
        
        original_pred = self.classifier(query).argmax(1).item()
        
        counterfactuals = []
        cf_preds = []
        radius = self.initial_radius
        
        while len(counterfactuals) < self.num_cfs and radius <= self.max_radius:
            # Sample points on sphere
            samples = self._sample_on_sphere(query.squeeze(0), radius, self.samples_per_sphere)
            
            # Check predictions
            preds = self.classifier(samples).argmax(1)
            
            # Find CFs
            cf_mask = preds == target_class
            new_cfs = samples[cf_mask]
            
            for cf in new_cfs:
                if len(counterfactuals) < self.num_cfs:
                    counterfactuals.append(cf.cpu())
                    cf_preds.append(target_class)
            
            radius *= self.radius_growth
        
        # Fill remaining with closest attempts if needed
        while len(counterfactuals) < self.num_cfs:
            noise = torch.randn(self.feature_count, device=self.device) * radius
            cf = (query.squeeze(0) + noise).cpu()
            pred = self.classifier(cf.unsqueeze(0).to(self.device)).argmax(1).item()
            counterfactuals.append(cf)
            cf_preds.append(pred)
        
        generation_time = time.time() - start_time
        
        # Compute metrics
        validity = sum(1 for p in cf_preds if p == target_class) / len(cf_preds)
        distances = [torch.abs(cf - query.cpu().squeeze(0)).mean().item() for cf in counterfactuals]
        proximity = np.mean(distances)
        changes = [torch.sum(torch.abs(cf - query.cpu().squeeze(0)) > 0.01).item() for cf in counterfactuals]
        sparsity = np.mean(changes)
        
        if len(counterfactuals) > 1:
            pairwise = [torch.norm(counterfactuals[i] - counterfactuals[j]).item() 
                       for i in range(len(counterfactuals)) for j in range(i+1, len(counterfactuals))]
            diversity = np.mean(pairwise) if pairwise else 0.0
        else:
            diversity = 0.0
        
        return DiffusionCFResult(
            original=query.squeeze(0).cpu(),
            counterfactuals=counterfactuals,
            original_pred=original_pred,
            cf_preds=cf_preds,
            validity=validity,
            proximity=proximity,
            sparsity=sparsity,
            diversity=diversity,
            realism_score=1.0 / (1.0 + proximity),
            generation_time=generation_time
        )
    
    def _sample_on_sphere(
        self,
        center: torch.Tensor,
        radius: float,
        n_samples: int
    ) -> torch.Tensor:
        """Sample points uniformly on a hypersphere"""
        # Sample from normal distribution
        samples = torch.randn(n_samples, self.feature_count, device=self.device)
        
        # Normalize to unit sphere
        norms = samples.norm(dim=1, keepdim=True)
        samples = samples / norms
        
        # Scale to radius and shift to center
        samples = samples * radius + center
        
        return samples
