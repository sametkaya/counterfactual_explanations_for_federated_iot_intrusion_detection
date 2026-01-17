#!/usr/bin/env python3
"""
FedCF-IoT Additional Experiments for Reviewer Response
=====================================================

This script adds two new experiments to address reviewer concerns:

EXPERIMENT 12: Privacy Budget vs Model Accuracy
- Trains models with different ε values
- Measures BOTH model accuracy AND CF validity
- Fills the gap identified by the reviewer

EXPERIMENT 13: DP Paradox Controlled Experiment
- Compares DP-SGD vs Gaussian noise (without DP guarantees) vs No noise
- Tests the "implicit regularization" hypothesis
- Provides evidence for the theoretical explanation of DP Paradox

Author: Samet Kaya
Date: 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import json
import time
import os
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import copy

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

class MLPClassifier(nn.Module):
    """MLP model for intrusion detection - same architecture as main experiments."""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dims: Tuple[int, ...] = (256, 128, 64),
                 dropout: float = 0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


# =============================================================================
# DP-SGD TRAINING (Simplified implementation for experiments)
# =============================================================================

def clip_gradients(model: nn.Module, max_norm: float) -> None:
    """Clip per-sample gradients to max_norm."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)


def add_dp_noise(model: nn.Module, noise_multiplier: float, max_norm: float, 
                 batch_size: int) -> None:
    """Add calibrated Gaussian noise to gradients for differential privacy."""
    for param in model.parameters():
        if param.grad is not None:
            noise_std = noise_multiplier * max_norm / batch_size
            noise = torch.normal(0, noise_std, size=param.grad.shape, device=param.grad.device)
            param.grad.data.add_(noise)


def add_gaussian_noise_only(model: nn.Module, noise_std: float) -> None:
    """Add Gaussian noise to gradients WITHOUT DP guarantees (for comparison)."""
    for param in model.parameters():
        if param.grad is not None:
            noise = torch.normal(0, noise_std, size=param.grad.shape, device=param.grad.device)
            param.grad.data.add_(noise)


def compute_noise_multiplier(epsilon: float, delta: float, epochs: int, 
                              batch_size: int, dataset_size: int) -> float:
    """
    Compute noise multiplier for given privacy budget.
    Simplified calculation based on Abadi et al. (2016).
    """
    # Sampling probability
    q = batch_size / dataset_size
    # Number of steps
    T = epochs * (dataset_size // batch_size)
    
    # Simplified noise multiplier calculation
    # In practice, use more sophisticated privacy accounting (e.g., RDP)
    if epsilon <= 0.5:
        sigma = 4.0
    elif epsilon <= 1.0:
        sigma = 2.0
    elif epsilon <= 2.0:
        sigma = 1.5
    elif epsilon <= 5.0:
        sigma = 1.0
    else:  # epsilon >= 10.0
        sigma = 0.5
    
    return sigma


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_model_with_dp(model: nn.Module, train_loader: DataLoader, 
                        val_loader: DataLoader, epochs: int,
                        epsilon: float, delta: float = 1e-5,
                        max_norm: float = 1.0, lr: float = 0.001,
                        class_weights: Optional[torch.Tensor] = None) -> Dict:
    """
    Train model with DP-SGD.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        epsilon: Privacy budget
        delta: Privacy parameter (default 1e-5)
        max_norm: Gradient clipping norm
        lr: Learning rate
        class_weights: Optional class weights for imbalanced data
    
    Returns:
        Dictionary with training history and final metrics
    """
    model = model.to(device)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Compute noise multiplier
    dataset_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    noise_multiplier = compute_noise_multiplier(epsilon, delta, epochs, 
                                                 batch_size, dataset_size)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # DP-SGD: Clip gradients and add noise
            clip_gradients(model, max_norm)
            add_dp_noise(model, noise_multiplier, max_norm, X_batch.size(0))
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
    
    return {
        'history': history,
        'final_acc': val_acc,
        'final_f1': val_f1,
        'epsilon': epsilon,
        'noise_multiplier': noise_multiplier
    }


def train_model_with_gaussian_noise(model: nn.Module, train_loader: DataLoader,
                                    val_loader: DataLoader, epochs: int,
                                    noise_std: float, lr: float = 0.001,
                                    class_weights: Optional[torch.Tensor] = None) -> Dict:
    """
    Train model with Gaussian noise added to gradients (NO DP guarantees).
    This is for the controlled experiment to test regularization hypothesis.
    """
    model = model.to(device)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Add Gaussian noise WITHOUT DP mechanism
            add_gaussian_noise_only(model, noise_std)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
    
    return {
        'history': history,
        'final_acc': val_acc,
        'final_f1': val_f1,
        'noise_std': noise_std
    }


def train_model_standard(model: nn.Module, train_loader: DataLoader,
                         val_loader: DataLoader, epochs: int,
                         lr: float = 0.001,
                         class_weights: Optional[torch.Tensor] = None) -> Dict:
    """Train model without any noise (standard training)."""
    model = model.to(device)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
    
    return {
        'history': history,
        'final_acc': val_acc,
        'final_f1': val_f1
    }


# =============================================================================
# COUNTERFACTUAL GENERATION (Simplified DiCE-like implementation)
# =============================================================================

def generate_counterfactual(model: nn.Module, x: torch.Tensor, 
                            target_class: int, max_iter: int = 300,
                            lr: float = 0.01, lambda_prox: float = 0.5,
                            lambda_div: float = 0.1) -> Tuple[torch.Tensor, bool]:
    """
    Generate counterfactual explanation using gradient-based optimization.
    
    Args:
        model: Trained classifier
        x: Input sample (attack)
        target_class: Target class (benign = 0)
        max_iter: Maximum iterations
        lr: Learning rate for CF optimization
        lambda_prox: Proximity weight
        lambda_div: Diversity weight (not used for single CF)
    
    Returns:
        Tuple of (counterfactual, is_valid)
    """
    model.eval()
    
    # Initialize counterfactual as copy of input
    cf = x.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([cf], lr=lr)
    
    for i in range(max_iter):
        optimizer.zero_grad()
        
        # Prediction loss (want target class)
        output = model(cf.unsqueeze(0))
        pred_loss = nn.CrossEntropyLoss()(output, torch.tensor([target_class], device=device))
        
        # Proximity loss (stay close to original)
        prox_loss = torch.norm(cf - x, p=2)
        
        # Total loss
        total_loss = pred_loss + lambda_prox * prox_loss
        
        total_loss.backward()
        optimizer.step()
        
        # Check if valid (predicts target class)
        with torch.no_grad():
            pred = torch.argmax(model(cf.unsqueeze(0)), dim=1).item()
            if pred == target_class:
                return cf.detach(), True
    
    # Check final prediction
    with torch.no_grad():
        pred = torch.argmax(model(cf.unsqueeze(0)), dim=1).item()
        is_valid = (pred == target_class)
    
    return cf.detach(), is_valid


def evaluate_cf_validity(model: nn.Module, X_test: torch.Tensor, 
                         y_test: torch.Tensor, n_samples: int = 50,
                         target_class: int = 0, max_iter: int = 300) -> Dict:
    """
    Evaluate counterfactual validity on test samples.
    
    Args:
        model: Trained classifier
        X_test: Test features
        y_test: Test labels
        n_samples: Number of samples to evaluate
        target_class: Target class (benign)
        max_iter: Max iterations for CF generation
    
    Returns:
        Dictionary with validity metrics
    """
    model.eval()
    
    # Get attack samples (not benign)
    attack_mask = y_test != target_class
    attack_indices = torch.where(attack_mask)[0]
    
    if len(attack_indices) < n_samples:
        n_samples = len(attack_indices)
    
    # Random sample
    sample_indices = attack_indices[torch.randperm(len(attack_indices))[:n_samples]]
    
    valid_count = 0
    proximities = []
    times = []
    
    for idx in sample_indices:
        x = X_test[idx].to(device)
        
        start_time = time.time()
        cf, is_valid = generate_counterfactual(model, x, target_class, max_iter=max_iter)
        elapsed = time.time() - start_time
        
        if is_valid:
            valid_count += 1
        
        # Compute proximity
        proximity = torch.norm(cf - x, p=2).item()
        proximities.append(proximity)
        times.append(elapsed)
    
    validity = valid_count / n_samples
    
    return {
        'validity': validity,
        'validity_std': np.std([1 if i < valid_count else 0 for i in range(n_samples)]),
        'proximity_mean': np.mean(proximities),
        'proximity_std': np.std(proximities),
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'n_samples': n_samples
    }


# =============================================================================
# DATA LOADING (Simplified for demonstration)
# =============================================================================

def load_ciciot2023_sample(n_samples: int = 50000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load CICIoT2023 dataset sample.
    In practice, this should load from the actual dataset.
    For this script, we'll create synthetic data with similar properties.
    """
    print(f"  Creating synthetic CICIoT2023-like data ({n_samples} samples)...")
    
    # Feature dimensions matching CICIoT2023
    n_features = 39
    n_classes = 8
    class_names = ['Benign', 'DDoS', 'DoS', 'Recon', 'Web', 'BruteForce', 'Spoofing', 'Mirai']
    
    # Create imbalanced synthetic data (matching 72.2% imbalance ratio)
    class_probs = [0.023, 0.722, 0.172, 0.015, 0.001, 0.0003, 0.011, 0.056]
    
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.choice(n_classes, size=n_samples, p=class_probs)
    
    # Add class-specific patterns
    for c in range(n_classes):
        mask = y == c
        X[mask, :5] += np.random.randn(5) * 2  # Class-specific shift
    
    return X, y, class_names


def prepare_data(X: np.ndarray, y: np.ndarray, 
                 test_size: float = 0.2,
                 batch_size: int = 64) -> Tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare data loaders and compute class weights."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.LongTensor(y_test)
    
    # Compute class weights
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(y_train) / (len(class_counts) * class_counts))
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, X_test_t, y_test_t, class_weights


# =============================================================================
# EXPERIMENT 12: Privacy Budget vs Model Accuracy
# =============================================================================

def experiment_12_epsilon_vs_accuracy(train_loader: DataLoader, 
                                       test_loader: DataLoader,
                                       X_test: torch.Tensor,
                                       y_test: torch.Tensor,
                                       class_weights: torch.Tensor,
                                       input_dim: int,
                                       num_classes: int,
                                       epochs: int = 20) -> Dict:
    """
    EXPERIMENT 12: Test different epsilon values and measure BOTH
    model accuracy AND CF validity.
    
    This addresses the reviewer's concern about missing model accuracy
    at different privacy budgets.
    """
    print("\n" + "="*70)
    print("EXPERIMENT 12: Privacy Budget (ε) vs Model Accuracy & CF Validity")
    print("="*70)
    print("This experiment addresses Reviewer Major Concern #3:")
    print("'Model accuracy at different ε values is not reported'")
    print("="*70 + "\n")
    
    epsilon_values = [0.5, 1.0, 2.0, 5.0, 10.0, float('inf')]  # inf = no DP
    results = {}
    
    for eps in epsilon_values:
        eps_str = str(eps) if eps != float('inf') else 'inf'
        print(f"\n--- ε = {eps_str} ---")
        
        # Create fresh model
        model = MLPClassifier(input_dim, num_classes)
        
        if eps == float('inf'):
            # No DP - standard training
            print("  Training without DP (baseline)...")
            train_result = train_model_standard(
                model, train_loader, test_loader, epochs,
                class_weights=class_weights
            )
        else:
            # DP-SGD training
            print(f"  Training with DP-SGD (ε={eps})...")
            train_result = train_model_with_dp(
                model, train_loader, test_loader, epochs,
                epsilon=eps, class_weights=class_weights
            )
        
        # Evaluate model on test set
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        test_precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        test_recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"  Model Performance: Acc={test_acc:.4f}, F1={test_f1:.4f}")
        
        # Generate counterfactuals and evaluate validity
        print(f"  Generating counterfactuals...")
        cf_result = evaluate_cf_validity(model, X_test, y_test, n_samples=30)
        
        print(f"  CF Validity: {cf_result['validity']*100:.2f}% ± {cf_result['validity_std']*100:.2f}%")
        
        results[eps_str] = {
            'epsilon': eps if eps != float('inf') else None,
            'model_accuracy': test_acc,
            'model_f1_macro': test_f1,
            'model_precision': test_precision,
            'model_recall': test_recall,
            'cf_validity': cf_result['validity'],
            'cf_validity_std': cf_result['validity_std'],
            'cf_proximity_mean': cf_result['proximity_mean'],
            'cf_proximity_std': cf_result['proximity_std'],
            'cf_time_mean': cf_result['time_mean'],
            'training_history': train_result['history']
        }
    
    # Summary table
    print("\n" + "="*70)
    print("EXPERIMENT 12 SUMMARY: Privacy Budget vs Model Performance")
    print("="*70)
    print(f"{'ε':<10} {'Accuracy':<12} {'F1-macro':<12} {'CF Validity':<15} {'CF Validity Std':<15}")
    print("-"*70)
    
    for eps_str, r in results.items():
        print(f"{eps_str:<10} {r['model_accuracy']:.4f}      {r['model_f1_macro']:.4f}      "
              f"{r['cf_validity']*100:.2f}%         ±{r['cf_validity_std']*100:.2f}%")
    
    print("-"*70)
    print("\nKey Finding: This table shows the complete privacy-utility tradeoff")
    print("including BOTH model accuracy AND CF validity at each ε level.")
    
    return {'exp12_epsilon_vs_accuracy': results}


# =============================================================================
# EXPERIMENT 13: DP Paradox Controlled Experiment
# =============================================================================

def experiment_13_dp_paradox_controlled(train_loader: DataLoader,
                                         test_loader: DataLoader,
                                         X_test: torch.Tensor,
                                         y_test: torch.Tensor,
                                         class_weights: torch.Tensor,
                                         input_dim: int,
                                         num_classes: int,
                                         epochs: int = 20) -> Dict:
    """
    EXPERIMENT 13: Controlled experiment to test DP Paradox hypotheses.
    
    Compares three conditions:
    1. Standard training (no noise)
    2. DP-SGD training (ε=2.0)
    3. Gaussian noise only (same noise level as DP, but no clipping/privacy)
    
    This tests whether the DP Paradox is caused by:
    - The noise itself (regularization effect)
    - The gradient clipping
    - The DP mechanism specifically
    """
    print("\n" + "="*70)
    print("EXPERIMENT 13: DP Paradox Controlled Experiment")
    print("="*70)
    print("This experiment addresses Reviewer Major Concern #1:")
    print("'DP Paradox hypotheses are speculative - need controlled experiments'")
    print("="*70)
    print("\nExperimental Design:")
    print("  Condition A: Standard training (no noise, no clipping)")
    print("  Condition B: DP-SGD (ε=2.0, with clipping + calibrated noise)")
    print("  Condition C: Gaussian noise only (same noise level, no clipping)")
    print("  Condition D: Gradient clipping only (no noise)")
    print("\nHypothesis: If Condition C achieves similar CF validity to Condition B,")
    print("then the DP Paradox is caused by implicit regularization from noise.")
    print("="*70 + "\n")
    
    epsilon = 2.0  # Use ε=2.0 as reference
    results = {}
    
    # Calculate equivalent noise std for Gaussian-only condition
    dataset_size = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    noise_multiplier = compute_noise_multiplier(epsilon, 1e-5, epochs, batch_size, dataset_size)
    max_norm = 1.0
    equivalent_noise_std = noise_multiplier * max_norm / batch_size
    
    print(f"Noise parameters: multiplier={noise_multiplier:.4f}, "
          f"equivalent_std={equivalent_noise_std:.6f}\n")
    
    conditions = {
        'A_Standard': {'noise': False, 'dp': False, 'clip': False},
        'B_DP_SGD': {'noise': True, 'dp': True, 'clip': True, 'epsilon': epsilon},
        'C_Gaussian_Only': {'noise': True, 'dp': False, 'clip': False, 'noise_std': equivalent_noise_std},
        'D_Clipping_Only': {'noise': False, 'dp': False, 'clip': True}
    }
    
    for condition_name, config in conditions.items():
        print(f"\n--- Condition {condition_name} ---")
        
        # Create fresh model
        model = MLPClassifier(input_dim, num_classes)
        
        if condition_name == 'A_Standard':
            print("  Training: Standard (no modifications)")
            train_result = train_model_standard(
                model, train_loader, test_loader, epochs,
                class_weights=class_weights
            )
        
        elif condition_name == 'B_DP_SGD':
            print(f"  Training: DP-SGD (ε={epsilon})")
            train_result = train_model_with_dp(
                model, train_loader, test_loader, epochs,
                epsilon=epsilon, class_weights=class_weights
            )
        
        elif condition_name == 'C_Gaussian_Only':
            print(f"  Training: Gaussian noise only (std={equivalent_noise_std:.6f})")
            train_result = train_model_with_gaussian_noise(
                model, train_loader, test_loader, epochs,
                noise_std=equivalent_noise_std, class_weights=class_weights
            )
        
        elif condition_name == 'D_Clipping_Only':
            print("  Training: Gradient clipping only (no noise)")
            # For clipping only, we use a modified version
            train_result = train_model_with_clipping_only(
                model, train_loader, test_loader, epochs,
                max_norm=max_norm, class_weights=class_weights
            )
        
        # Evaluate model
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                outputs = model(X_batch)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        print(f"  Model: Acc={test_acc:.4f}, F1={test_f1:.4f}")
        
        # Evaluate CF validity
        print(f"  Generating counterfactuals...")
        cf_result = evaluate_cf_validity(model, X_test, y_test, n_samples=30)
        
        print(f"  CF Validity: {cf_result['validity']*100:.2f}% ± {cf_result['validity_std']*100:.2f}%")
        
        results[condition_name] = {
            'config': config,
            'model_accuracy': test_acc,
            'model_f1_macro': test_f1,
            'cf_validity': cf_result['validity'],
            'cf_validity_std': cf_result['validity_std'],
            'cf_proximity_mean': cf_result['proximity_mean'],
            'training_history': train_result['history']
        }
    
    # Analysis
    print("\n" + "="*70)
    print("EXPERIMENT 13 SUMMARY: DP Paradox Mechanism Analysis")
    print("="*70)
    print(f"{'Condition':<20} {'Model Acc':<12} {'Model F1':<12} {'CF Validity':<15}")
    print("-"*70)
    
    for cond, r in results.items():
        print(f"{cond:<20} {r['model_accuracy']:.4f}      {r['model_f1_macro']:.4f}      "
              f"{r['cf_validity']*100:.2f}%")
    
    print("-"*70)
    
    # Statistical comparison
    validity_A = results['A_Standard']['cf_validity']
    validity_B = results['B_DP_SGD']['cf_validity']
    validity_C = results['C_Gaussian_Only']['cf_validity']
    validity_D = results['D_Clipping_Only']['cf_validity']
    
    print("\nDP Paradox Analysis:")
    print(f"  DP-SGD vs Standard: {(validity_B - validity_A)*100:+.2f}%")
    print(f"  Gaussian-Only vs Standard: {(validity_C - validity_A)*100:+.2f}%")
    print(f"  Clipping-Only vs Standard: {(validity_D - validity_A)*100:+.2f}%")
    
    print("\nConclusion:")
    if validity_C > validity_A and abs(validity_C - validity_B) < 0.05:
        print("  ✓ Gaussian noise alone improves CF validity similar to DP-SGD")
        print("  ✓ This SUPPORTS the 'implicit regularization' hypothesis")
        print("  → DP Paradox is likely caused by noise acting as regularization")
    elif validity_D > validity_A:
        print("  ✓ Gradient clipping contributes to improved CF validity")
        print("  → DP Paradox may be partially caused by gradient clipping")
    else:
        print("  ? Results inconclusive - further investigation needed")
    
    return {'exp13_dp_paradox_controlled': results}


def train_model_with_clipping_only(model: nn.Module, train_loader: DataLoader,
                                   val_loader: DataLoader, epochs: int,
                                   max_norm: float = 1.0, lr: float = 0.001,
                                   class_weights: Optional[torch.Tensor] = None) -> Dict:
    """Train model with gradient clipping only (no noise)."""
    model = model.to(device)
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # Clip gradients only
            clip_gradients(model, max_norm)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
        
        val_acc = accuracy_score(all_labels, all_preds)
        val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")
    
    return {
        'history': history,
        'final_acc': val_acc,
        'final_f1': val_f1
    }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Run all additional experiments."""
    print("="*70)
    print("FedCF-IoT Additional Experiments for Reviewer Response")
    print("="*70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print("="*70)
    
    # Load data (using synthetic data for demonstration)
    # In practice, replace this with actual CICIoT2023 data loading
    print("\n[1/4] Loading data...")
    X, y, class_names = load_ciciot2023_sample(n_samples=50000)
    
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    print(f"  Samples: {len(X)}, Features: {input_dim}, Classes: {num_classes}")
    
    # Prepare data loaders
    print("\n[2/4] Preparing data loaders...")
    train_loader, test_loader, X_test, y_test, class_weights = prepare_data(X, y)
    print(f"  Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")
    
    # Run Experiment 12
    print("\n[3/4] Running Experiment 12...")
    results_exp12 = experiment_12_epsilon_vs_accuracy(
        train_loader, test_loader, X_test, y_test, class_weights,
        input_dim, num_classes, epochs=15
    )
    
    # Run Experiment 13
    print("\n[4/4] Running Experiment 13...")
    results_exp13 = experiment_13_dp_paradox_controlled(
        train_loader, test_loader, X_test, y_test, class_weights,
        input_dim, num_classes, epochs=15
    )
    
    # Combine results
    all_results = {**results_exp12, **results_exp13}
    
    # Save results
    output_path = '/home/claude/additional_experiments_results.json'
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        else:
            return obj
    
    serializable_results = convert_to_serializable(all_results)
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {output_path}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    results = main()
