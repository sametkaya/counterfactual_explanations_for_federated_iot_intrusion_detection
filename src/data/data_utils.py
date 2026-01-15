"""
Data Loading and Preprocessing
==============================
CICIoT2023 ve Edge-IIoTset veri setleri için utilities
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Dict, Optional
import os


class IoTDataset(Dataset):
    """
    IoT-IDS Dataset class
    """
    
    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        transform: Optional[callable] = None
    ):
        self.features = features
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.features[idx]
        y = self.labels[idx]
        
        if self.transform:
            x = self.transform(x)
        
        return x, y


def generate_synthetic_ciciot2023(
    num_samples: int = 10000,
    num_features: int = 47,
    num_classes: int = 8,
    attack_ratio: float = 0.7,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sentetik CICIoT2023 benzeri veri üret
    
    Gerçek veri seti çok büyük olduğundan (~47 milyon sample),
    test ve development için sentetik veri kullanıyoruz.
    
    Args:
        num_samples: Toplam sample sayısı
        num_features: Feature sayısı (CICIoT2023: 47)
        num_classes: Sınıf sayısı (1 benign + 7 attack kategori)
        attack_ratio: Attack sample oranı
        seed: Random seed
        
    Returns:
        (features, labels) tuple
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_attacks = int(num_samples * attack_ratio)
    num_benign = num_samples - num_attacks
    
    # Class distribution (realistic imbalance)
    # 0: Benign, 1-7: Attack categories
    class_dist = {
        0: num_benign,           # Benign
        1: int(num_attacks * 0.35),  # DDoS
        2: int(num_attacks * 0.18),  # DoS
        3: int(num_attacks * 0.12),  # Recon
        4: int(num_attacks * 0.09),  # Web-based
        5: int(num_attacks * 0.08),  # BruteForce
        6: int(num_attacks * 0.05),  # Spoofing
        7: int(num_attacks * 0.13),  # Mirai
    }
    
    features_list = []
    labels_list = []
    
    for class_id, count in class_dist.items():
        if count == 0:
            continue
        
        # Her sınıf için farklı distribution
        if class_id == 0:  # Benign
            # Normal trafik: düşük rate, normal pattern
            mean = np.array([0.3] * num_features)
            std = np.array([0.1] * num_features)
        elif class_id == 1:  # DDoS
            # Yüksek packet rate
            mean = np.array([0.8] * num_features)
            mean[4:7] = 0.95  # Rate features
            std = np.array([0.15] * num_features)
        elif class_id == 2:  # DoS
            # Orta-yüksek rate
            mean = np.array([0.7] * num_features)
            mean[4:7] = 0.85
            std = np.array([0.2] * num_features)
        elif class_id == 3:  # Recon
            # Düşük rate, çok port
            mean = np.array([0.4] * num_features)
            mean[3] = 0.2  # Short duration
            std = np.array([0.25] * num_features)
        elif class_id == 4:  # Web-based
            # HTTP/HTTPS traffic
            mean = np.array([0.5] * num_features)
            mean[19:21] = 0.9  # HTTP, HTTPS
            std = np.array([0.15] * num_features)
        elif class_id == 5:  # BruteForce
            # Çok connection attempt
            mean = np.array([0.6] * num_features)
            mean[8] = 0.8  # SYN flag
            std = np.array([0.2] * num_features)
        elif class_id == 6:  # Spoofing
            # ARP/DNS patterns
            mean = np.array([0.45] * num_features)
            mean[28:31] = 0.85  # DHCP, ARP, ICMP
            std = np.array([0.2] * num_features)
        else:  # Mirai
            # Botnet pattern
            mean = np.array([0.75] * num_features)
            mean[7:14] = 0.9  # TCP flags
            std = np.array([0.1] * num_features)
        
        # Generate samples
        class_features = np.random.normal(mean, std, size=(count, num_features))
        class_features = np.clip(class_features, 0, 1)  # Normalize
        
        features_list.append(class_features)
        labels_list.extend([class_id] * count)
    
    # Combine
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    
    # Shuffle
    indices = np.random.permutation(len(labels))
    features = features[indices]
    labels = labels[indices]
    
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


def generate_synthetic_edge_iiotset(
    num_samples: int = 5000,
    num_features: int = 61,
    num_classes: int = 15,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sentetik Edge-IIoTset benzeri veri üret
    
    Args:
        num_samples: Toplam sample sayısı
        num_features: Feature sayısı (Edge-IIoTset: 61)
        num_classes: Sınıf sayısı (1 benign + 14 attack)
        seed: Random seed
    """
    np.random.seed(seed)
    
    # Her sınıf için sample sayısı (non-uniform)
    class_samples = np.random.dirichlet(np.ones(num_classes) * 0.5) * num_samples
    class_samples = class_samples.astype(int)
    class_samples[0] = num_samples - class_samples[1:].sum()  # Fix rounding
    
    features_list = []
    labels_list = []
    
    for class_id in range(num_classes):
        count = class_samples[class_id]
        if count <= 0:
            continue
        
        # Random mean/std for each class
        mean = np.random.uniform(0.2, 0.8, num_features)
        std = np.random.uniform(0.1, 0.3, num_features)
        
        class_features = np.random.normal(mean, std, size=(count, num_features))
        class_features = np.clip(class_features, 0, 1)
        
        features_list.append(class_features)
        labels_list.extend([class_id] * count)
    
    features = np.vstack(features_list)
    labels = np.array(labels_list)
    
    indices = np.random.permutation(len(labels))
    
    return (
        torch.tensor(features[indices], dtype=torch.float32),
        torch.tensor(labels[indices], dtype=torch.long)
    )


def create_federated_partitions(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_clients: int = 10,
    iid: bool = True,
    alpha: float = 0.5,
    seed: int = 42
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Federated learning için veri partitioning
    
    Args:
        features: Feature tensor
        labels: Label tensor
        num_clients: Client sayısı
        iid: IID veya Non-IID dağılım
        alpha: Dirichlet alpha (non-IID için, düşük = daha non-iid)
        seed: Random seed
        
    Returns:
        List of (features, labels) tuples for each client
    """
    np.random.seed(seed)
    
    num_samples = len(labels)
    
    if iid:
        # IID: Random partition
        indices = np.random.permutation(num_samples)
        split_size = num_samples // num_clients
        
        partitions = []
        for i in range(num_clients):
            start = i * split_size
            end = (i + 1) * split_size if i < num_clients - 1 else num_samples
            idx = indices[start:end]
            partitions.append((features[idx], labels[idx]))
        
        return partitions
    
    else:
        # Non-IID: Dirichlet-based partition
        num_classes = len(torch.unique(labels))
        
        # Her class için Dirichlet distribution
        class_indices = [np.where(labels.numpy() == c)[0] for c in range(num_classes)]
        
        client_indices = [[] for _ in range(num_clients)]
        
        for c in range(num_classes):
            np.random.shuffle(class_indices[c])
            
            # Dirichlet proportions
            proportions = np.random.dirichlet([alpha] * num_clients)
            proportions = (proportions * len(class_indices[c])).astype(int)
            
            # Fix rounding
            proportions[-1] = len(class_indices[c]) - proportions[:-1].sum()
            
            # Distribute
            start = 0
            for client_id in range(num_clients):
                end = start + proportions[client_id]
                client_indices[client_id].extend(class_indices[c][start:end])
                start = end
        
        # Create partitions
        partitions = []
        for idx_list in client_indices:
            if len(idx_list) > 0:
                idx = np.array(idx_list)
                partitions.append((features[idx], labels[idx]))
            else:
                # Empty partition - assign random samples
                idx = np.random.choice(num_samples, 100, replace=False)
                partitions.append((features[idx], labels[idx]))
        
        return partitions


def create_data_loaders(
    features: torch.Tensor,
    labels: torch.Tensor,
    test_size: float = 0.2,
    batch_size: int = 64,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Train/test DataLoader oluştur
    """
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        features.numpy(), labels.numpy(),
        test_size=test_size,
        stratify=labels.numpy(),
        random_state=seed
    )
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)
    
    # Datasets
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def get_data_statistics(features: torch.Tensor, labels: torch.Tensor) -> Dict:
    """Veri seti istatistikleri"""
    num_samples = len(labels)
    num_features = features.size(1)
    num_classes = len(torch.unique(labels))
    
    class_counts = {}
    for c in range(num_classes):
        class_counts[c] = (labels == c).sum().item()
    
    return {
        'num_samples': num_samples,
        'num_features': num_features,
        'num_classes': num_classes,
        'class_distribution': class_counts,
        'feature_mean': features.mean(dim=0).tolist(),
        'feature_std': features.std(dim=0).tolist(),
    }


# Test
if __name__ == "__main__":
    print("Testing Data Utilities...")
    
    # Generate synthetic data
    print("\n1. Generating synthetic CICIoT2023 data...")
    X, y = generate_synthetic_ciciot2023(num_samples=5000)
    print(f"   Shape: X={X.shape}, y={y.shape}")
    
    # Statistics
    stats = get_data_statistics(X, y)
    print(f"   Classes: {stats['num_classes']}")
    print(f"   Distribution: {stats['class_distribution']}")
    
    # Data loaders
    print("\n2. Creating data loaders...")
    train_loader, test_loader = create_data_loaders(X, y)
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Test batches: {len(test_loader)}")
    
    # Federated partitions
    print("\n3. Creating federated partitions...")
    
    print("   IID partitioning (5 clients):")
    iid_partitions = create_federated_partitions(X, y, num_clients=5, iid=True)
    for i, (px, py) in enumerate(iid_partitions):
        print(f"     Client {i}: {len(py)} samples")
    
    print("\n   Non-IID partitioning (5 clients, alpha=0.5):")
    noniid_partitions = create_federated_partitions(X, y, num_clients=5, iid=False, alpha=0.5)
    for i, (px, py) in enumerate(noniid_partitions):
        class_dist = {c: (py == c).sum().item() for c in range(8)}
        dominant = max(class_dist, key=class_dist.get)
        print(f"     Client {i}: {len(py)} samples, dominant class: {dominant}")
    
    # Edge-IIoTset
    print("\n4. Generating synthetic Edge-IIoTset data...")
    X_edge, y_edge = generate_synthetic_edge_iiotset(num_samples=3000)
    print(f"   Shape: X={X_edge.shape}, y={y_edge.shape}")
    
    print("\nData utilities test completed!")
