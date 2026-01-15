"""
FedCF-IoT Configuration
=======================
Tüm hyperparameter ve ayarlar
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import torch

@dataclass
class ModelConfig:
    """IDS Model Konfigürasyonu"""
    input_dim: int = 47  # CICIoT2023 için 47, Edge-IIoTset için 61
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    num_classes: int = 8  # 7 attack + 1 benign
    dropout: float = 0.3
    activation: str = "relu"

@dataclass
class CFConfig:
    """Counterfactual Generation Konfigürasyonu"""
    num_cfs: int = 5
    max_iter: int = 500
    learning_rate: float = 0.01
    lambda_distance: float = 0.5
    lambda_diversity: float = 0.1
    lambda_feasibility: float = 1.0
    confidence_margin: float = 0.8
    early_stop_threshold: float = 0.95

@dataclass
class PrivacyConfig:
    """Differential Privacy Konfigürasyonu"""
    epsilon: float = 1.0
    delta: float = 1e-5
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.1
    secure_aggregation: bool = True

@dataclass
class FLConfig:
    """Federated Learning Konfigürasyonu"""
    num_clients: int = 10
    num_rounds: int = 50
    local_epochs: int = 5
    batch_size: int = 64
    client_fraction: float = 1.0  # Her round'da katılan client oranı
    aggregation_strategy: str = "fedavg"  # fedavg, fedprox, scaffold

@dataclass
class ExperimentConfig:
    """Deney Konfigürasyonu"""
    dataset: str = "ciciot2023"  # ciciot2023, edge_iiotset
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir: str = "./results"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    cf: CFConfig = field(default_factory=CFConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    fl: FLConfig = field(default_factory=FLConfig)

# Feature isimleri (CICIoT2023)
CICIOT_FEATURES = [
    'flow_duration', 'Header_Length', 'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number', 'psh_flag_number', 'ack_flag_number',
    'ece_flag_number', 'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'urg_count',
    'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
    'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std',
    'Tot size', 'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
]

# Attack kategorileri
ATTACK_CATEGORIES = {
    0: 'Benign',
    1: 'DDoS',
    2: 'DoS', 
    3: 'Recon',
    4: 'Web-based',
    5: 'BruteForce',
    6: 'Spoofing',
    7: 'Mirai'
}

# Immutable feature indices (değiştirilemez)
IMMUTABLE_FEATURES = [2, 26, 27, 28, 29, 30, 31, 32]  # Protocol, TCP, UDP, vs.

# Feature ranges
FEATURE_RANGES = {
    'flow_duration': (0, 3600),
    'Rate': (0, 1e9),
    'fin_flag_number': (0, 1),
    'syn_flag_number': (0, 1),
    'rst_flag_number': (0, 1),
    'psh_flag_number': (0, 1),
    'ack_flag_number': (0, 1),
}
