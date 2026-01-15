"""
Real Dataset Loaders
====================
CICIoT2023, Edge-IIoTset ve ToN-IoT veri setleri için data loading

Senin Klasör Yapın:
    dataset/
    ├── ciciot2023/
    │   ├── Merged01.csv
    │   ├── Merged02.csv
    │   ├── Merged03.csv
    │   ├── Merged04.csv
    │   ├── Merged05.csv
    │   └── Merged06.csv
    ├── edge-iiot/
    │   └── DNN-EdgeIIoT-dataset.csv
    └── toniot/
        └── train_test_network.csv

Kullanım:
    from src.data.real_data_loader import CICIoT2023Loader, EdgeIIoTLoader, ToNIoTLoader
    
    # CICIoT2023
    loader = CICIoT2023Loader('dataset/ciciot2023/')
    X_train, X_test, y_train, y_test, info = loader.load()
    
    # Edge-IIoT
    loader = EdgeIIoTLoader('dataset/edge-iiot/')
    X_train, X_test, y_train, y_test, info = loader.load()
    
    # ToN-IoT
    loader = ToNIoTLoader('dataset/toniot/')
    X_train, X_test, y_train, y_test, info = loader.load()
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, List, Any
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CICIoT2023 LOADER
# =============================================================================

class CICIoT2023Loader:
    """
    CICIoT2023 Dataset Loader
    
    - 46 features (label hariç)
    - 8 sınıf: Benign + 7 attack category
    - ~47 milyon sample (full dataset)
    - Dosyalar: Merged01.csv - Merged06.csv
    """
    
    # CICIoT2023 attack mapping - CASE INSENSITIVE
    # Tüm key'ler lowercase olarak tutulacak, lookup sırasında da lowercase yapılacak
    ATTACK_MAPPING = {
        # Benign
        'benigntraffic': 'Benign',
        'benign': 'Benign',
        
        # DDoS - tüm varyasyonlar
        'ddos-icmp_flood': 'DDoS',
        'ddos-udp_flood': 'DDoS', 
        'ddos-tcp_flood': 'DDoS',
        'ddos-pshack_flood': 'DDoS',
        'ddos-syn_flood': 'DDoS',
        'ddos-rstfinflood': 'DDoS',
        'ddos-synonymousip_flood': 'DDoS',
        'ddos-icmp_fragmentation': 'DDoS',
        'ddos-udp_fragmentation': 'DDoS',
        'ddos-ack_fragmentation': 'DDoS',
        'ddos-http_flood': 'DDoS',
        'ddos-slowloris': 'DDoS',
        'ddos_icmp': 'DDoS',
        'ddos_udp': 'DDoS',
        'ddos_tcp': 'DDoS',
        'ddos_http': 'DDoS',
        
        # DoS
        'dos-udp_flood': 'DoS',
        'dos-tcp_flood': 'DoS',
        'dos-syn_flood': 'DoS',
        'dos-http_flood': 'DoS',
        'dos_udp': 'DoS',
        'dos_tcp': 'DoS',
        
        # Recon
        'recon-pingsweep': 'Recon',
        'recon-osscan': 'Recon',
        'recon-portscan': 'Recon',
        'recon-hostdiscovery': 'Recon',
        'vulnerabilityscan': 'Recon',
        'vulnerability_scanner': 'Recon',
        'portscan': 'Recon',
        'port_scanning': 'Recon',
        'fingerprinting': 'Recon',
        
        # Web-based
        'sqlinjection': 'Web',
        'sql_injection': 'Web',
        'commandinjection': 'Web',
        'command_injection': 'Web',
        'xss': 'Web',
        'uploading_attack': 'Web',
        'uploading': 'Web',
        'backdoor_malware': 'Web',
        'backdoor': 'Web',
        
        # BruteForce
        'bruteforce': 'BruteForce',
        'brute_force': 'BruteForce',
        'dictionarybruteforce': 'BruteForce',
        'dictionary_bruteforce': 'BruteForce',
        'password': 'BruteForce',
        
        # Spoofing
        'dns_spoofing': 'Spoofing',
        'mitm-arpspoofing': 'Spoofing',
        'mitm': 'Spoofing',
        'arp_spoofing': 'Spoofing',
        
        # Mirai
        'mirai-greeth_flood': 'Mirai',
        'mirai-greip_flood': 'Mirai',
        'mirai-udpplain': 'Mirai',
        'mirai': 'Mirai',
        
        # Ransomware (Edge-IIoT'da var)
        'ransomware': 'Mirai',  # veya ayrı kategori
    }
    
    CATEGORY_TO_ID = {
        'Benign': 0,
        'DDoS': 1,
        'DoS': 2,
        'Recon': 3,
        'Web': 4,
        'BruteForce': 5,
        'Spoofing': 6,
        'Mirai': 7
    }
    
    def __init__(
        self,
        data_path: str,
        sample_fraction: float = 1.0,
        max_samples: Optional[int] = None,
        random_state: int = 42
    ):
        """
        Args:
            data_path: dataset/ciciot2023/ klasörü
            sample_fraction: Her dosyadan alınacak oran (0.1 = %10)
            max_samples: Maksimum toplam sample sayısı
            random_state: Random seed
        """
        self.data_path = data_path
        self.sample_fraction = sample_fraction
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.feature_names = None
        self.label_encoder = None
    
    def load(
        self,
        test_size: float = 0.2,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Veriyi yükle ve preprocess et
        
        Returns:
            X_train, X_test, y_train, y_test, info_dict
        """
        print("="*60)
        print("Loading CICIoT2023 Dataset")
        print("="*60)
        
        # CSV dosyalarını bul
        csv_files = sorted(glob.glob(os.path.join(self.data_path, 'Merged*.csv')))
        
        if not csv_files:
            csv_files = sorted(glob.glob(os.path.join(self.data_path, '*.csv')))
        
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")
        
        print(f"Found {len(csv_files)} CSV files")
        
        # Dosyaları yükle
        dfs = []
        for f in csv_files:
            print(f"  Loading {os.path.basename(f)}...", end=" ")
            df = pd.read_csv(f, low_memory=False)
            
            # Sampling
            if self.sample_fraction < 1.0:
                df = df.sample(frac=self.sample_fraction, random_state=self.random_state)
            
            dfs.append(df)
            print(f"({len(df):,} rows)")
        
        # Birleştir
        data = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal samples: {len(data):,}")
        
        # Max samples limit
        if self.max_samples and len(data) > self.max_samples:
            data = data.sample(n=self.max_samples, random_state=self.random_state)
            print(f"Sampled to: {len(data):,}")
        
        # Label column bul
        label_col = None
        for col in ['label', 'Label', 'attack', 'Attack', 'class']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            # Son column'u label olarak al
            label_col = data.columns[-1]
        
        print(f"Label column: {label_col}")
        
        # Labels
        y_raw = data[label_col].astype(str)
        
        # Attack category mapping - CASE INSENSITIVE
        def map_label(label):
            label_lower = str(label).lower().strip()
            return self.ATTACK_MAPPING.get(label_lower, 'Unknown')
        
        y_category = y_raw.map(map_label)
        
        # Unknown'ları kontrol et ve smart mapping yap
        unknown_mask = y_category == 'Unknown'
        if unknown_mask.any():
            unknown_labels = y_raw[unknown_mask].unique()
            print(f"  Found {len(unknown_labels)} unknown label types, attempting smart mapping...")
            
            # Smart mapping: label isminden kategori tahmin et
            def smart_map(label):
                label_lower = str(label).lower()
                if 'ddos' in label_lower:
                    return 'DDoS'
                elif 'dos' in label_lower:
                    return 'DoS'
                elif 'recon' in label_lower or 'scan' in label_lower or 'fingerprint' in label_lower:
                    return 'Recon'
                elif 'sql' in label_lower or 'xss' in label_lower or 'injection' in label_lower or 'backdoor' in label_lower or 'upload' in label_lower:
                    return 'Web'
                elif 'brute' in label_lower or 'password' in label_lower:
                    return 'BruteForce'
                elif 'spoof' in label_lower or 'mitm' in label_lower:
                    return 'Spoofing'
                elif 'mirai' in label_lower or 'ransomware' in label_lower:
                    return 'Mirai'
                elif 'benign' in label_lower or 'normal' in label_lower:
                    return 'Benign'
                else:
                    return 'Unknown'
            
            # Unknown olanları smart mapping ile güncelle
            for label in unknown_labels:
                smart_category = smart_map(label)
                if smart_category != 'Unknown':
                    mask = y_raw == label
                    y_category = y_category.where(~mask, smart_category)
                    print(f"    '{label}' -> '{smart_category}'")
            
            # Hala unknown kalanları kontrol et
            still_unknown = y_category == 'Unknown'
            if still_unknown.any():
                remaining = y_raw[still_unknown].unique()[:10]
                print(f"  Warning: Still unknown labels: {remaining}")
                # Unknown'ları çıkar
                data = data[~still_unknown]
                y_category = y_category[~still_unknown]
                y_raw = y_raw[~still_unknown]
                print(f"  Removed {still_unknown.sum()} unknown samples")
        
        # Numeric labels
        y = y_category.map(self.CATEGORY_TO_ID)
        
        # Features
        X = data.drop(columns=[label_col])
        
        # Non-numeric columns'ı çıkar
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        # Feature names
        self.feature_names = X.columns.tolist()
        
        # Missing values
        X = X.fillna(0)
        
        # Inf değerleri
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Final samples: {len(X):,}")
        
        # Class distribution
        print("\nClass distribution:")
        for cat, idx in sorted(self.CATEGORY_TO_ID.items(), key=lambda x: x[1]):
            count = (y == idx).sum()
            pct = count / len(y) * 100
            print(f"  {idx} - {cat}: {count:,} ({pct:.1f}%)")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values,
            test_size=test_size,
            stratify=y.values,
            random_state=self.random_state
        )
        
        # Normalize
        if normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # To tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")
        
        info = {
            'dataset': 'CICIoT2023',
            'num_features': len(self.feature_names),
            'num_classes': len(self.CATEGORY_TO_ID),
            'feature_names': self.feature_names,
            'class_names': list(self.CATEGORY_TO_ID.keys()),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return X_train, X_test, y_train, y_test, info


# =============================================================================
# EDGE-IIoT LOADER
# =============================================================================

class EdgeIIoTLoader:
    """
    Edge-IIoTset Dataset Loader
    
    - 61 features
    - 15 sınıf: Normal + 14 attack types
    - Dosya: DNN-EdgeIIoT-dataset.csv
    """
    
    ATTACK_MAPPING = {
        'Normal': 0,
        'DDoS_UDP': 1,
        'DDoS_ICMP': 2,
        'DDoS_HTTP': 3,
        'DDoS_TCP': 4,
        'SQL_injection': 5,
        'Vulnerability_scanner': 6,
        'Password': 7,
        'Uploading': 8,
        'Backdoor': 9,
        'Port_Scanning': 10,
        'XSS': 11,
        'Ransomware': 12,
        'Fingerprinting': 13,
        'MITM': 14
    }
    
    def __init__(
        self,
        data_path: str,
        sample_fraction: float = 1.0,
        max_samples: Optional[int] = None,
        random_state: int = 42
    ):
        self.data_path = data_path
        self.sample_fraction = sample_fraction
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load(
        self,
        test_size: float = 0.2,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Veriyi yükle"""
        print("="*60)
        print("Loading Edge-IIoTset Dataset")
        print("="*60)
        
        # CSV dosyasını bul
        csv_file = os.path.join(self.data_path, 'DNN-EdgeIIoT-dataset.csv')
        
        if not os.path.exists(csv_file):
            csv_files = glob.glob(os.path.join(self.data_path, '*.csv'))
            if csv_files:
                csv_file = csv_files[0]
            else:
                raise FileNotFoundError(f"No CSV file found in {self.data_path}")
        
        print(f"Loading {os.path.basename(csv_file)}...")
        data = pd.read_csv(csv_file, low_memory=False)
        print(f"Total samples: {len(data):,}")
        
        # Sampling
        if self.sample_fraction < 1.0:
            data = data.sample(frac=self.sample_fraction, random_state=self.random_state)
            print(f"Sampled to: {len(data):,}")
        
        if self.max_samples and len(data) > self.max_samples:
            data = data.sample(n=self.max_samples, random_state=self.random_state)
            print(f"Limited to: {len(data):,}")
        
        # Label column
        label_col = None
        for col in ['Attack_type', 'attack_type', 'Attack', 'label', 'Label']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            label_col = data.columns[-1]
        
        print(f"Label column: {label_col}")
        
        # Labels
        y_raw = data[label_col].astype(str)
        
        # Encode labels
        unique_labels = y_raw.unique()
        label_to_id = {}
        for i, label in enumerate(sorted(unique_labels)):
            if label in self.ATTACK_MAPPING:
                label_to_id[label] = self.ATTACK_MAPPING[label]
            else:
                label_to_id[label] = i
        
        y = y_raw.map(label_to_id)
        
        # Features
        X = data.drop(columns=[label_col])
        
        # Drop non-numeric and ID columns
        drop_cols = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.src.proto_ipv4', 
                     'arp.dst.proto_ipv4', 'Attack_label']
        drop_cols = [c for c in drop_cols if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
        
        # Numeric only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        self.feature_names = X.columns.tolist()
        
        # Clean data
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Features: {len(self.feature_names)}")
        
        # Class distribution
        print("\nClass distribution:")
        for label, idx in sorted(label_to_id.items(), key=lambda x: x[1]):
            count = (y == idx).sum()
            pct = count / len(y) * 100
            print(f"  {idx} - {label}: {count:,} ({pct:.1f}%)")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values,
            test_size=test_size,
            stratify=y.values,
            random_state=self.random_state
        )
        
        # Normalize
        if normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # To tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")
        
        info = {
            'dataset': 'Edge-IIoTset',
            'num_features': len(self.feature_names),
            'num_classes': len(label_to_id),
            'feature_names': self.feature_names,
            'class_names': list(label_to_id.keys()),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return X_train, X_test, y_train, y_test, info


# =============================================================================
# ToN-IoT LOADER
# =============================================================================

class ToNIoTLoader:
    """
    ToN-IoT Dataset Loader
    
    - 44 features
    - 10 sınıf: normal + 9 attack types
    - Dosya: train_test_network.csv
    """
    
    ATTACK_MAPPING = {
        'normal': 0,
        'backdoor': 1,
        'ddos': 2,
        'dos': 3,
        'injection': 4,
        'mitm': 5,
        'password': 6,
        'ransomware': 7,
        'scanning': 8,
        'xss': 9
    }
    
    def __init__(
        self,
        data_path: str,
        sample_fraction: float = 1.0,
        max_samples: Optional[int] = None,
        random_state: int = 42
    ):
        self.data_path = data_path
        self.sample_fraction = sample_fraction
        self.max_samples = max_samples
        self.random_state = random_state
        
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load(
        self,
        test_size: float = 0.2,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Veriyi yükle"""
        print("="*60)
        print("Loading ToN-IoT Dataset")
        print("="*60)
        
        csv_file = os.path.join(self.data_path, 'train_test_network.csv')
        
        if not os.path.exists(csv_file):
            csv_files = glob.glob(os.path.join(self.data_path, '*.csv'))
            if csv_files:
                csv_file = csv_files[0]
            else:
                raise FileNotFoundError(f"No CSV file found in {self.data_path}")
        
        print(f"Loading {os.path.basename(csv_file)}...")
        data = pd.read_csv(csv_file, low_memory=False)
        print(f"Total samples: {len(data):,}")
        
        # Sampling
        if self.sample_fraction < 1.0:
            data = data.sample(frac=self.sample_fraction, random_state=self.random_state)
        
        if self.max_samples and len(data) > self.max_samples:
            data = data.sample(n=self.max_samples, random_state=self.random_state)
        
        # Label column
        label_col = None
        for col in ['type', 'label', 'Label', 'attack', 'Attack']:
            if col in data.columns:
                label_col = col
                break
        
        if label_col is None:
            label_col = data.columns[-1]
        
        print(f"Label column: {label_col}")
        
        # Labels
        y_raw = data[label_col].astype(str).str.lower()
        
        # Encode
        unique_labels = y_raw.unique()
        label_to_id = {}
        for label in unique_labels:
            if label in self.ATTACK_MAPPING:
                label_to_id[label] = self.ATTACK_MAPPING[label]
            else:
                label_to_id[label] = len(label_to_id)
        
        y = y_raw.map(label_to_id)
        
        # Features
        X = data.drop(columns=[label_col])
        
        # Drop ID columns
        drop_cols = ['ts', 'src_ip', 'dst_ip', 'src_port', 'dst_port']
        drop_cols = [c for c in drop_cols if c in X.columns]
        if drop_cols:
            X = X.drop(columns=drop_cols)
        
        # Numeric only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        self.feature_names = X.columns.tolist()
        
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"Features: {len(self.feature_names)}")
        
        # Class distribution
        print("\nClass distribution:")
        for label, idx in sorted(label_to_id.items(), key=lambda x: x[1]):
            count = (y == idx).sum()
            pct = count / len(y) * 100
            print(f"  {idx} - {label}: {count:,} ({pct:.1f}%)")
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values,
            test_size=test_size,
            stratify=y.values,
            random_state=self.random_state
        )
        
        if normalize:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        print(f"\nTrain: {len(X_train):,}, Test: {len(X_test):,}")
        
        info = {
            'dataset': 'ToN-IoT',
            'num_features': len(self.feature_names),
            'num_classes': len(label_to_id),
            'feature_names': self.feature_names,
            'class_names': list(label_to_id.keys()),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        return X_train, X_test, y_train, y_test, info


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_data_loaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    batch_size: int = 64
) -> Tuple[DataLoader, DataLoader]:
    """DataLoader oluştur"""
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def load_dataset(
    dataset_name: str,
    data_path: str,
    sample_fraction: float = 1.0,
    max_samples: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Dataset yükle (convenience function)
    
    Args:
        dataset_name: 'ciciot2023', 'edge-iiot', 'toniot'
        data_path: Dataset klasörü
        sample_fraction: Sampling oranı
        max_samples: Max sample sayısı
    """
    loaders = {
        'ciciot2023': CICIoT2023Loader,
        'edge-iiot': EdgeIIoTLoader,
        'edge_iiot': EdgeIIoTLoader,
        'toniot': ToNIoTLoader,
        'ton-iot': ToNIoTLoader
    }
    
    if dataset_name.lower() not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(loaders.keys())}")
    
    loader_class = loaders[dataset_name.lower()]
    loader = loader_class(
        data_path=data_path,
        sample_fraction=sample_fraction,
        max_samples=max_samples
    )
    
    return loader.load()


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Real Data Loader Test")
    print("="*60)
    
    # Test paths (kendi yapına göre değiştir)
    BASE_PATH = "dataset"
    
    datasets = [
        ('ciciot2023', f'{BASE_PATH}/ciciot2023/'),
        ('edge-iiot', f'{BASE_PATH}/edge-iiot/'),
        ('toniot', f'{BASE_PATH}/toniot/')
    ]
    
    for name, path in datasets:
        print(f"\n{'='*60}")
        print(f"Testing {name}")
        print(f"{'='*60}")
        
        if os.path.exists(path):
            try:
                X_train, X_test, y_train, y_test, info = load_dataset(
                    name, path, 
                    sample_fraction=0.1,  # Test için %10
                    max_samples=10000     # Max 10k sample
                )
                print(f"\n✓ {name} loaded successfully!")
                print(f"  Features: {info['num_features']}")
                print(f"  Classes: {info['num_classes']}")
            except Exception as e:
                print(f"✗ Error loading {name}: {e}")
        else:
            print(f"✗ Path not found: {path}")
