"""
Real Dataset Loaders - FIXED VERSION
=====================================
CICIoT2023, Edge-IIoTset ve ToN-IoT veri setleri için data loading

TÜM CATEGORICAL KOLONLAR ENCODE EDİLİR - Hiçbir feature kaybedilmez!

Klasör Yapısı:
    dataset/
    ├── ciciot2023/
    │   ├── Merged01.csv ... Merged63.csv
    ├── edge-iiot/
    │   └── DNN-EdgeIIoT-dataset.csv
    └── toniot/
        └── Processed_Network_dataset/
            └── Network_dataset_1.csv ... Network_dataset_23.csv
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# BASE LOADER CLASS
# =============================================================================

class BaseDatasetLoader:
    """
    Tüm dataset loader'lar için base class.
    Categorical encoding ve preprocessing burada merkezi olarak yapılır.
    """

    # Her dataset için override edilecek
    DROP_COLUMNS = []  # Sadece gerçek high-cardinality string kolonlar

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
        self.label_encoders = {}  # Categorical kolonlar için encoder'lar

    def _encode_categorical_columns(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Tüm object/categorical kolonları Label Encoding ile numeric'e çevir.
        """
        encoded_cols = []

        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                # NaN değerleri 'unknown' ile doldur
                X[col] = X[col].fillna('unknown').astype(str)

                # Label Encoding uygula
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])

                # Encoder'ı sakla
                self.label_encoders[col] = le
                encoded_cols.append(col)

        if encoded_cols:
            print(f"  Encoded {len(encoded_cols)} categorical columns: {encoded_cols}")

        return X

    def _clean_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        NaN, inf değerleri temizle ve tüm kolonların numeric olduğunu doğrula.
        """
        # NaN ve inf temizle
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)

        # Tüm kolonların numeric olduğunu doğrula
        for col in X.columns:
            if not np.issubdtype(X[col].dtype, np.number):
                print(f"  Warning: Converting non-numeric column '{col}' to numeric")
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)

        return X


# =============================================================================
# CICIoT2023 LOADER
# =============================================================================

class CICIoT2023Loader(BaseDatasetLoader):
    """
    CICIoT2023 Dataset Loader

    - ~46 features (tümü numeric)
    - 8 sınıf: Benign + 7 attack category
    - ~47 milyon sample (full dataset)
    """

    # CICIoT2023 hepsi numeric, sadece label çıkarılacak
    DROP_COLUMNS = []  # CICIoT2023'te string kolon yok

    # Attack category mapping
    ATTACK_MAPPING = {
        # Benign
        'benigntraffic': 'Benign',
        'benign': 'Benign',

        # DDoS
        'ddos-icmp_flood': 'DDoS', 'ddos-udp_flood': 'DDoS', 'ddos-tcp_flood': 'DDoS',
        'ddos-pshack_flood': 'DDoS', 'ddos-syn_flood': 'DDoS', 'ddos-rstfinflood': 'DDoS',
        'ddos-synonymousip_flood': 'DDoS', 'ddos-icmp_fragmentation': 'DDoS',
        'ddos-udp_fragmentation': 'DDoS', 'ddos-ack_fragmentation': 'DDoS',
        'ddos-http_flood': 'DDoS', 'ddos-slowloris': 'DDoS',

        # DoS
        'dos-udp_flood': 'DoS', 'dos-tcp_flood': 'DoS', 'dos-syn_flood': 'DoS',
        'dos-http_flood': 'DoS',

        # Mirai
        'mirai-greeth_flood': 'Mirai', 'mirai-greip_flood': 'Mirai',
        'mirai-udpplain': 'Mirai',

        # Recon
        'recon-pingsweep': 'Recon', 'recon-osscan': 'Recon', 'recon-portscan': 'Recon',
        'recon-hostdiscovery': 'Recon',

        # Spoofing
        'spoofing-dns': 'Spoofing', 'spoofing-arp': 'Spoofing',
        'dns_spoofing': 'Spoofing', 'arp_spoofing': 'Spoofing',

        # Web attacks
        'sqlinjection': 'Web', 'sql_injection': 'Web',
        'commandinjection': 'Web', 'backdoor_malware': 'Web',
        'xss': 'Web', 'uploadingattack': 'Web',
        'browserhijacking': 'Web', 'vulnerability_scanner': 'Web',

        # Brute Force
        'dictionarybruteforce': 'BruteForce', 'bruteforce': 'BruteForce',
    }

    CATEGORY_TO_ID = {
        'Benign': 0, 'DDoS': 1, 'DoS': 2, 'Recon': 3,
        'Web': 4, 'BruteForce': 5, 'Spoofing': 6, 'Mirai': 7
    }

    def load(
        self,
        test_size: float = 0.2,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """CICIoT2023 verisini yükle"""
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
            dfs.append(df)
            print(f"({len(df):,} rows)")

        data = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal samples: {len(data):,}")

        # Sampling
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
            label_col = data.columns[-1]

        print(f"Label column: {label_col}")

        # Label mapping
        y_raw = data[label_col].astype(str)

        def map_label(label):
            label_lower = str(label).lower().strip().replace(' ', '').replace('-', '_')
            # Exact match
            if label_lower in self.ATTACK_MAPPING:
                return self.ATTACK_MAPPING[label_lower]
            # Partial match
            for key, val in self.ATTACK_MAPPING.items():
                if key in label_lower or label_lower in key:
                    return val
            return 'Unknown'

        y_category = y_raw.apply(map_label)

        # Unknown kontrol
        unknown_mask = y_category == 'Unknown'
        if unknown_mask.any():
            unknown_labels = y_raw[unknown_mask].unique()
            print(f"  Warning: {len(unknown_labels)} unknown labels: {list(unknown_labels)[:5]}...")
            # Unknown'ları çıkar
            data = data[~unknown_mask]
            y_category = y_category[~unknown_mask]
            print(f"  Removed {unknown_mask.sum()} unknown samples")

        # Category to ID
        y = y_category.map(self.CATEGORY_TO_ID)

        # Features
        X = data.drop(columns=[label_col])

        # Categorical encoding (varsa)
        X = self._encode_categorical_columns(X)

        # Clean data
        X = self._clean_data(X)

        # Sadece numeric kolonları tut
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        self.feature_names = X.columns.tolist()
        print(f"Features: {len(self.feature_names)}")

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
            'label_encoders': self.label_encoders,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        return X_train, X_test, y_train, y_test, info


# =============================================================================
# EDGE-IIoT LOADER
# =============================================================================

class EdgeIIoTLoader(BaseDatasetLoader):
    """
    Edge-IIoTset Dataset Loader

    - 61 features (bazıları categorical)
    - 15 sınıf: Normal + 14 attack types
    """

    # Sadece gerçek high-cardinality string kolonlar çıkarılacak
    DROP_COLUMNS = [
        'frame.time',           # Timestamp string
        'ip.src_host',          # IP adresi
        'ip.dst_host',          # IP adresi
        'arp.src.proto_ipv4',   # IP adresi
        'arp.dst.proto_ipv4',   # IP adresi
        'Attack_label'          # Binary label (Attack_type kullanacağız)
    ]

    ATTACK_MAPPING = {
        'Normal': 0,
        'DDoS_UDP': 1, 'DDoS_ICMP': 2, 'DDoS_HTTP': 3, 'DDoS_TCP': 4,
        'SQL_injection': 5, 'Vulnerability_scanner': 6, 'Password': 7,
        'Uploading': 8, 'Backdoor': 9, 'Port_Scanning': 10,
        'XSS': 11, 'Ransomware': 12, 'Fingerprinting': 13, 'MITM': 14
    }

    def load(
        self,
        test_size: float = 0.2,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Edge-IIoTset verisini yükle"""
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

        # Label mapping
        label_to_id = {}
        for label in y_raw.unique():
            if label in self.ATTACK_MAPPING:
                label_to_id[label] = self.ATTACK_MAPPING[label]
            else:
                # Fuzzy match
                matched = False
                for known in self.ATTACK_MAPPING:
                    if known.lower() in label.lower() or label.lower() in known.lower():
                        label_to_id[label] = self.ATTACK_MAPPING[known]
                        matched = True
                        break
                if not matched:
                    print(f"  Warning: Unknown label '{label}', assigning new ID")
                    label_to_id[label] = len(label_to_id)

        y = y_raw.map(label_to_id)

        # Features
        X = data.drop(columns=[label_col])

        # Drop high-cardinality columns
        cols_to_drop = [c for c in self.DROP_COLUMNS if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            print(f"  Dropped {len(cols_to_drop)} high-cardinality columns")

        # Categorical encoding
        X = self._encode_categorical_columns(X)

        # Clean data
        X = self._clean_data(X)

        # Numeric only
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        self.feature_names = X.columns.tolist()
        print(f"Features: {len(self.feature_names)}")

        # Class distribution
        print("\nClass distribution:")
        for label, idx in sorted(label_to_id.items(), key=lambda x: x[1]):
            count = (y == idx).sum()
            pct = count / len(y) * 100
            print(f"  {idx} - {label}: {count:,} ({pct:.1f}%)")

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
            'dataset': 'Edge-IIoTset',
            'num_features': len(self.feature_names),
            'num_classes': len(label_to_id),
            'feature_names': self.feature_names,
            'class_names': list(label_to_id.keys()),
            'label_encoders': self.label_encoders,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

        return X_train, X_test, y_train, y_test, info


# =============================================================================
# ToN-IoT LOADER
# =============================================================================

class ToNIoTLoader(BaseDatasetLoader):
    """
    ToN-IoT Dataset Loader

    - 44 raw features -> ~30 usable features (after encoding)
    - 10 sınıf: normal + 9 attack types
    - Kaynak: Processed_Network_dataset/ (23 CSV)
    """

    # Sadece gerçek high-cardinality string kolonlar çıkarılacak
    DROP_COLUMNS = [
        # IP adresleri - milyonlarca unique
        'src_ip', 'dst_ip',

        # High-cardinality string alanlar
        'dns_query',            # DNS sorgusu
        'ssl_subject',          # SSL sertifika
        'ssl_issuer',           # SSL sertifika
        'http_uri',             # URL path
        'http_referrer',        # Referrer URL
        'http_user_agent',      # User agent
        'http_orig_mime_types', # MIME type
        'http_resp_mime_types', # MIME type
        'weird_name',           # Zeek weird
        'weird_addl',           # Zeek weird
        'weird_notice',         # Zeek weird

        # Binary label (type kullanacağız)
        'label'
    ]

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

    def load(
        self,
        test_size: float = 0.2,
        normalize: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """ToN-IoT verisini yükle"""
        print("="*60)
        print("Loading ToN-IoT Dataset (Full Features)")
        print("="*60)

        # CSV dosyalarını bul
        csv_files = []
        search_paths = [
            self.data_path,
            os.path.join(self.data_path, 'Processed_datasets', 'Processed_Network_dataset'),
            os.path.join(self.data_path, 'Processed_Network_dataset'),
        ]

        for path in search_paths:
            if os.path.exists(path):
                found = glob.glob(os.path.join(path, '*.csv'))
                if found:
                    csv_files = found
                    print(f"Loading from: {path}")
                    break

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_path}")

        print(f"Found {len(csv_files)} CSV files")

        # CSV'leri yükle
        dfs = []
        for csv_file in sorted(csv_files):
            filename = os.path.basename(csv_file)
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                dfs.append(df)
                print(f"  Loading {filename}... ({len(df):,} rows)")
            except Exception as e:
                print(f"  Warning: Could not load {filename}: {e}")

        data = pd.concat(dfs, ignore_index=True)
        print(f"\nTotal samples: {len(data):,}")

        # Sampling
        if self.max_samples and len(data) > self.max_samples:
            data = data.sample(n=self.max_samples, random_state=self.random_state)
            print(f"Limited to: {len(data):,}")

        # Label column
        label_col = 'type'
        if label_col not in data.columns:
            for col in ['label', 'Label', 'attack', 'Attack']:
                if col in data.columns:
                    label_col = col
                    break

        print(f"Label column: {label_col}")

        # Labels
        y_raw = data[label_col].astype(str).str.lower().str.strip()

        # Label mapping
        label_to_id = {}
        for label in y_raw.unique():
            if label in self.ATTACK_MAPPING:
                label_to_id[label] = self.ATTACK_MAPPING[label]
            else:
                print(f"  Warning: Unknown label '{label}'")
                label_to_id[label] = len(label_to_id)

        y = y_raw.map(label_to_id)

        # Features
        X = data.drop(columns=[label_col])

        # Drop high-cardinality columns
        cols_to_drop = [c for c in self.DROP_COLUMNS if c in X.columns]
        if cols_to_drop:
            X = X.drop(columns=cols_to_drop)
            print(f"  Dropped {len(cols_to_drop)} high-cardinality columns")

        # ============================================
        # CATEGORICAL ENCODING - TÜM OBJECT KOLONLAR
        # ============================================
        X = self._encode_categorical_columns(X)

        # Clean data
        X = self._clean_data(X)

        # Numeric only (güvenlik için)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]

        self.feature_names = X.columns.tolist()
        print(f"\nFinal feature count: {len(self.feature_names)}")
        print(f"Features: {self.feature_names}")

        # Class distribution
        print("\nClass distribution:")
        for label, idx in sorted(label_to_id.items(), key=lambda x: x[1]):
            count = (y == idx).sum()
            pct = count / len(y) * 100
            print(f"  {idx} - {label}: {count:,} ({pct:.1f}%)")

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
            'dataset': 'ToN-IoT',
            'num_features': len(self.feature_names),
            'num_classes': len(label_to_id),
            'feature_names': self.feature_names,
            'class_names': list(label_to_id.keys()),
            'label_encoders': self.label_encoders,
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
                    max_samples=10000  # Test için küçük sample
                )
                print(f"\n✓ {name} loaded successfully!")
                print(f"  Features: {info['num_features']}")
                print(f"  Classes: {info['num_classes']}")
                print(f"  Encoded columns: {list(info.get('label_encoders', {}).keys())}")
            except Exception as e:
                print(f"✗ Error loading {name}: {e}")
        else:
            print(f"✗ Path not found: {path}")