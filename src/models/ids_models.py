"""
IDS Models
==========
IoT Intrusion Detection için Neural Network modelleri
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class IDSMLP(nn.Module):
    """
    Multi-Layer Perceptron tabanlı IDS
    
    Args:
        input_dim: Feature sayısı (CICIoT2023: 47, Edge-IIoTset: 61)
        hidden_dims: Hidden layer boyutları
        num_classes: Sınıf sayısı (binary: 2, multi-class: 8)
        dropout: Dropout oranı
    """
    
    def __init__(
        self,
        input_dim: int = 47,
        hidden_dims: List[int] = [256, 128, 64],
        num_classes: int = 8,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Layer'ları dinamik oluştur
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
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Softmax probabilities döndür"""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Sınıf tahmini döndür"""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


class IDSLSTM(nn.Module):
    """
    LSTM tabanlı IDS - Sequential/Temporal data için
    
    Args:
        input_dim: Feature sayısı
        hidden_dim: LSTM hidden boyutu
        num_layers: LSTM layer sayısı
        num_classes: Sınıf sayısı
        dropout: Dropout oranı
    """
    
    def __init__(
        self,
        input_dim: int = 47,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 8,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Shape (batch, seq_len, input_dim) veya (batch, input_dim)
        """
        # Eğer 2D ise, sequence boyutu ekle
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, input_dim)
        
        lstm_out, _ = self.lstm(x)
        
        # Son timestep'i al
        last_output = lstm_out[:, -1, :]
        
        return self.classifier(last_output)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)


class IDSTransformer(nn.Module):
    """
    Transformer tabanlı IDS
    
    Args:
        input_dim: Feature sayısı
        d_model: Transformer model boyutu
        nhead: Attention head sayısı
        num_layers: Transformer layer sayısı
        num_classes: Sınıf sayısı
        dropout: Dropout oranı
    """
    
    def __init__(
        self,
        input_dim: int = 47,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        num_classes: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Shape (batch, seq_len, input_dim) veya (batch, input_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size = x.size(0)
        
        # Project to d_model
        x = self.input_projection(x)
        
        # CLS token ekle
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer
        x = self.transformer(x)
        
        # CLS token output'unu al
        cls_output = x[:, 0, :]
        
        return self.classifier(cls_output)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=-1)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Attention weights döndür (XAI için)"""
        # Not: Bu simplified version, full attention extraction için
        # hook kullanmak gerekir
        pass


def create_model(
    model_type: str = "mlp",
    input_dim: int = 47,
    num_classes: int = 8,
    **kwargs
) -> nn.Module:
    """
    Model factory
    
    Args:
        model_type: "mlp", "lstm", "transformer"
        input_dim: Feature sayısı
        num_classes: Sınıf sayısı
    """
    models = {
        "mlp": IDSMLP,
        "lstm": IDSLSTM,
        "transformer": IDSTransformer
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    
    return models[model_type](input_dim=input_dim, num_classes=num_classes, **kwargs)


# Test
if __name__ == "__main__":
    # Test all models
    batch_size = 32
    input_dim = 47
    num_classes = 8
    
    x = torch.randn(batch_size, input_dim)
    
    print("Testing IDSMLP...")
    model_mlp = IDSMLP(input_dim=input_dim, num_classes=num_classes)
    out = model_mlp(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    print("\nTesting IDSLSTM...")
    model_lstm = IDSLSTM(input_dim=input_dim, num_classes=num_classes)
    out = model_lstm(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    print("\nTesting IDSTransformer...")
    model_transformer = IDSTransformer(input_dim=input_dim, num_classes=num_classes)
    out = model_transformer(x)
    print(f"  Input: {x.shape}, Output: {out.shape}")
    
    print("\nAll models working correctly!")
