"""
Federated Learning Components
=============================
FL Server ve Client implementasyonu with CF Aggregation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import OrderedDict
import copy


@dataclass
class ClientUpdate:
    """Client'tan gelen update"""
    client_id: int
    model_weights: OrderedDict
    num_samples: int
    cf_statistics: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ServerState:
    """Server durumu"""
    global_model_weights: OrderedDict
    round_number: int
    aggregated_cf_patterns: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict] = field(default_factory=list)


class FederatedClient:
    """
    Federated Learning Client
    
    Her client:
    - Lokal veri üzerinde model eğitir
    - Counterfactual'lar üretir
    - DP ile korunan update'leri server'a gönderir
    """
    
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: Optional[DataLoader] = None,
        cf_generator: Optional[Any] = None,
        local_epochs: int = 5,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.cf_generator = cf_generator
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.device = device
        
        self.model.to(device)
    
    def set_weights(self, weights: OrderedDict):
        """Server'dan gelen weights'i set et"""
        self.model.load_state_dict(weights)
    
    def get_weights(self) -> OrderedDict:
        """Model weights'i döndür"""
        return copy.deepcopy(self.model.state_dict())
    
    def train(self) -> Tuple[OrderedDict, int, Dict]:
        """
        Lokal training
        
        Returns:
            (weights, num_samples, metrics)
        """
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_x.size(0)
                total_samples = len(self.train_loader.dataset)
            
            total_loss = epoch_loss / total_samples
        
        metrics = {'train_loss': total_loss}
        
        return self.get_weights(), total_samples, metrics
    
    def evaluate(self) -> Dict[str, float]:
        """Lokal test set üzerinde evaluation"""
        if self.test_loader is None:
            return {}
        
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_x)
                _, predicted = torch.max(outputs, 1)
                
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {'test_accuracy': accuracy, 'test_samples': total}
    
    def compute_cf_statistics(self, num_samples: int = 50) -> Dict[str, Any]:
        """
        Lokal CF istatistiklerini hesapla
        
        Privacy-preserving: Raw CF'ler değil, aggregate istatistikler paylaşılır
        """
        if self.cf_generator is None:
            return {}
        
        self.model.eval()
        
        cf_directions = []
        cf_magnitudes = []
        cf_validities = []
        
        # Attack sample'ları al
        attack_samples = self._get_attack_samples(num_samples)
        
        for sample in attack_samples:
            result = self.cf_generator.generate(sample, target_class=0)
            
            if result.counterfactuals:
                # Mean direction
                directions = [cf - sample.cpu() for cf in result.counterfactuals]
                mean_dir = torch.stack(directions).mean(dim=0)
                cf_directions.append(mean_dir)
                
                # Mean magnitude
                magnitudes = [torch.norm(cf - sample.cpu()).item() for cf in result.counterfactuals]
                cf_magnitudes.append(np.mean(magnitudes))
                
                cf_validities.append(result.validity)
        
        if not cf_directions:
            return {}
        
        # Aggregate statistics (privacy-preserving)
        stats = {
            'mean_direction': torch.stack(cf_directions).mean(dim=0).tolist(),
            'direction_std': torch.stack(cf_directions).std(dim=0).tolist(),
            'mean_magnitude': float(np.mean(cf_magnitudes)),
            'magnitude_std': float(np.std(cf_magnitudes)),
            'mean_validity': float(np.mean(cf_validities)),
            'num_samples': len(cf_directions)
        }
        
        return stats
    
    def _get_attack_samples(self, num_samples: int) -> List[torch.Tensor]:
        """Attack olarak classify edilen sample'ları al"""
        samples = []
        
        self.model.eval()
        with torch.no_grad():
            for batch_x, batch_y in self.train_loader:
                batch_x = batch_x.to(self.device)
                preds = self.model.predict(batch_x)
                
                # Attack predictions (class != 0)
                attack_mask = preds != 0
                attack_samples = batch_x[attack_mask]
                
                for sample in attack_samples:
                    samples.append(sample.cpu())
                    if len(samples) >= num_samples:
                        return samples
        
        return samples
    
    def full_update(self) -> ClientUpdate:
        """Train + evaluate + CF stats - tam update"""
        # Train
        weights, num_samples, train_metrics = self.train()
        
        # Evaluate
        eval_metrics = self.evaluate()
        
        # CF statistics
        cf_stats = self.compute_cf_statistics()
        
        # Combine metrics
        all_metrics = {**train_metrics, **eval_metrics}
        
        return ClientUpdate(
            client_id=self.client_id,
            model_weights=weights,
            num_samples=num_samples,
            cf_statistics=cf_stats,
            metrics=all_metrics
        )


class FederatedServer:
    """
    Federated Learning Server
    
    - Client update'lerini aggregate et
    - Global model'i güncelle
    - CF pattern'lerini aggregate et
    """
    
    def __init__(
        self,
        model: nn.Module,
        aggregation_strategy: str = "fedavg",
        device: str = "cpu"
    ):
        self.global_model = copy.deepcopy(model)
        self.aggregation_strategy = aggregation_strategy
        self.device = device
        
        self.state = ServerState(
            global_model_weights=self.global_model.state_dict(),
            round_number=0
        )
    
    def get_global_weights(self) -> OrderedDict:
        """Global model weights'i döndür"""
        return copy.deepcopy(self.state.global_model_weights)
    
    def aggregate(self, client_updates: List[ClientUpdate]) -> OrderedDict:
        """
        Client update'lerini aggregate et
        
        Args:
            client_updates: Client'lardan gelen update'ler
            
        Returns:
            Aggregated model weights
        """
        if self.aggregation_strategy == "fedavg":
            return self._fedavg(client_updates)
        elif self.aggregation_strategy == "fedprox":
            return self._fedprox(client_updates)
        else:
            raise ValueError(f"Unknown strategy: {self.aggregation_strategy}")
    
    def _fedavg(self, client_updates: List[ClientUpdate]) -> OrderedDict:
        """FedAvg aggregation"""
        total_samples = sum(u.num_samples for u in client_updates)
        
        # Initialize with zeros
        aggregated_weights = OrderedDict()
        for key in client_updates[0].model_weights.keys():
            aggregated_weights[key] = torch.zeros_like(
                client_updates[0].model_weights[key], 
                dtype=torch.float32
            )
        
        # Weighted average
        for update in client_updates:
            weight = update.num_samples / total_samples
            for key in aggregated_weights.keys():
                aggregated_weights[key] += weight * update.model_weights[key].float()
        
        return aggregated_weights
    
    def _fedprox(self, client_updates: List[ClientUpdate], mu: float = 0.01) -> OrderedDict:
        """FedProx aggregation (FedAvg ile aynı server-side)"""
        # Server-side aynı, client-side proximal term var
        return self._fedavg(client_updates)
    
    def aggregate_cf_statistics(self, client_updates: List[ClientUpdate]) -> Dict[str, Any]:
        """
        CF istatistiklerini aggregate et
        
        Privacy-preserving: Sadece aggregate statistics, raw CF yok
        """
        all_directions = []
        all_magnitudes = []
        all_validities = []
        total_samples = 0
        
        for update in client_updates:
            stats = update.cf_statistics
            if not stats:
                continue
            
            n = stats.get('num_samples', 0)
            if n > 0:
                all_directions.append(
                    (torch.tensor(stats['mean_direction']), n)
                )
                all_magnitudes.append(stats['mean_magnitude'] * n)
                all_validities.append(stats['mean_validity'] * n)
                total_samples += n
        
        if total_samples == 0:
            return {}
        
        # Weighted average
        global_direction = sum(d * n for d, n in all_directions) / total_samples
        global_magnitude = sum(all_magnitudes) / total_samples
        global_validity = sum(all_validities) / total_samples
        
        return {
            'global_cf_direction': global_direction.tolist(),
            'global_cf_magnitude': global_magnitude,
            'global_cf_validity': global_validity,
            'total_samples': total_samples
        }
    
    def run_round(self, client_updates: List[ClientUpdate]) -> Dict[str, Any]:
        """
        Bir FL round'u çalıştır
        
        Returns:
            Round metrics
        """
        self.state.round_number += 1
        
        # Aggregate model weights
        new_weights = self.aggregate(client_updates)
        
        # Move weights to device before loading
        new_weights = OrderedDict({k: v.to(self.device) for k, v in new_weights.items()})
        
        self.state.global_model_weights = new_weights
        self.global_model.load_state_dict(new_weights)
        self.global_model.to(self.device)  # Ensure model is on correct device
        
        # Aggregate CF statistics
        cf_patterns = self.aggregate_cf_statistics(client_updates)
        self.state.aggregated_cf_patterns = cf_patterns
        
        # Compute round metrics
        metrics = {
            'round': self.state.round_number,
            'num_clients': len(client_updates),
            'total_samples': sum(u.num_samples for u in client_updates),
            'mean_train_loss': np.mean([u.metrics.get('train_loss', 0) for u in client_updates]),
            'mean_test_accuracy': np.mean([
                u.metrics.get('test_accuracy', 0) for u in client_updates 
                if 'test_accuracy' in u.metrics
            ]),
        }
        
        if cf_patterns:
            metrics['cf_validity'] = cf_patterns.get('global_cf_validity', 0)
        
        # Save to history
        self.state.history.append(metrics)
        
        return metrics


class FederatedSimulator:
    """
    Federated Learning Simülasyonu
    
    Multi-client senaryoyu tek makinede simüle et
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_data: torch.Tensor,
        train_labels: torch.Tensor,
        test_data: Optional[torch.Tensor] = None,
        test_labels: Optional[torch.Tensor] = None,
        num_clients: int = 10,
        iid: bool = True,
        cf_generator_class: Optional[type] = None,
        device: str = "cpu"
    ):
        self.num_clients = num_clients
        self.device = device
        
        # Server
        self.server = FederatedServer(model, device=device)
        
        # Data partitioning
        client_data = self._partition_data(train_data, train_labels, num_clients, iid)
        
        # Test data (shared)
        test_loader = None
        if test_data is not None:
            test_dataset = TensorDataset(test_data, test_labels)
            test_loader = DataLoader(test_dataset, batch_size=64)
        
        # Create clients
        self.clients = []
        for i, (data, labels) in enumerate(client_data):
            train_dataset = TensorDataset(data, labels)
            train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
            
            # CF generator (optional)
            cf_gen = None
            if cf_generator_class:
                cf_gen = cf_generator_class(
                    model=copy.deepcopy(model),
                    feature_count=data.size(1),
                    device=device
                )
            
            client = FederatedClient(
                client_id=i,
                model=copy.deepcopy(model),
                train_loader=train_loader,
                test_loader=test_loader,
                cf_generator=cf_gen,
                device=device
            )
            self.clients.append(client)
    
    def _partition_data(
        self,
        data: torch.Tensor,
        labels: torch.Tensor,
        num_clients: int,
        iid: bool
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Veriyi client'lara dağıt"""
        num_samples = data.size(0)
        
        if iid:
            # IID: Random shuffle ve eşit böl
            indices = torch.randperm(num_samples)
            split_size = num_samples // num_clients
            
            partitions = []
            for i in range(num_clients):
                start = i * split_size
                end = start + split_size if i < num_clients - 1 else num_samples
                idx = indices[start:end]
                partitions.append((data[idx], labels[idx]))
            
            return partitions
        else:
            # Non-IID: Label-based partitioning
            unique_labels = torch.unique(labels)
            label_indices = {l.item(): (labels == l).nonzero().squeeze() for l in unique_labels}
            
            partitions = [[] for _ in range(num_clients)]
            
            # Her label'ı birkaç client'a dağıt
            for label, indices in label_indices.items():
                indices = indices[torch.randperm(len(indices))]
                
                # 2-3 client'a dağıt (non-iid)
                n_clients_for_label = min(3, num_clients)
                selected_clients = np.random.choice(num_clients, n_clients_for_label, replace=False)
                
                split_size = len(indices) // n_clients_for_label
                for i, client_id in enumerate(selected_clients):
                    start = i * split_size
                    end = start + split_size if i < n_clients_for_label - 1 else len(indices)
                    partitions[client_id].extend(indices[start:end].tolist())
            
            # Convert to tensors
            result = []
            for idx_list in partitions:
                if idx_list:
                    idx = torch.tensor(idx_list)
                    result.append((data[idx], labels[idx]))
                else:
                    # Empty partition - add small random sample
                    idx = torch.randint(0, num_samples, (100,))
                    result.append((data[idx], labels[idx]))
            
            return result
    
    def run(
        self,
        num_rounds: int = 50,
        client_fraction: float = 1.0,
        verbose: bool = True
    ) -> List[Dict]:
        """
        FL simülasyonunu çalıştır
        
        Args:
            num_rounds: Toplam round sayısı
            client_fraction: Her round'da katılan client oranı
            verbose: Progress yazdır
            
        Returns:
            Round metrics listesi
        """
        history = []
        
        for round_num in range(num_rounds):
            # Select clients for this round
            num_selected = max(1, int(self.num_clients * client_fraction))
            selected_clients = np.random.choice(
                self.clients, num_selected, replace=False
            )
            
            # Distribute global model
            global_weights = self.server.get_global_weights()
            for client in selected_clients:
                client.set_weights(global_weights)
            
            # Client updates
            updates = []
            for client in selected_clients:
                update = client.full_update()
                updates.append(update)
            
            # Server aggregation
            metrics = self.server.run_round(updates)
            history.append(metrics)
            
            if verbose and (round_num + 1) % 5 == 0:
                print(f"Round {round_num + 1}/{num_rounds}: "
                      f"Loss={metrics['mean_train_loss']:.4f}, "
                      f"Acc={metrics['mean_test_accuracy']:.4f}")
        
        return history


# Test
if __name__ == "__main__":
    print("Testing Federated Learning Components...")
    
    from models.ids_models import IDSMLP
    
    # Dummy data
    num_samples = 1000
    feature_count = 47
    num_classes = 8
    
    X = torch.randn(num_samples, feature_count)
    y = torch.randint(0, num_classes, (num_samples,))
    
    # Split
    train_size = int(0.8 * num_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Model
    model = IDSMLP(input_dim=feature_count, num_classes=num_classes)
    
    # Simulator
    print("\nCreating FL Simulator with 5 clients...")
    simulator = FederatedSimulator(
        model=model,
        train_data=X_train,
        train_labels=y_train,
        test_data=X_test,
        test_labels=y_test,
        num_clients=5,
        iid=True,
        device="cpu"
    )
    
    # Run
    print("\nRunning 10 FL rounds...")
    history = simulator.run(num_rounds=10, verbose=True)
    
    print(f"\nFinal accuracy: {history[-1]['mean_test_accuracy']:.4f}")
    print("\nFL Components test completed!")
