"""
Privacy Attack Evaluation
=========================
Differential Privacy korumalarını test etmek için Membership Inference Attack (MIA)

References:
- Shokri et al. "Membership Inference Attacks Against Machine Learning Models" (2017)
- Carlini et al. "Membership Inference Attacks From First Principles" (2022)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings


@dataclass
class MIAResult:
    """Membership Inference Attack sonucu"""
    attack_accuracy: float      # MIA başarı oranı (0.5 = random guess)
    attack_precision: float
    attack_recall: float
    attack_f1: float
    attack_auc: float
    privacy_leakage: float      # = accuracy - 0.5 (ne kadar bilgi sızdı)
    vulnerability_score: float  # Normalize edilmiş zafiyet skoru [0,1]
    
    def to_dict(self) -> Dict:
        return {
            'attack_accuracy': self.attack_accuracy,
            'attack_precision': self.attack_precision,
            'attack_recall': self.attack_recall,
            'attack_f1': self.attack_f1,
            'attack_auc': self.attack_auc,
            'privacy_leakage': self.privacy_leakage,
            'vulnerability_score': self.vulnerability_score
        }
    
    def is_vulnerable(self, threshold: float = 0.6) -> bool:
        """Model MIA'ya karşı savunmasız mı?"""
        return self.attack_accuracy > threshold


# =============================================================================
# MEMBERSHIP INFERENCE ATTACKS
# =============================================================================

class MembershipInferenceAttack:
    """
    Membership Inference Attack (MIA)
    
    Amaç: Bir veri noktasının modelin training set'inde olup olmadığını tahmin etmek
    
    Attack Types:
    1. Threshold-based: Confidence/loss değerine göre
    2. Shadow Model: Benzer modeller train edip sınıflandırıcı öğrenmek
    3. LiRA (Likelihood Ratio Attack): En güçlü attack
    """
    
    def __init__(
        self,
        target_model: nn.Module,
        device: str = "cuda"
    ):
        self.target_model = target_model
        self.device = device
        self.target_model.eval()
    
    def _get_model_output(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Get model confidence and loss for a batch"""
        x = x.to(self.device)
        
        with torch.no_grad():
            logits = self.target_model(x)
            probs = F.softmax(logits, dim=1)
            confidence = probs.max(dim=1)[0].cpu().numpy()
            
            # Entropy as uncertainty measure
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).cpu().numpy()
        
        return confidence, entropy


class ThresholdMIA(MembershipInferenceAttack):
    """
    Threshold-based Membership Inference Attack
    
    En basit attack: Yüksek confidence = member, düşük confidence = non-member
    """
    
    def attack(
        self,
        member_data: torch.Tensor,      # Training set samples
        non_member_data: torch.Tensor,   # Test/holdout samples
        threshold: float = None
    ) -> MIAResult:
        """
        Execute threshold-based MIA
        
        Args:
            member_data: Training set'ten örnekler
            non_member_data: Training set dışından örnekler
            threshold: Confidence threshold (None = optimal threshold bulunur)
        """
        # Get model outputs
        member_conf, member_entropy = self._get_model_output(member_data)
        non_member_conf, non_member_entropy = self._get_model_output(non_member_data)
        
        # Create labels (1 = member, 0 = non-member)
        y_true = np.concatenate([
            np.ones(len(member_conf)),
            np.zeros(len(non_member_conf))
        ])
        
        # Attack signal (confidence score)
        attack_signal = np.concatenate([member_conf, non_member_conf])
        
        # Find optimal threshold if not provided
        if threshold is None:
            threshold = self._find_optimal_threshold(attack_signal, y_true)
        
        # Predictions
        y_pred = (attack_signal > threshold).astype(int)
        
        # Calculate metrics
        return self._calculate_metrics(y_true, y_pred, attack_signal)
    
    def _find_optimal_threshold(
        self, 
        scores: np.ndarray, 
        labels: np.ndarray
    ) -> float:
        """Find threshold that maximizes attack accuracy"""
        thresholds = np.percentile(scores, np.arange(0, 100, 5))
        best_threshold = 0.5
        best_acc = 0.5
        
        for t in thresholds:
            preds = (scores > t).astype(int)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
        
        return best_threshold
    
    def _calculate_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_scores: np.ndarray
    ) -> MIAResult:
        """Calculate attack performance metrics"""
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.5
        
        # Privacy leakage: how much better than random guess
        privacy_leakage = max(0, accuracy - 0.5)
        
        # Vulnerability score: normalized to [0, 1]
        # 0 = perfectly private, 1 = completely vulnerable
        vulnerability = min(1.0, privacy_leakage * 2)
        
        return MIAResult(
            attack_accuracy=accuracy,
            attack_precision=precision,
            attack_recall=recall,
            attack_f1=f1,
            attack_auc=auc,
            privacy_leakage=privacy_leakage,
            vulnerability_score=vulnerability
        )


class LossMIA(MembershipInferenceAttack):
    """
    Loss-based Membership Inference Attack
    
    Training samples typically have lower loss than non-members
    """
    
    def attack(
        self,
        member_data: torch.Tensor,
        member_labels: torch.Tensor,
        non_member_data: torch.Tensor,
        non_member_labels: torch.Tensor
    ) -> MIAResult:
        """Execute loss-based MIA"""
        
        # Calculate losses
        member_losses = self._get_losses(member_data, member_labels)
        non_member_losses = self._get_losses(non_member_data, non_member_labels)
        
        # Create labels
        y_true = np.concatenate([
            np.ones(len(member_losses)),
            np.zeros(len(non_member_losses))
        ])
        
        # Attack signal (negative loss - lower loss = more likely member)
        attack_signal = -np.concatenate([member_losses, non_member_losses])
        
        # Find threshold
        threshold = self._find_optimal_threshold(attack_signal, y_true)
        y_pred = (attack_signal > threshold).astype(int)
        
        return self._calculate_metrics(y_true, y_pred, attack_signal)
    
    def _get_losses(
        self, 
        x: torch.Tensor, 
        y: torch.Tensor
    ) -> np.ndarray:
        """Calculate per-sample losses"""
        x = x.to(self.device)
        y = y.to(self.device)
        
        losses = []
        batch_size = 256
        
        with torch.no_grad():
            for i in range(0, len(x), batch_size):
                batch_x = x[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                logits = self.target_model(batch_x)
                batch_losses = F.cross_entropy(logits, batch_y, reduction='none')
                losses.extend(batch_losses.cpu().numpy())
        
        return np.array(losses)
    
    def _find_optimal_threshold(self, scores, labels):
        thresholds = np.percentile(scores, np.arange(0, 100, 5))
        best_threshold = np.median(scores)
        best_acc = 0.5
        
        for t in thresholds:
            preds = (scores > t).astype(int)
            acc = accuracy_score(labels, preds)
            if acc > best_acc:
                best_acc = acc
                best_threshold = t
        
        return best_threshold
    
    def _calculate_metrics(self, y_true, y_pred, y_scores):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        try:
            auc = roc_auc_score(y_true, y_scores)
        except:
            auc = 0.5
        
        privacy_leakage = max(0, accuracy - 0.5)
        vulnerability = min(1.0, privacy_leakage * 2)
        
        return MIAResult(
            attack_accuracy=accuracy,
            attack_precision=precision,
            attack_recall=recall,
            attack_f1=f1,
            attack_auc=auc,
            privacy_leakage=privacy_leakage,
            vulnerability_score=vulnerability
        )


# =============================================================================
# COUNTERFACTUAL-SPECIFIC MIA
# =============================================================================

class CounterfactualMIA:
    """
    Counterfactual Explanation'lara yönelik MIA
    
    Hipotez: CF'ler training data hakkında bilgi sızdırabilir
    
    Attack: CF generation sırasında eklenen DP noise'un etkisini ölç
    """
    
    def __init__(
        self,
        cf_generator,  # CF generator instance
        device: str = "cuda"
    ):
        self.cf_generator = cf_generator
        self.device = device
    
    def evaluate_privacy_leakage(
        self,
        member_samples: torch.Tensor,
        non_member_samples: torch.Tensor,
        target_class: int,
        num_cfs: int = 5
    ) -> Dict:
        """
        CF generation üzerinden privacy leakage değerlendir
        
        Returns:
            Privacy metrics and attack results
        """
        # Generate CFs for members
        member_cfs = []
        member_proximities = []
        
        for x in member_samples:
            try:
                result = self.cf_generator.generate(x, target_class, num_cfs)
                if result.counterfactuals:
                    member_cfs.append(result)
                    member_proximities.append(result.proximity)
            except:
                continue
        
        # Generate CFs for non-members
        non_member_cfs = []
        non_member_proximities = []
        
        for x in non_member_samples:
            try:
                result = self.cf_generator.generate(x, target_class, num_cfs)
                if result.counterfactuals:
                    non_member_cfs.append(result)
                    non_member_proximities.append(result.proximity)
            except:
                continue
        
        # Attack: Can we distinguish members from non-members based on CF characteristics?
        if len(member_proximities) == 0 or len(non_member_proximities) == 0:
            return {
                'error': 'Not enough samples',
                'attack_accuracy': 0.5,
                'privacy_leakage': 0.0
            }
        
        # Simple threshold attack on proximity
        all_proximities = np.array(member_proximities + non_member_proximities)
        y_true = np.concatenate([
            np.ones(len(member_proximities)),
            np.zeros(len(non_member_proximities))
        ])
        
        # Members typically have lower proximity (model memorized them)
        threshold = np.median(all_proximities)
        y_pred = (all_proximities < threshold).astype(int)  # Lower proximity = member guess
        
        accuracy = accuracy_score(y_true, y_pred)
        
        return {
            'attack_accuracy': accuracy,
            'privacy_leakage': max(0, accuracy - 0.5),
            'member_mean_proximity': np.mean(member_proximities),
            'non_member_mean_proximity': np.mean(non_member_proximities),
            'n_member_samples': len(member_proximities),
            'n_non_member_samples': len(non_member_proximities)
        }


# =============================================================================
# PRIVACY EVALUATION RUNNER
# =============================================================================

class PrivacyEvaluator:
    """
    Comprehensive privacy evaluation
    
    Runs multiple MIA attacks and reports overall privacy assessment
    """
    
    def __init__(
        self,
        model: nn.Module,
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        device: str = "cuda"
    ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.device = device
    
    def evaluate(
        self,
        n_samples: int = 1000
    ) -> Dict:
        """
        Run comprehensive privacy evaluation
        
        Args:
            n_samples: Number of samples per attack
        
        Returns:
            Dictionary with all attack results
        """
        results = {}
        
        # Sample data
        n_train = min(n_samples, len(self.X_train))
        n_test = min(n_samples, len(self.X_test))
        
        train_idx = np.random.choice(len(self.X_train), n_train, replace=False)
        test_idx = np.random.choice(len(self.X_test), n_test, replace=False)
        
        member_data = self.X_train[train_idx]
        member_labels = self.y_train[train_idx]
        non_member_data = self.X_test[test_idx]
        non_member_labels = self.y_test[test_idx]
        
        # 1. Threshold-based MIA
        threshold_attack = ThresholdMIA(self.model, self.device)
        results['threshold_mia'] = threshold_attack.attack(
            member_data, non_member_data
        ).to_dict()
        
        # 2. Loss-based MIA
        loss_attack = LossMIA(self.model, self.device)
        results['loss_mia'] = loss_attack.attack(
            member_data, member_labels,
            non_member_data, non_member_labels
        ).to_dict()
        
        # Aggregate results
        avg_accuracy = np.mean([
            results['threshold_mia']['attack_accuracy'],
            results['loss_mia']['attack_accuracy']
        ])
        
        max_leakage = max(
            results['threshold_mia']['privacy_leakage'],
            results['loss_mia']['privacy_leakage']
        )
        
        results['summary'] = {
            'average_attack_accuracy': avg_accuracy,
            'max_privacy_leakage': max_leakage,
            'overall_vulnerability': 'HIGH' if avg_accuracy > 0.65 else 
                                    'MEDIUM' if avg_accuracy > 0.55 else 'LOW',
            'dp_recommendation': self._get_dp_recommendation(max_leakage)
        }
        
        return results
    
    def _get_dp_recommendation(self, leakage: float) -> str:
        """Get DP epsilon recommendation based on leakage"""
        if leakage > 0.2:
            return "Strong DP needed (ε ≤ 1.0)"
        elif leakage > 0.1:
            return "Moderate DP recommended (ε ≤ 5.0)"
        elif leakage > 0.05:
            return "Light DP sufficient (ε ≤ 10.0)"
        else:
            return "Model has good privacy properties"
    
    def format_report(self, results: Dict) -> str:
        """Format privacy evaluation results as report"""
        lines = []
        lines.append("\n" + "="*70)
        lines.append("PRIVACY EVALUATION REPORT")
        lines.append("="*70)
        
        # Summary
        lines.append("\nSUMMARY:")
        lines.append("-"*40)
        summary = results['summary']
        lines.append(f"  Overall Vulnerability: {summary['overall_vulnerability']}")
        lines.append(f"  Average Attack Accuracy: {summary['average_attack_accuracy']:.2%}")
        lines.append(f"  Max Privacy Leakage: {summary['max_privacy_leakage']:.2%}")
        lines.append(f"  Recommendation: {summary['dp_recommendation']}")
        
        # Detailed results
        lines.append("\nDETAILED RESULTS:")
        lines.append("-"*40)
        
        for attack_name in ['threshold_mia', 'loss_mia']:
            attack = results[attack_name]
            lines.append(f"\n  {attack_name.upper()}:")
            lines.append(f"    Accuracy: {attack['attack_accuracy']:.2%}")
            lines.append(f"    Precision: {attack['attack_precision']:.2%}")
            lines.append(f"    Recall: {attack['attack_recall']:.2%}")
            lines.append(f"    AUC: {attack['attack_auc']:.4f}")
            lines.append(f"    Privacy Leakage: {attack['privacy_leakage']:.2%}")
        
        lines.append("\n" + "="*70)
        lines.append("INTERPRETATION:")
        lines.append("-"*40)
        lines.append("  - Attack Accuracy > 60%: Model is vulnerable to MIA")
        lines.append("  - Attack Accuracy ≈ 50%: Model has good privacy")
        lines.append("  - Privacy Leakage = Attack Accuracy - 50%")
        lines.append("="*70 + "\n")
        
        return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Privacy Attack Evaluation - Test")
    print("="*60)
    
    # Dummy model
    class DummyModel(nn.Module):
        def __init__(self, n_features, n_classes):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(n_features, 64),
                nn.ReLU(),
                nn.Linear(64, n_classes)
            )
        
        def forward(self, x):
            return self.fc(x)
    
    n_features, n_classes = 10, 3
    model = DummyModel(n_features, n_classes)
    
    # Train model briefly
    X_train = torch.randn(500, n_features)
    y_train = torch.randint(0, n_classes, (500,))
    X_test = torch.randn(200, n_features)
    y_test = torch.randint(0, n_classes, (200,))
    
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    for _ in range(50):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    model.eval()
    
    # Run privacy evaluation
    evaluator = PrivacyEvaluator(
        model, X_train, y_train, X_test, y_test,
        device='cpu'
    )
    
    results = evaluator.evaluate(n_samples=200)
    print(evaluator.format_report(results))
    
    print("Test completed!")
