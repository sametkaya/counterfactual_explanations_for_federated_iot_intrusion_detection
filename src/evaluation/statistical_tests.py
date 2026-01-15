"""
Statistical Analysis & CF Quality Metrics
==========================================
Experiment sonuçları için istatistiksel testler ve CF kalite metrikleri
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings


# =============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# =============================================================================

@dataclass
class StatTestResult:
    """İstatistiksel test sonucu"""
    test_name: str
    statistic: float
    p_value: float
    significant: bool  # p < 0.05
    effect_size: float
    effect_interpretation: str  # "small", "medium", "large"
    confidence_interval: Tuple[float, float]
    sample_sizes: Tuple[int, int]


class StatisticalTests:
    """İstatistiksel anlamlılık testleri"""
    
    @staticmethod
    def wilcoxon_signed_rank(
        x: np.ndarray, 
        y: np.ndarray, 
        alpha: float = 0.05
    ) -> StatTestResult:
        """
        Wilcoxon Signed-Rank Test (paired samples, non-parametric)
        
        Use when: Comparing two related samples (e.g., before/after)
        """
        # Remove pairs where difference is zero
        diff = x - y
        non_zero_mask = diff != 0
        x_clean = x[non_zero_mask]
        y_clean = y[non_zero_mask]
        
        if len(x_clean) < 10:
            warnings.warn("Sample size too small for reliable Wilcoxon test")
        
        stat, p_value = stats.wilcoxon(x_clean, y_clean)
        
        # Effect size: r = Z / sqrt(N)
        n = len(x_clean)
        z = stats.norm.ppf(p_value / 2)
        effect_size = abs(z) / np.sqrt(n) if n > 0 else 0
        
        # Interpret effect size (Cohen's guidelines for r)
        if effect_size < 0.1:
            interpretation = "negligible"
        elif effect_size < 0.3:
            interpretation = "small"
        elif effect_size < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        # Bootstrap CI for median difference
        ci = StatisticalTests._bootstrap_ci(x - y)
        
        return StatTestResult(
            test_name="Wilcoxon Signed-Rank",
            statistic=stat,
            p_value=p_value,
            significant=p_value < alpha,
            effect_size=effect_size,
            effect_interpretation=interpretation,
            confidence_interval=ci,
            sample_sizes=(len(x), len(y))
        )
    
    @staticmethod
    def mann_whitney_u(
        x: np.ndarray, 
        y: np.ndarray, 
        alpha: float = 0.05
    ) -> StatTestResult:
        """
        Mann-Whitney U Test (independent samples, non-parametric)
        
        Use when: Comparing two independent samples
        """
        stat, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')
        
        # Effect size: r = Z / sqrt(N)
        n1, n2 = len(x), len(y)
        n = n1 + n2
        z = stats.norm.ppf(p_value / 2)
        effect_size = abs(z) / np.sqrt(n) if n > 0 else 0
        
        if effect_size < 0.1:
            interpretation = "negligible"
        elif effect_size < 0.3:
            interpretation = "small"
        elif effect_size < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        # CI for difference in medians
        ci = StatisticalTests._bootstrap_ci_two_sample(x, y)
        
        return StatTestResult(
            test_name="Mann-Whitney U",
            statistic=stat,
            p_value=p_value,
            significant=p_value < alpha,
            effect_size=effect_size,
            effect_interpretation=interpretation,
            confidence_interval=ci,
            sample_sizes=(n1, n2)
        )
    
    @staticmethod
    def kruskal_wallis(
        *groups: np.ndarray,
        alpha: float = 0.05
    ) -> StatTestResult:
        """
        Kruskal-Wallis H Test (multiple independent samples, non-parametric)
        
        Use when: Comparing 3+ independent groups
        """
        stat, p_value = stats.kruskal(*groups)
        
        # Effect size: eta-squared = (H - k + 1) / (n - k)
        k = len(groups)
        n = sum(len(g) for g in groups)
        eta_squared = (stat - k + 1) / (n - k) if n > k else 0
        
        if eta_squared < 0.01:
            interpretation = "negligible"
        elif eta_squared < 0.06:
            interpretation = "small"
        elif eta_squared < 0.14:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return StatTestResult(
            test_name="Kruskal-Wallis H",
            statistic=stat,
            p_value=p_value,
            significant=p_value < alpha,
            effect_size=eta_squared,
            effect_interpretation=interpretation,
            confidence_interval=(0, 0),  # Not applicable for this test
            sample_sizes=tuple(len(g) for g in groups)
        )
    
    @staticmethod
    def friedman_test(
        *groups: np.ndarray,
        alpha: float = 0.05
    ) -> StatTestResult:
        """
        Friedman Test (repeated measures, non-parametric)
        
        Use when: Comparing 3+ related samples (e.g., same subjects, different conditions)
        """
        stat, p_value = stats.friedmanchisquare(*groups)
        
        # Effect size: Kendall's W
        k = len(groups)
        n = len(groups[0])
        W = stat / (n * (k - 1)) if n > 0 and k > 1 else 0
        
        if W < 0.1:
            interpretation = "negligible"
        elif W < 0.3:
            interpretation = "small"
        elif W < 0.5:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return StatTestResult(
            test_name="Friedman",
            statistic=stat,
            p_value=p_value,
            significant=p_value < alpha,
            effect_size=W,
            effect_interpretation=interpretation,
            confidence_interval=(0, 0),
            sample_sizes=tuple(len(g) for g in groups)
        )
    
    @staticmethod
    def _bootstrap_ci(
        data: np.ndarray, 
        n_bootstrap: int = 10000, 
        ci: float = 0.95
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for median"""
        if len(data) == 0:
            return (0.0, 0.0)
        
        medians = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            medians.append(np.median(sample))
        
        alpha = (1 - ci) / 2
        lower = np.percentile(medians, alpha * 100)
        upper = np.percentile(medians, (1 - alpha) * 100)
        
        return (lower, upper)
    
    @staticmethod
    def _bootstrap_ci_two_sample(
        x: np.ndarray, 
        y: np.ndarray, 
        n_bootstrap: int = 10000, 
        ci: float = 0.95
    ) -> Tuple[float, float]:
        """Bootstrap CI for difference in medians"""
        if len(x) == 0 or len(y) == 0:
            return (0.0, 0.0)
        
        diffs = []
        for _ in range(n_bootstrap):
            x_sample = np.random.choice(x, size=len(x), replace=True)
            y_sample = np.random.choice(y, size=len(y), replace=True)
            diffs.append(np.median(x_sample) - np.median(y_sample))
        
        alpha = (1 - ci) / 2
        lower = np.percentile(diffs, alpha * 100)
        upper = np.percentile(diffs, (1 - alpha) * 100)
        
        return (lower, upper)
    
    @staticmethod
    def cohens_d(x: np.ndarray, y: np.ndarray) -> Tuple[float, str]:
        """
        Cohen's d effect size for two groups
        """
        n1, n2 = len(x), len(y)
        var1, var2 = np.var(x, ddof=1), np.var(y, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        d = (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0
        
        if abs(d) < 0.2:
            interpretation = "negligible"
        elif abs(d) < 0.5:
            interpretation = "small"
        elif abs(d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return d, interpretation


# =============================================================================
# CF QUALITY METRICS
# =============================================================================

@dataclass
class CFQualityMetrics:
    """CF kalite metrikleri"""
    validity: float          # % of CFs reaching target class
    proximity_l1: float      # Mean L1 distance to original
    proximity_l2: float      # Mean L2 distance to original
    sparsity: float          # % of features changed
    diversity: float         # Mean pairwise distance between CFs
    plausibility: float      # How realistic is the CF (based on data distribution)
    actionability: float     # % of changes that are actionable
    stability: float         # Consistency across similar inputs
    
    def to_dict(self) -> Dict:
        return {
            'validity': self.validity,
            'proximity_l1': self.proximity_l1,
            'proximity_l2': self.proximity_l2,
            'sparsity': self.sparsity,
            'diversity': self.diversity,
            'plausibility': self.plausibility,
            'actionability': self.actionability,
            'stability': self.stability
        }


class CFQualityEvaluator:
    """Counterfactual kalite değerlendirici"""
    
    def __init__(
        self,
        X_train: np.ndarray,
        feature_ranges: Optional[Dict[int, Tuple[float, float]]] = None,
        immutable_features: Optional[List[int]] = None,
        n_neighbors: int = 5
    ):
        """
        Args:
            X_train: Training data for plausibility calculation
            feature_ranges: Valid ranges for each feature {idx: (min, max)}
            immutable_features: Indices of features that shouldn't change
            n_neighbors: K for KNN-based plausibility
        """
        self.X_train = X_train
        self.feature_ranges = feature_ranges or {}
        self.immutable_features = immutable_features or []
        self.n_neighbors = n_neighbors
        
        # Fit KNN for plausibility
        from sklearn.neighbors import NearestNeighbors
        self.knn = NearestNeighbors(n_neighbors=n_neighbors)
        self.knn.fit(X_train)
        
        # Calculate training data statistics
        self.train_mean = np.mean(X_train, axis=0)
        self.train_std = np.std(X_train, axis=0) + 1e-8
    
    def evaluate(
        self,
        original: np.ndarray,
        counterfactuals: List[np.ndarray],
        original_pred: int,
        cf_preds: List[int],
        target_class: int
    ) -> CFQualityMetrics:
        """Evaluate CF quality with all metrics"""
        
        if len(counterfactuals) == 0:
            return CFQualityMetrics(
                validity=0.0, proximity_l1=0.0, proximity_l2=0.0,
                sparsity=0.0, diversity=0.0, plausibility=0.0,
                actionability=0.0, stability=0.0
            )
        
        # Convert to numpy
        orig = original.flatten() if hasattr(original, 'flatten') else np.array(original).flatten()
        cfs = [cf.flatten() if hasattr(cf, 'flatten') else np.array(cf).flatten() for cf in counterfactuals]
        
        # 1. Validity
        validity = sum(1 for p in cf_preds if p == target_class) / len(cf_preds)
        
        # 2. Proximity (L1 and L2)
        proximity_l1 = np.mean([np.mean(np.abs(cf - orig)) for cf in cfs])
        proximity_l2 = np.mean([np.sqrt(np.mean((cf - orig) ** 2)) for cf in cfs])
        
        # 3. Sparsity (% of features changed)
        threshold = 0.01
        sparsity = np.mean([
            np.sum(np.abs(cf - orig) > threshold) / len(orig) 
            for cf in cfs
        ])
        
        # 4. Diversity (mean pairwise L1 distance)
        diversity = self._calc_diversity(cfs)
        
        # 5. Plausibility (based on distance to training data)
        plausibility = self._calc_plausibility(cfs)
        
        # 6. Actionability (% of changes that are actionable)
        actionability = self._calc_actionability(orig, cfs)
        
        # 7. Stability (placeholder - would need multiple similar inputs)
        stability = 1.0 - np.std([np.mean(np.abs(cf - orig)) for cf in cfs]) if len(cfs) > 1 else 1.0
        
        return CFQualityMetrics(
            validity=validity,
            proximity_l1=proximity_l1,
            proximity_l2=proximity_l2,
            sparsity=sparsity,
            diversity=diversity,
            plausibility=plausibility,
            actionability=actionability,
            stability=stability
        )
    
    def _calc_diversity(self, cfs: List[np.ndarray]) -> float:
        """Calculate mean pairwise distance between CFs"""
        if len(cfs) < 2:
            return 0.0
        
        dists = []
        for i in range(len(cfs)):
            for j in range(i + 1, len(cfs)):
                dists.append(np.mean(np.abs(cfs[i] - cfs[j])))
        
        return np.mean(dists) if dists else 0.0
    
    def _calc_plausibility(self, cfs: List[np.ndarray]) -> float:
        """
        Calculate plausibility based on distance to training data manifold
        
        Score: 1 / (1 + mean_knn_distance)
        Higher score = CF is closer to real data points
        """
        if len(cfs) == 0:
            return 0.0
        
        scores = []
        for cf in cfs:
            distances, _ = self.knn.kneighbors([cf])
            mean_dist = np.mean(distances)
            # Normalize by training data spread
            normalized_dist = mean_dist / (np.mean(self.train_std) + 1e-8)
            score = 1.0 / (1.0 + normalized_dist)
            scores.append(score)
        
        return np.mean(scores)
    
    def _calc_actionability(self, orig: np.ndarray, cfs: List[np.ndarray]) -> float:
        """
        Calculate actionability score
        
        Penalizes:
        - Changes to immutable features
        - Changes outside valid ranges
        """
        if len(cfs) == 0:
            return 0.0
        
        scores = []
        for cf in cfs:
            n_features = len(orig)
            actionable_changes = 0
            total_changes = 0
            
            for i in range(n_features):
                if abs(cf[i] - orig[i]) > 0.01:  # Feature changed
                    total_changes += 1
                    
                    # Check if change is actionable
                    is_actionable = True
                    
                    # Immutable check
                    if i in self.immutable_features:
                        is_actionable = False
                    
                    # Range check
                    if i in self.feature_ranges:
                        min_val, max_val = self.feature_ranges[i]
                        if cf[i] < min_val or cf[i] > max_val:
                            is_actionable = False
                    
                    if is_actionable:
                        actionable_changes += 1
            
            if total_changes > 0:
                scores.append(actionable_changes / total_changes)
            else:
                scores.append(1.0)  # No changes = fully actionable
        
        return np.mean(scores)


# =============================================================================
# EXPERIMENT RESULT ANALYZER
# =============================================================================

class ExperimentAnalyzer:
    """Tüm deneyler için istatistiksel analiz"""
    
    @staticmethod
    def compare_methods(
        results: Dict[str, List[float]],
        metric_name: str = "validity"
    ) -> Dict[str, any]:
        """
        Compare multiple CF methods statistically
        
        Args:
            results: {method_name: [values]} dictionary
            metric_name: Name of the metric being compared
        
        Returns:
            Comprehensive statistical analysis
        """
        method_names = list(results.keys())
        method_values = [np.array(results[m]) for m in method_names]
        
        analysis = {
            'metric': metric_name,
            'methods': method_names,
            'descriptive': {},
            'pairwise_tests': [],
            'overall_test': None
        }
        
        # Descriptive statistics
        for name, values in zip(method_names, method_values):
            analysis['descriptive'][name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'ci_95': StatisticalTests._bootstrap_ci(values)
            }
        
        # Overall test (Kruskal-Wallis for 3+ groups)
        if len(method_values) >= 3:
            overall = StatisticalTests.kruskal_wallis(*method_values)
            analysis['overall_test'] = {
                'test': overall.test_name,
                'statistic': overall.statistic,
                'p_value': overall.p_value,
                'significant': overall.significant,
                'effect_size': overall.effect_size,
                'effect_interpretation': overall.effect_interpretation
            }
        
        # Pairwise comparisons (Mann-Whitney U)
        for i in range(len(method_names)):
            for j in range(i + 1, len(method_names)):
                test_result = StatisticalTests.mann_whitney_u(
                    method_values[i], 
                    method_values[j]
                )
                analysis['pairwise_tests'].append({
                    'comparison': f"{method_names[i]} vs {method_names[j]}",
                    'test': test_result.test_name,
                    'statistic': test_result.statistic,
                    'p_value': test_result.p_value,
                    'significant': test_result.significant,
                    'effect_size': test_result.effect_size,
                    'effect_interpretation': test_result.effect_interpretation,
                    'ci_95': test_result.confidence_interval
                })
        
        return analysis
    
    @staticmethod
    def format_results_table(analysis: Dict) -> str:
        """Format analysis results as readable table"""
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"Statistical Analysis: {analysis['metric']}")
        lines.append(f"{'='*70}")
        
        # Descriptive stats
        lines.append("\nDescriptive Statistics:")
        lines.append("-" * 70)
        lines.append(f"{'Method':<20} {'Mean':>10} {'Std':>10} {'Median':>10} {'95% CI':>18}")
        lines.append("-" * 70)
        
        for method, stats in analysis['descriptive'].items():
            ci = stats['ci_95']
            lines.append(
                f"{method:<20} {stats['mean']:>10.4f} {stats['std']:>10.4f} "
                f"{stats['median']:>10.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
            )
        
        # Overall test
        if analysis['overall_test']:
            lines.append(f"\nOverall Test: {analysis['overall_test']['test']}")
            lines.append(f"  Statistic: {analysis['overall_test']['statistic']:.4f}")
            lines.append(f"  p-value: {analysis['overall_test']['p_value']:.4e}")
            sig = "YES" if analysis['overall_test']['significant'] else "NO"
            lines.append(f"  Significant (α=0.05): {sig}")
            lines.append(f"  Effect Size: {analysis['overall_test']['effect_size']:.4f} ({analysis['overall_test']['effect_interpretation']})")
        
        # Pairwise comparisons
        if analysis['pairwise_tests']:
            lines.append("\nPairwise Comparisons:")
            lines.append("-" * 70)
            
            for test in analysis['pairwise_tests']:
                sig = "*" if test['significant'] else ""
                lines.append(f"  {test['comparison']}: p={test['p_value']:.4e}{sig}, "
                           f"effect={test['effect_size']:.3f} ({test['effect_interpretation']})")
        
        lines.append("\n" + "="*70)
        return "\n".join(lines)


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("Statistical Tests & CF Quality Metrics - Test")
    print("="*60)
    
    # Test statistical tests
    np.random.seed(42)
    x = np.random.normal(0.7, 0.1, 50)
    y = np.random.normal(0.6, 0.1, 50)
    
    print("\n1. Wilcoxon Test:")
    result = StatisticalTests.wilcoxon_signed_rank(x, y)
    print(f"   p-value: {result.p_value:.4e}")
    print(f"   Significant: {result.significant}")
    print(f"   Effect: {result.effect_size:.3f} ({result.effect_interpretation})")
    
    print("\n2. Mann-Whitney U:")
    result = StatisticalTests.mann_whitney_u(x, y)
    print(f"   p-value: {result.p_value:.4e}")
    print(f"   Significant: {result.significant}")
    
    # Test CF quality metrics
    print("\n3. CF Quality Metrics:")
    X_train = np.random.rand(100, 10)
    evaluator = CFQualityEvaluator(X_train)
    
    orig = np.random.rand(10)
    cfs = [orig + np.random.rand(10) * 0.2 for _ in range(5)]
    
    metrics = evaluator.evaluate(orig, cfs, 0, [1, 1, 0, 1, 1], 1)
    print(f"   Validity: {metrics.validity:.2%}")
    print(f"   Proximity (L1): {metrics.proximity_l1:.4f}")
    print(f"   Plausibility: {metrics.plausibility:.4f}")
    print(f"   Actionability: {metrics.actionability:.2%}")
    
    # Test method comparison
    print("\n4. Method Comparison:")
    results = {
        'DiCE': np.random.normal(0.7, 0.1, 30),
        'Wachter': np.random.normal(0.65, 0.1, 30),
        'FACE': np.random.normal(0.6, 0.15, 30)
    }
    
    analysis = ExperimentAnalyzer.compare_methods(results, "validity")
    print(ExperimentAnalyzer.format_results_table(analysis))
    
    print("\nTest completed!")
