#!/usr/bin/env python3
"""
Comprehensive evaluation script for clustering results.
Calculates all metrics and generates evaluation report.

Usage:
    python scripts/run_evaluation.py --features path/to/features.csv --labels path/to/labels.csv
    python scripts/run_evaluation.py --features outputs/csv/temporal_features/*.csv --labels outputs/csv/clusters/*.csv
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.logger import logger
from src.utils.file_handler import FileHandler


class ClusteringEvaluator:
    """
    Comprehensive clustering evaluation with multiple metrics and visualizations.
    """
    
    def __init__(self, output_dir: Path = Path("outputs/evaluation")):
        """
        Initialize evaluator.
        
        Args:
            output_dir: Directory for evaluation outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.file_handler = FileHandler(output_dir.parent)
        self.results: Dict[str, Any] = {}
        
    def evaluate(self, 
                 X: np.ndarray, 
                 labels: np.ndarray,
                 feature_names: Optional[List[str]] = None,
                 ground_truth: Optional[np.ndarray] = None,
                 run_stability: bool = True,
                 n_folds: int = 5) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            labels: Cluster labels
            feature_names: Names of features
            ground_truth: Ground truth labels (optional)
            run_stability: Whether to run stability analysis
            n_folds: Number of folds for stability analysis
            
        Returns:
            Dictionary with all evaluation results
        """
        logger.info("Starting comprehensive clustering evaluation...")
        
        # Filter noise points
        mask = labels != -1
        X_valid = X[mask]
        labels_valid = labels[mask]
        n_noise = np.sum(~mask)
        noise_ratio = n_noise / len(labels) if len(labels) > 0 else 0
        
        n_clusters = len(set(labels_valid))
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': noise_ratio,
            },
            'internal_metrics': {},
            'external_metrics': {},
            'stability_analysis': {},
            'cluster_analysis': {},
            'feature_importance': {},
        }
        
        # Internal metrics (don't need ground truth)
        if n_clusters > 1 and len(X_valid) > 1:
            logger.info("Calculating internal metrics...")
            results['internal_metrics'] = self._calculate_internal_metrics(
                X_valid, labels_valid
            )
        
        # External metrics (if ground truth available)
        if ground_truth is not None and n_clusters > 1:
            logger.info("Calculating external metrics against ground truth...")
            gt_valid = ground_truth[mask]
            results['external_metrics'] = self._calculate_external_metrics(
                labels_valid, gt_valid
            )
        
        # Stability analysis
        if run_stability and n_clusters > 1:
            logger.info("Running stability analysis...")
            results['stability_analysis'] = self._analyze_stability(
                X_valid, n_folds
            )
        
        # Cluster analysis
        if n_clusters > 0:
            logger.info("Analyzing cluster characteristics...")
            results['cluster_analysis'] = self._analyze_clusters(
                X_valid, labels_valid, feature_names
            )
        
        # Feature importance for clustering
        if feature_names is not None and n_clusters > 1:
            logger.info("Calculating feature importance...")
            results['feature_importance'] = self._calculate_feature_importance(
                X_valid, labels_valid, feature_names
            )
        
        # Overall quality rating
        results['overall_quality'] = self._rate_quality(results)
        
        # Save results
        self.results = results
        self._save_results(results)
        
        logger.info("Evaluation complete!")
        self._print_summary(results)
        
        return results
    
    def _calculate_internal_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate internal clustering metrics.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            
        Returns:
            Dictionary of internal metrics
        """
        metrics = {}
        
        try:
            metrics['silhouette_score'] = silhouette_score(X, labels)  # type: ignore
        except Exception as e:
            logger.warning(f"Silhouette calculation failed: {e}")
            metrics['silhouette_score'] = -1.0
        
        try:
            metrics['davies_bouldin_index'] = davies_bouldin_score(X, labels)  # type: ignore
        except Exception as e:
            logger.warning(f"Davies-Bouldin calculation failed: {e}")
            metrics['davies_bouldin_index'] = float('inf')
        
        try:
            metrics['calinski_harabasz_index'] = calinski_harabasz_score(X, labels)  # type: ignore
        except Exception as e:
            logger.warning(f"Calinski-Harabasz calculation failed: {e}")
            metrics['calinski_harabasz_index'] = -1.0
        
        # Intra-cluster distance (cohesion)
        unique_labels = np.unique(labels)
        intra_distances = []
        for label in unique_labels:
            mask = labels == label
            cluster_points = X[mask]
            if len(cluster_points) > 1:
                center = cluster_points.mean(axis=0)
                distances = cdist(cluster_points, center.reshape(1, -1))
                intra_distances.append(np.mean(distances))
        
        metrics['avg_intra_cluster_distance'] = float(np.mean(intra_distances)) if intra_distances else 0.0
        
        # Inter-cluster distance (separation)
        if len(unique_labels) > 1:
            centers = np.array([X[labels == l].mean(axis=0) for l in unique_labels])
            inter_distances = cdist(centers, centers)
            np.fill_diagonal(inter_distances, np.inf)
            metrics['avg_inter_cluster_distance'] = float(np.mean(inter_distances[inter_distances != np.inf]))
        else:
            metrics['avg_inter_cluster_distance'] = 0.0
        
        # Dunn index approximation
        if metrics['avg_intra_cluster_distance'] > 0:
            metrics['dunn_index_approx'] = (
                metrics['avg_inter_cluster_distance'] / metrics['avg_intra_cluster_distance']
            )
        else:
            metrics['dunn_index_approx'] = 0.0
        
        return metrics
    
    def _calculate_external_metrics(self, 
                                   labels: np.ndarray, 
                                   ground_truth: np.ndarray) -> Dict[str, float]:
        """
        Calculate external validation metrics against ground truth.
        
        Args:
            labels: Predicted cluster labels
            ground_truth: True labels
            
        Returns:
            Dictionary of external metrics
        """
        metrics = {}
        
        try:
            metrics['adjusted_rand_index'] = adjusted_rand_score(ground_truth, labels)
        except Exception:
            metrics['adjusted_rand_index'] = 0.0
        
        try:
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(ground_truth, labels)
        except Exception:
            metrics['normalized_mutual_info'] = 0.0
        
        try:
            metrics['homogeneity'] = homogeneity_score(ground_truth, labels)
        except Exception:
            metrics['homogeneity'] = 0.0
        
        try:
            metrics['completeness'] = completeness_score(ground_truth, labels)
        except Exception:
            metrics['completeness'] = 0.0
        
        try:
            metrics['v_measure'] = v_measure_score(ground_truth, labels)
        except Exception:
            metrics['v_measure'] = 0.0
        
        return metrics
    
    def _analyze_stability(self, X: np.ndarray, n_folds: int = 5) -> Dict[str, Any]:
        """
        Analyze clustering stability using cross-validation.
        
        Args:
            X: Feature matrix
            n_folds: Number of folds
            
        Returns:
            Dictionary with stability metrics
        """
        from sklearn.cluster import DBSCAN
        
        n_samples = len(X)
        min_samples_per_fold = max(3, n_samples // (n_folds * 2))
        
        stability_scores = []
        fold_results = []
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):  # type: ignore
            try:
                X_train = X[train_idx]
                X_test = X[test_idx]
                
                # Fit on training
                dbscan = DBSCAN(eps=0.5, min_samples=min_samples_per_fold)
                train_labels = dbscan.fit_predict(X_train)
                
                # Filter noise
                valid_mask = train_labels != -1
                
                if np.sum(valid_mask) < 2:
                    continue
                
                # Calculate stability score for this fold
                fold_score = float(silhouette_score(
                    X_train[valid_mask], 
                    train_labels[valid_mask]
                )) if np.sum(valid_mask) > 1 else -1
                
                stability_scores.append(fold_score)
                
                fold_results.append({
                    'fold': fold_idx + 1,
                    'train_size': len(train_idx),
                    'test_size': len(test_idx),
                    'n_clusters': len(set(train_labels)) - (1 if -1 in train_labels else 0),
                    'silhouette': fold_score
                })
                
            except Exception as e:
                logger.warning(f"Fold {fold_idx} failed: {e}")
                continue
        
        stability = {
            'n_folds': n_folds,
            'completed_folds': len(stability_scores),
            'mean_stability': float(np.mean(stability_scores)) if stability_scores else 0.0,
            'std_stability': float(np.std(stability_scores)) if stability_scores else 0.0,
            'stability_cv': float(np.std(stability_scores) / (np.mean(stability_scores) + 1e-6)) if stability_scores else 0.0,
            'fold_details': fold_results
        }
        
        return stability
    
    def _analyze_clusters(self, 
                          X: np.ndarray, 
                          labels: np.ndarray,
                          feature_names: Optional[List[str]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Analyze characteristics of each cluster.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Feature names
            
        Returns:
            Dictionary with per-cluster analysis
        """
        unique_labels = np.unique(labels)
        analysis = {}
        
        for label in unique_labels:
            if label == -1:
                continue
            
            mask = labels == label
            cluster_data = X[mask]
            
            cluster_info = {
                'size': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(labels) * 100),
                'density': float(np.sum(mask) / len(labels)),
            }
            
            # Feature statistics
            if feature_names:
                for i, name in enumerate(feature_names):
                    if i < cluster_data.shape[1]:
                        cluster_info[f'{name}_mean'] = float(np.mean(cluster_data[:, i]))
                        cluster_info[f'{name}_std'] = float(np.std(cluster_data[:, i]))
                        cluster_info[f'{name}_min'] = float(np.min(cluster_data[:, i]))
                        cluster_info[f'{name}_max'] = float(np.max(cluster_data[:, i]))
            
            # Compactness
            if len(cluster_data) > 1:
                center = cluster_data.mean(axis=0)
                distances = cdist(cluster_data, center.reshape(1, -1))
                cluster_info['compactness'] = float(np.mean(distances))
                cluster_info['max_distance_to_center'] = float(np.max(distances))
            
            analysis[int(label)] = cluster_info
        
        return analysis
    
    def _calculate_feature_importance(self, 
                                     X: np.ndarray, 
                                     labels: np.ndarray,
                                     feature_names: List[str]) -> Dict[str, float]:
        """
        Calculate feature importance for clustering using variance ratio.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Feature names
            
        Returns:
            Dictionary of feature importance scores
        """
        importance = {}
        n_features = X.shape[1]
        
        for i, name in enumerate(feature_names):
            if i >= n_features:
                break
            
            # Between-cluster variance / within-cluster variance
            overall_mean = np.mean(X[:, i])
            
            between_var = 0.0
            within_var = 0.0
            
            for label in np.unique(labels):
                if label == -1:
                    continue
                
                mask = labels == label
                cluster_data = X[mask, i]
                cluster_mean = np.mean(cluster_data)
                
                between_var += len(cluster_data) * (cluster_mean - overall_mean)**2
                within_var += np.sum((cluster_data - cluster_mean)**2)
            
            if within_var > 0:
                importance[name] = float(between_var / within_var)
            else:
                importance[name] = 0.0
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def _rate_quality(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Rate overall clustering quality.
        
        Args:
            results: Evaluation results
            
        Returns:
            Quality rating dictionary
        """
        internal = results.get('internal_metrics', {})
        stability = results.get('stability_analysis', {})
        noise_ratio = results.get('dataset_info', {}).get('noise_ratio', 0)
        
        score = 0.0
        categories = []
        
        # Silhouette score (0.35 weight)
        sil = internal.get('silhouette_score', -1)
        if sil > 0.7:
            score += 0.35
            categories.append("Excellent cluster separation")
        elif sil > 0.5:
            score += 0.25
            categories.append("Good cluster separation")
        elif sil > 0.3:
            score += 0.15
            categories.append("Moderate cluster separation")
        elif sil > 0:
            score += 0.05
            categories.append("Poor cluster separation")
        else:
            categories.append("No cluster structure")
        
        # Davies-Bouldin (0.25 weight)
        db = internal.get('davies_bouldin_index', float('inf'))
        if db < 0.5:
            score += 0.25
        elif db < 1.0:
            score += 0.18
        elif db < 2.0:
            score += 0.10
        elif db != float('inf'):
            score += 0.05
        
        # Stability (0.20 weight)
        stab_cv = stability.get('stability_cv', 1.0)
        if stab_cv < 0.1:
            score += 0.20
        elif stab_cv < 0.2:
            score += 0.15
        elif stab_cv < 0.3:
            score += 0.10
        elif stab_cv < 0.5:
            score += 0.05
        
        # Noise ratio penalty (0.20 weight)
        if noise_ratio < 0.1:
            score += 0.20
        elif noise_ratio < 0.2:
            score += 0.15
        elif noise_ratio < 0.3:
            score += 0.10
        elif noise_ratio < 0.5:
            score += 0.05
        else:
            categories.append("High noise ratio")
        
        # Overall rating
        if score >= 0.8:
            rating = "Excellent"
        elif score >= 0.6:
            rating = "Good"
        elif score >= 0.4:
            rating = "Fair"
        elif score >= 0.2:
            rating = "Poor"
        else:
            rating = "Unsatisfactory"
        
        return {
            'quality_score': score,
            'rating': rating,
            'categories': categories
        }
    
    def _save_results(self, results: Dict[str, Any]) -> None:
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results dictionary
        """
        # Save JSON
        self.file_handler.save_json(
            results,
            "evaluation_report",
            str(self.output_dir.relative_to(self.output_dir.parent))
        )
        
        # Save CSV summary
        summary_data = {
            'Metric': [],
            'Value': []
        }
        
        # Internal metrics
        for metric, value in results.get('internal_metrics', {}).items():
            summary_data['Metric'].append(metric.replace('_', ' ').title())
            summary_data['Value'].append(f"{value:.4f}" if isinstance(value, float) else value)
        
        # Stability
        stability = results.get('stability_analysis', {})
        summary_data['Metric'].append('Stability Score')
        summary_data['Value'].append(f"{stability.get('mean_stability', 0):.4f}")
        
        summary_data['Metric'].append('Stability CV')
        summary_data['Value'].append(f"{stability.get('stability_cv', 0):.4f}")
        
        # Quality
        quality = results.get('overall_quality', {})
        summary_data['Metric'].append('Overall Rating')
        summary_data['Value'].append(quality.get('rating', 'N/A'))
        
        summary_df = pd.DataFrame(summary_data)
        self.file_handler.save_csv(
            summary_df,
            "evaluation_summary",
            str(self.output_dir.relative_to(self.output_dir.parent))
        )
        
        # Save per-cluster analysis
        cluster_analysis = results.get('cluster_analysis', {})
        if cluster_analysis:
            cluster_data = []
            for cluster_id, info in cluster_analysis.items():
                row = {'cluster_id': cluster_id}
                row.update(info)
                cluster_data.append(row)
            
            cluster_df = pd.DataFrame(cluster_data)
            self.file_handler.save_csv(
                cluster_df,
                "cluster_analysis",
                str(self.output_dir.relative_to(self.output_dir.parent))
            )
        
        # Save feature importance
        feature_importance = results.get('feature_importance', {})
        if feature_importance:
            importance_df = pd.DataFrame([
                {'feature': k, 'importance': v}
                for k, v in feature_importance.items()
            ])
            self.file_handler.save_csv(
                importance_df,
                "feature_importance",
                str(self.output_dir.relative_to(self.output_dir.parent))
            )
        
        logger.info(f"Evaluation results saved to: {self.output_dir}")
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation summary.
        
        Args:
            results: Evaluation results
        """
        print("\n" + "="*60)
        print("CLUSTERING EVALUATION SUMMARY")
        print("="*60)
        
        # Dataset info
        info = results.get('dataset_info', {})
        print(f"\nDataset Information:")
        print(f"  Samples: {info.get('n_samples', 0)}")
        print(f"  Features: {info.get('n_features', 0)}")
        print(f"  Clusters found: {info.get('n_clusters', 0)}")
        print(f"  Noise points: {info.get('n_noise', 0)} ({info.get('noise_ratio', 0)*100:.1f}%)")
        
        # Internal metrics
        internal = results.get('internal_metrics', {})
        if internal:
            print(f"\nInternal Metrics:")
            print(f"  Silhouette Score:     {internal.get('silhouette_score', -1):.4f}")
            print(f"  Davies-Bouldin Index: {internal.get('davies_bouldin_index', -1):.4f}")
            print(f"  Calinski-Harabasz:    {internal.get('calinski_harabasz_index', -1):.1f}")
            print(f"  Avg Intra-cluster:    {internal.get('avg_intra_cluster_distance', 0):.4f}")
            print(f"  Avg Inter-cluster:    {internal.get('avg_inter_cluster_distance', 0):.4f}")
        
        # External metrics
        external = results.get('external_metrics', {})
        if external:
            print(f"\nExternal Metrics (vs Ground Truth):")
            print(f"  Adjusted Rand Index:     {external.get('adjusted_rand_index', 0):.4f}")
            print(f"  Normalized Mutual Info:  {external.get('normalized_mutual_info', 0):.4f}")
            print(f"  Homogeneity:             {external.get('homogeneity', 0):.4f}")
            print(f"  Completeness:            {external.get('completeness', 0):.4f}")
            print(f"  V-Measure:               {external.get('v_measure', 0):.4f}")
        
        # Stability
        stability = results.get('stability_analysis', {})
        if stability:
            print(f"\nStability Analysis:")
            print(f"  Mean Stability:  {stability.get('mean_stability', 0):.4f}")
            print(f"  Std Stability:   {stability.get('std_stability', 0):.4f}")
            print(f"  Stability CV:    {stability.get('stability_cv', 0):.4f}")
        
        # Feature importance (top 5)
        importance = results.get('feature_importance', {})
        if importance:
            print(f"\nTop 5 Most Important Features:")
            for i, (feature, imp) in enumerate(list(importance.items())[:5]):
                print(f"  {i+1}. {feature}: {imp:.4f}")
        
        # Overall quality
        quality = results.get('overall_quality', {})
        print(f"\nOverall Quality: {quality.get('rating', 'N/A')} (Score: {quality.get('quality_score', 0):.2f})")
        print("="*60 + "\n")
    
    def generate_visualizations(self, 
                               X: np.ndarray,
                               labels: np.ndarray,
                               feature_names: List[str],
                               results: Optional[Dict[str, Any]] = None) -> Dict[str, Path]:
        """
        Generate evaluation visualizations.
        
        Args:
            X: Feature matrix
            labels: Cluster labels
            feature_names: Feature names
            results: Evaluation results
            
        Returns:
            Dictionary of visualization paths
        """
        if results is None:
            results = self.results
        
        viz_paths = {}
        
        # 1. Metrics comparison bar chart
        viz_paths['metrics_comparison'] = self._plot_metrics_comparison(results)
        
        # 2. Feature importance bar chart
        viz_paths['feature_importance'] = self._plot_feature_importance(results)
        
        # 3. Cluster size distribution
        viz_paths['cluster_distribution'] = self._plot_cluster_distribution(X, labels)
        
        # 4. Stability analysis
        if results.get('stability_analysis', {}).get('fold_details'):
            viz_paths['stability_analysis'] = self._plot_stability_analysis(results)
        
        return viz_paths
    
    def _plot_metrics_comparison(self, results: Dict[str, Any]) -> Path:
        """Plot metrics comparison bar chart."""
        internal = results.get('internal_metrics', {})
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics_to_plot = {
            'Silhouette': internal.get('silhouette_score', 0),
            'Dunn Index': internal.get('dunn_index_approx', 0),
        }
        
        # Filter out negative values
        metrics_to_plot = {k: v for k, v in metrics_to_plot.items() if v >= 0}
        
        bars = ax.bar(list(metrics_to_plot.keys()), list(metrics_to_plot.values()), 
                     color=['#2196F3', '#4CAF50', '#FF9800', '#F44336'])
        
        ax.set_ylabel('Score')
        ax.set_title('Clustering Metrics Comparison')
        ax.set_ylim(0, max(1.0, max(metrics_to_plot.values()) * 1.2))
        
        # Add value labels
        for bar, val in zip(bars, metrics_to_plot.values()):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{val:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        path = self.output_dir / "metrics_comparison.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return path
    
    def _plot_feature_importance(self, results: Dict[str, Any]) -> Path:
        """Plot feature importance bar chart."""
        importance = results.get('feature_importance', {})
        
        if not importance:
            return Path()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(importance.keys())
        values = list(importance.values())
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(features)))
        bars = ax.barh(features, values, color=colors)
        
        ax.set_xlabel('Relative Importance')
        ax.set_title('Feature Importance for Clustering')
        
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center')
        
        plt.tight_layout()
        path = self.output_dir / "feature_importance.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return path
    
    def _plot_cluster_distribution(self, X: np.ndarray, labels: np.ndarray) -> Path:
        """Plot cluster size distribution."""
        unique, counts = np.unique(labels, return_counts=True)
        
        # Separate noise
        noise_mask = unique == -1
        noise_count = counts[noise_mask].sum() if noise_mask.any() else 0
        
        cluster_labels = unique[~noise_mask]
        cluster_counts = counts[~noise_mask]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bar chart
        x_pos = range(len(cluster_labels))
        bars = ax.bar(x_pos, cluster_counts, color=plt.cm.Set3(np.linspace(0, 1, len(cluster_labels))))
        
        # Add noise as separate bar
        if noise_count > 0:
            ax.bar(len(cluster_labels), noise_count, color='gray', label='Noise')
        
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Cluster Size Distribution')
        ax.set_xticks(list(x_pos) + ([len(cluster_labels)] if noise_count > 0 else []))
        ax.set_xticklabels(
            [f'C{i}' for i in cluster_labels] + (['Noise'] if noise_count > 0 else [])
        )
        
        # Add count labels
        for bar, count in zip(bars, cluster_counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   str(count), ha='center', va='bottom')
        
        if noise_count > 0:
            ax.legend()
        
        plt.tight_layout()
        path = self.output_dir / "cluster_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return path
    
    def _plot_stability_analysis(self, results: Dict[str, Any]) -> Path:
        """Plot stability analysis results."""
        stability = results.get('stability_analysis', {})
        fold_details = stability.get('fold_details', [])
        
        if not fold_details:
            return Path()
        
        folds = [f['fold'] for f in fold_details]
        scores = [f['silhouette'] for f in fold_details]
        clusters = [f['n_clusters'] for f in fold_details]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot silhouette scores across folds
        ax1.plot(folds, scores, 'bo-', linewidth=2, markersize=8)
        ax1.axhline(y=stability.get('mean_stability', 0), color='r', 
                   linestyle='--', label=f"Mean: {stability.get('mean_stability', 0):.3f}")
        ax1.fill_between(folds, 
                        [stability.get('mean_stability', 0) - stability.get('std_stability', 0)] * len(folds),
                        [stability.get('mean_stability', 0) + stability.get('std_stability', 0)] * len(folds),
                        alpha=0.2, color='r')
        ax1.set_xlabel('Fold')
        ax1.set_ylabel('Silhouette Score')
        ax1.set_title('Stability Across Folds')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot number of clusters across folds
        ax2.plot(folds, clusters, 'go-', linewidth=2, markersize=8)
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Cluster Count Across Folds')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f"Stability Analysis (CV: {stability.get('stability_cv', 0):.3f})")
        plt.tight_layout()
        
        path = self.output_dir / "stability_analysis.png"
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return path


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Clustering Evaluation"
    )
    
    parser.add_argument(
        '--features', '-f',
        type=str,
        required=True,
        help='Path to features CSV file'
    )
    
    parser.add_argument(
        '--labels', '-l',
        type=str,
        required=True,
        help='Path to cluster labels CSV file'
    )
    
    parser.add_argument(
        '--ground-truth', '-g',
        type=str,
        default=None,
        help='Path to ground truth labels CSV (optional)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/evaluation',
        help='Output directory for evaluation results'
    )
    
    parser.add_argument(
        '--no-stability',
        action='store_true',
        help='Skip stability analysis'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )
    
    args = parser.parse_args()
    
    # Load data
    features_path = Path(args.features)
    labels_path = Path(args.labels)
    
    if not features_path.exists():
        # Try glob pattern
        features_files = list(Path('.').glob(args.features))
        if features_files:
            features_path = features_files[0]
        else:
            logger.error(f"Features file not found: {args.features}")
            sys.exit(1)
    
    if not labels_path.exists():
        labels_files = list(Path('.').glob(args.labels))
        if labels_files:
            labels_path = labels_files[0]
        else:
            logger.error(f"Labels file not found: {args.labels}")
            sys.exit(1)
    
    logger.info(f"Loading features from: {features_path}")
    features_df = pd.read_csv(features_path)
    
    logger.info(f"Loading labels from: {labels_path}")
    labels_df = pd.read_csv(labels_path)
    
    # Extract feature matrix
    feature_columns = [
        'avg_vehicle_count', 'avg_density', 'avg_occupancy',
        'avg_speed', 'avg_flow', 'avg_congestion_index',
        'speed_variance', 'density_variance'
    ]
    feature_columns = [col for col in feature_columns if col in features_df.columns]
    
    if not feature_columns:
        # Use all numeric columns except timestamp
        feature_columns = features_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = [c for c in feature_columns if 'timestamp' not in c.lower()]
    
    X = features_df[feature_columns].values
    labels = labels_df['cluster_label'].values
    
    logger.info(f"Feature matrix: {X.shape}")
    logger.info(f"Unique labels: {np.unique(labels)}")
    
    # Load ground truth if available
    ground_truth = None
    if args.ground_truth:
        gt_path = Path(args.ground_truth)
        if gt_path.exists():
            gt_df = pd.read_csv(gt_path)
            if 'true_label' in gt_df.columns:
                ground_truth = gt_df['true_label'].values
            elif 'label' in gt_df.columns:
                ground_truth = gt_df['label'].values
            
            if ground_truth is not None:
                logger.info(f"Ground truth loaded: {len(ground_truth)} samples")
    
    # Run evaluation
    evaluator = ClusteringEvaluator(output_dir=Path(args.output))
    
    results = evaluator.evaluate(
        X=X,
        labels=labels,
        feature_names=feature_columns,
        ground_truth=ground_truth,
        run_stability=not args.no_stability
    )
    
    # Generate visualizations
    if not args.no_visualizations:
        logger.info("Generating evaluation visualizations...")
        viz_paths = evaluator.generate_visualizations(
            X, labels, feature_columns, results
        )
        logger.info(f"Generated {len(viz_paths)} visualizations")
    
    logger.info(f"\n✅ Evaluation complete! Results saved to: {args.output}")
    
    # Print final message
    quality = results.get('overall_quality', {})
    print(f"\n{'='*60}")
    print(f"FINAL VERDICT: {quality.get('rating', 'N/A')}")
    print(f"Quality Score: {quality.get('quality_score', 0):.2f} / 1.00")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()