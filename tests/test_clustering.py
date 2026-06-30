"""
Unit tests for clustering module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.config import ClusteringConfig
from src.clustering.dbscan_clustering import DBSCANClusterer
from src.clustering.cluster_evaluator import ClusterEvaluator


class TestDBSCANClusterer:
    """Test suite for DBSCANClusterer."""
    
    @pytest.fixture
    def config(self):
        """Create clustering config."""
        return ClusteringConfig(eps=0.5, min_samples=3)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample clustered data."""
        np.random.seed(42)
        
        # Create 3 clusters
        cluster1 = np.random.randn(50, 5) + np.array([0, 0, 0, 0, 0])
        cluster2 = np.random.randn(50, 5) + np.array([5, 5, 5, 5, 5])
        cluster3 = np.random.randn(50, 5) + np.array([-5, -5, -5, -5, -5])
        
        X = np.vstack([cluster1, cluster2, cluster3])
        return X
    
    def test_fit(self, config, sample_data):
        """Test clustering fit."""
        clusterer = DBSCANClusterer(config)
        feature_names = ['f1', 'f2', 'f3', 'f4', 'f5']
        result = clusterer.fit(sample_data, feature_names=feature_names)
        
        assert result.n_clusters > 0
        assert len(result.labels) == len(sample_data)
        assert len(result.cluster_centers) == result.n_clusters
        
        # Verify cluster statistics are calculated with feature-wise mean and std
        assert len(result.cluster_statistics) > 0
        for label, stats in result.cluster_statistics.items():
            assert 'size' in stats
            assert 'density' in stats
            assert 'f1_mean' in stats
            assert 'f1_std' in stats
    
    def test_cluster_summary(self, config, sample_data):
        """Test cluster summary generation."""
        clusterer = DBSCANClusterer(config)
        result = clusterer.fit(sample_data)
        summary = clusterer.get_cluster_summary()
        
        assert 'n_clusters' in summary
        assert 'silhouette_score' in summary
        assert 'cluster_sizes' in summary
        assert 'state_mapping' in summary
    
    def test_empty_data(self, config):
        """Test handling of empty data."""
        clusterer = DBSCANClusterer(config)
        
        with pytest.raises(ValueError):
            clusterer.fit(np.array([]))


class TestClusterEvaluator:
    """Test suite for ClusterEvaluator."""
    
    @pytest.fixture
    def sample_data_and_labels(self):
        """Create sample data with known labels."""
        np.random.seed(42)
        
        # Create 2 distinct clusters
        X1 = np.random.randn(30, 4)
        X2 = np.random.randn(30, 4) + 3
        X = np.vstack([X1, X2])
        labels = np.array([0]*30 + [1]*30)
        
        return X, labels
    
    def test_evaluation(self, sample_data_and_labels):
        """Test clustering evaluation."""
        X, labels = sample_data_and_labels
        
        evaluator = ClusterEvaluator()
        
        # Create mock cluster result
        from src.clustering.dbscan_clustering import ClusterResult
        result = ClusterResult(
            labels=labels,
            n_clusters=2,
            n_noise=0,
            cluster_sizes={0: 30, 1: 30},
            cluster_centers={0: X[:30].mean(axis=0), 1: X[30:].mean(axis=0)},
            silhouette_score=0.8,
            parameters={'eps': 0.5, 'min_samples': 5},
            feature_names=['f1', 'f2', 'f3', 'f4']
        )
        
        report = evaluator.evaluate(X, result)
        
        assert report.silhouette_score > 0
        assert report.davies_bouldin_index < 10
        assert report.noise_ratio == 0.0
        assert report.cluster_separation > 0