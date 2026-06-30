#!/usr/bin/env python3
"""
Grid search for optimal DBSCAN parameters.
Usage: python scripts/run_grid_search.py --features path/to/features.csv
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ProjectConfig, ClusteringConfig
from src.clustering.grid_search import GridSearchOptimizer
from src.clustering.cluster_evaluator import ClusterEvaluator
from src.clustering.dbscan_clustering import DBSCANClusterer
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="DBSCAN Parameter Grid Search"
    )
    
    parser.add_argument(
        '--features', '-f',
        type=str,
        required=True,
        help='Path to features CSV file'
    )
    
    parser.add_argument(
        '--eps-range', '-e',
        type=float,
        nargs='+',
        default=[0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0],
        help='Epsilon values to search'
    )
    
    parser.add_argument(
        '--min-samples-range', '-m',
        type=int,
        nargs='+',
        default=[3, 5, 10, 15, 20],
        help='Min samples values to search'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/csv/clusters/grid_search_results.csv',
        help='Output path for results'
    )
    
    args = parser.parse_args()
    
    # Load features
    features_path = Path(args.features)
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        sys.exit(1)
    
    df = pd.read_csv(features_path)
    
    # Select numeric features
    feature_columns = [
        'avg_vehicle_count', 'avg_density', 'avg_occupancy',
        'avg_speed', 'avg_flow', 'avg_congestion_index',
        'speed_variance', 'density_variance'
    ]
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    X = df[feature_columns].values
    
    # Configure clustering
    config = ClusteringConfig()
    config.eps_range = args.eps_range
    config.min_samples_range = args.min_samples_range
    
    # Run grid search
    logger.info(f"Running grid search: {len(args.eps_range)} x {len(args.min_samples_range)} combinations")
    
    searcher = GridSearchOptimizer(config)
    grid_result = searcher.search(X, feature_columns)
    
    # Save results
    results_df = searcher.get_search_dataframe()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    logger.info(f"✅ Grid search complete")
    logger.info(f"Best parameters: eps={grid_result.best_params['eps']}, "
               f"min_samples={grid_result.best_params['min_samples']}")
    logger.info(f"Best score: {grid_result.best_score:.4f}")
    logger.info(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()