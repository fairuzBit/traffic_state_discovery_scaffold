#!/usr/bin/env python3
"""
Generate all publication-ready figures for paper.
Usage: python scripts/generate_paper_figures.py --results path/to/results
"""

import sys
import argparse
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ProjectConfig
from src.visualization.paper_plotter import PaperPlotter
from src.clustering.dbscan_clustering import ClusterResult
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Generate Paper-Ready Figures"
    )
    
    parser.add_argument(
        '--results', '-r',
        type=str,
        required=True,
        help='Path to pipeline results directory'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='paper/figures',
        help='Output directory for figures'
    )
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    output_path = Path(args.output)
    
    if not results_path.exists():
        logger.error(f"Results directory not found: {results_path}")
        sys.exit(1)
    
    logger.info("Loading pipeline results...")
    
    # Load aggregated features
    features_csv = list(results_path.glob("csv/temporal_features/*.csv"))
    if not features_csv:
        logger.error("No temporal features found")
        sys.exit(1)
    
    df = pd.read_csv(features_csv[0])
    
    # Load cluster results
    cluster_json = list(results_path.glob("clusters/statistics/*.json"))
    if not cluster_json:
        logger.error("No cluster results found")
        sys.exit(1)
    
    with open(cluster_json[0]) as f:
        cluster_data = json.load(f)
    
    # Extract feature matrix
    feature_columns = [
        'avg_vehicle_count', 'avg_density', 'avg_occupancy',
        'avg_speed', 'avg_flow', 'avg_congestion_index'
    ]
    feature_columns = [col for col in feature_columns if col in df.columns]
    X = df[feature_columns].values
    
    # Reconstruct cluster result (simplified)
    cluster_result = ClusterResult(
        labels=np.zeros(len(df)),  # Placeholder
        n_clusters=cluster_data.get('n_clusters', 0),
        n_noise=cluster_data.get('n_noise', 0),
        cluster_sizes=cluster_data.get('cluster_sizes', {}),
        cluster_centers={},
        silhouette_score=cluster_data.get('silhouette_score', 0),
        parameters=cluster_data.get('parameters', {}),
        feature_names=feature_columns,
        state_mapping=cluster_data.get('state_mapping', {})
    )
    
    # Generate figures
    config = ProjectConfig()
    plotter = PaperPlotter(config.visualization)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Generating paper figures...")
    
    # Pipeline overview
    plotter.plot_pipeline_overview(output_path / "pipeline_overview")
    logger.info("✅ Pipeline overview")
    
    # Cluster comparison
    plotter.plot_cluster_comparison(X, cluster_result, output_path / "cluster_comparison")
    logger.info("✅ Cluster comparison")
    
    # Evaluation summary (if available)
    eval_json = list(results_path.glob("csv/evaluation/*.json"))
    if eval_json:
        with open(eval_json[0]) as f:
            eval_data = json.load(f)
        
        # Add grid search data if available
        grid_csv = list(results_path.glob("csv/clusters/grid_search*.csv"))
        if grid_csv:
            eval_data['grid_search'] = pd.read_csv(grid_csv[0]).to_dict('records')
        
        plotter.plot_evaluation_summary(eval_data, output_path / "evaluation_summary")
        logger.info("✅ Evaluation summary")
    
    logger.info(f"\n✅ All paper figures saved to: {output_path}")


if __name__ == "__main__":
    main()