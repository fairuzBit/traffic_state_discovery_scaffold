# Design Spec: Traffic State Discovery Project Restructuring

## 1. Overview
The goal of this task is to align the project directory structure of the `traffic_state_discovery_scaffold` repository with the guidelines specified in `struktur folder.md`. 

To achieve a clean state as requested by the user, we will follow the **Clean and Rebuild** approach. All files in the `src/` and `configs/` directories, and other non-standard files will be removed, and the directory tree will be rebuilt from scratch with empty (0-byte) placeholder files as specified in the reference structure.

## 2. Scope & Constraints
- **Preserve**: Root-level files: `README.md`, `requirements.txt`, and `struktur folder.md`.
- **Remove**: Current `src/` directory, current `configs/` directory, and `run_pipeline.py` at the root.
- **Create**: Fully empty (0-byte) placeholder files matching the new tree structure.
- **Languages**: Standard Python files (`.py`), JSON configs (`.json`), YAML configs (`.yaml`), LaTeX tables (`.tex`), markdown readmes (`.md`), and config/root files (`setup.py`, `.gitignore`, `LICENSE`, `.env.example`).
- **Data Content**: All newly created files must be completely empty (0 bytes).

## 3. Detailed Structure Alignment

### Root Files to Keep
- `/README.md`
- `/requirements.txt`
- `/struktur folder.md`

### Root Files to Create (Empty)
- `/.env.example`
- `/.gitignore`
- `/LICENSE`
- `/setup.py`

### Configuration Files to Create
- `/configs/default_config.yaml`
- `/configs/roi_config.json`
- `/configs/pipeline_config.yaml`

### Datasets, Videos, Models, Weights
- `/datasets/README.md`
- `/videos/README.md`
- `/models/README.md`
- `/weights/README.md`

### Main Source Modules (under `/src/`)
- `/src/__init__.py`
- `/src/config.py`
- `/src/logger.py`
- `/src/pipeline.py`

#### Modules
- **Detection**: `/src/detection/__init__.py`, `/src/detection/detector.py`, `/src/detection/model_loader.py`
- **Tracking**: `/src/tracking/__init__.py`, `/src/tracking/byte_tracker.py`, `/src/tracking/track_manager.py`
- **ROI**: `/src/roi/__init__.py`, `/src/roi/roi_selector.py`, `/src/roi/roi_manager.py`, `/src/roi/roi_validator.py`
- **Features**: `/src/features/__init__.py`, `/src/features/feature_extractor.py`, `/src/features/speed_estimator.py`, `/src/features/density_calculator.py`, `/src/features/flow_analyzer.py`
- **Temporal**: `/src/temporal/__init__.py`, `/src/temporal/temporal_aggregator.py`, `/src/temporal/window_manager.py`
- **Clustering**: `/src/clustering/__init__.py`, `/src/clustering/dbscan_clustering.py`, `/src/clustering/grid_search.py`, `/src/clustering/cluster_evaluator.py`
- **Analytics**: `/src/analytics/__init__.py`, `/src/analytics/traffic_state_analyzer.py`, `/src/analytics/statistics_calculator.py`
- **Visualization**: `/src/visualization/__init__.py`, `/src/visualization/traffic_visualizer.py`, `/src/visualization/cluster_visualizer.py`, `/src/visualization/heatmap_generator.py`, `/src/visualization/video_renderer.py`, `/src/visualization/paper_plotter.py`
- **Evaluation**: `/src/evaluation/__init__.py`, `/src/evaluation/metrics_calculator.py`, `/src/evaluation/result_analyzer.py`
- **Utils**: `/src/utils/__init__.py`, `/src/utils/file_handler.py`, `/src/utils/video_reader.py`, `/src/utils/data_validator.py`, `/src/utils/progress_tracker.py`

### Output Directories to Create
- `/outputs/csv/raw_features/`
- `/outputs/csv/temporal_features/`
- `/outputs/csv/clusters/`
- `/outputs/csv/evaluation/`
- `/outputs/plots/density/`
- `/outputs/plots/occupancy/`
- `/outputs/plots/speed/`
- `/outputs/plots/flow/`
- `/outputs/plots/clusters/`
- `/outputs/plots/heatmaps/`
- `/outputs/clusters/assignments/`
- `/outputs/clusters/statistics/`
- `/outputs/videos/tracking/`
- `/outputs/videos/roi/`
- `/outputs/videos/traffic_state/`
- `/outputs/logs/pipeline_runs/`

### Paper Directory
- `/paper/tables/cluster_statistics.tex`
- `/paper/tables/evaluation_metrics.tex`
- `/paper/tables/dataset_summary.tex`
- `/paper/figures/` (directory created, figures generated dynamically via scripts)

### Execution Scripts
- `/scripts/run_pipeline.py`
- `/scripts/run_roi_selector.py`
- `/scripts/run_grid_search.py`
- `/scripts/run_evaluation.py`
- `/scripts/generate_paper_figures.py`

### Tests
- `/tests/test_detection.py`
- `/tests/test_tracking.py`
- `/tests/test_features.py`
- `/tests/test_clustering.py`

## 4. Execution Plan
1. Delete old `/src`, `/configs`, `/run_pipeline.py`.
2. Recreate all folder trees using shell commands.
3. Write empty contents (touch files) to ensure they are 0-byte.
4. Verify the tree structure matches the specified layout exactly.
