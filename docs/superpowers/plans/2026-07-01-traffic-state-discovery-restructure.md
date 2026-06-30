# Traffic State Discovery Project Restructuring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the `traffic_state_discovery_scaffold` repository to match the `struktur folder.md` reference precisely, removing all unwanted files and replacing all Python, configuration, LaTeX, and documentation files in the target structure with 0-byte (empty) files.

**Architecture:** Clean and Rebuild. All files in existing directories (`src/`, `configs/`, etc.) and the root `run_pipeline.py` script will be deleted first. Then, the directory structure will be systematically rebuilt using terminal commands and touch operations to ensure all files are created as completely empty (0-byte) files.

**Tech Stack:** Bash shell commands

## Global Constraints
- Target files must be completely empty (0-byte).
- The root files `README.md`, `requirements.txt`, and `struktur folder.md` must be preserved.
- The `docs/` folder (including design specs and plans) must be preserved.
- Folder structure must be exactly as defined in `struktur folder.md`.

---

### Task 1: Clean Existing Directories and Root Pipeline
**Files:**
- Modify: Delete `src/` directory tree
- Modify: Delete `configs/` directory tree
- Modify: Delete `run_pipeline.py`

**Interfaces:**
- Consumes: None
- Produces: A clean workspace containing only `README.md`, `requirements.txt`, `struktur folder.md`, and the `docs/` directory.

- [ ] **Step 1: Delete unwanted files and folders**
  Run:
  ```bash
  rm -rf src/ configs/ run_pipeline.py
  ```
- [ ] **Step 2: Verify deletion**
  Run:
  ```bash
  [ ! -d src ] && [ ! -d configs ] && [ ! -f run_pipeline.py ] && echo "Clean successful" || echo "Clean failed"
  ```
  Expected output: `Clean successful`
- [ ] **Step 3: Commit deletions**
  Run:
  ```bash
  git add -A
  git commit -m "style: remove old src, configs, and root run_pipeline"
  ```

---

### Task 2: Rebuild Core Configuration and Root files
**Files:**
- Create: `.env.example`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `setup.py`
- Create: `configs/default_config.yaml`
- Create: `configs/roi_config.json`
- Create: `configs/pipeline_config.yaml`

**Interfaces:**
- Consumes: Clean workspace from Task 1
- Produces: Empty root level configuration files and configurations in `/configs/`

- [ ] **Step 1: Create root files**
  Run:
  ```bash
  touch .env.example .gitignore LICENSE setup.py
  ```
- [ ] **Step 2: Create configs directory and config files**
  Run:
  ```bash
  mkdir -p configs
  touch configs/default_config.yaml configs/roi_config.json configs/pipeline_config.yaml
  ```
- [ ] **Step 3: Verify existence and size of files**
  Run:
  ```bash
  for f in .env.example .gitignore LICENSE setup.py configs/default_config.yaml configs/roi_config.json configs/pipeline_config.yaml; do
    [ -f "$f" ] && [ ! -s "$f" ] && echo "$f: OK" || echo "$f: FAIL"
  done
  ```
  Expected output:
  ```
  .env.example: OK
  .gitignore: OK
  LICENSE: OK
  setup.py: OK
  configs/default_config.yaml: OK
  configs/roi_config.json: OK
  configs/pipeline_config.yaml: OK
  ```
- [ ] **Step 4: Commit new configurations**
  Run:
  ```bash
  git add .env.example .gitignore LICENSE setup.py configs/
  git commit -m "style: create empty configuration and root files"
  ```

---

### Task 3: Rebuild Datasets, Videos, Models, Weights, and Paper Files
**Files:**
- Create: `datasets/README.md`
- Create: `videos/README.md`
- Create: `models/README.md`
- Create: `weights/README.md`
- Create: `paper/tables/cluster_statistics.tex`
- Create: `paper/tables/evaluation_metrics.tex`
- Create: `paper/tables/dataset_summary.tex`
- Create: `paper/figures/` (directory only)

**Interfaces:**
- Consumes: Workspace state from Task 2
- Produces: Project resources directories and paper outline directories

- [ ] **Step 1: Create directories**
  Run:
  ```bash
  mkdir -p datasets videos models weights paper/tables paper/figures
  ```
- [ ] **Step 2: Create files in resources and paper**
  Run:
  ```bash
  touch datasets/README.md videos/README.md models/README.md weights/README.md
  touch paper/tables/cluster_statistics.tex paper/tables/evaluation_metrics.tex paper/tables/dataset_summary.tex
  ```
- [ ] **Step 3: Verify resource files**
  Run:
  ```bash
  for f in datasets/README.md videos/README.md models/README.md weights/README.md paper/tables/cluster_statistics.tex paper/tables/evaluation_metrics.tex paper/tables/dataset_summary.tex; do
    [ -f "$f" ] && [ ! -s "$f" ] && echo "$f: OK" || echo "$f: FAIL"
  done
  [ -d paper/figures ] && echo "paper/figures: OK" || echo "paper/figures: FAIL"
  ```
  Expected output: All lines printed with `OK`
- [ ] **Step 4: Commit resource files**
  Run:
  ```bash
  git add datasets/ videos/ models/ weights/ paper/
  git commit -m "style: create resource directories and empty paper files"
  ```

---

### Task 4: Rebuild Main Source Code Structure (src/)
**Files:**
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `src/logger.py`
- Create: `src/pipeline.py`
- Create: `src/detection/__init__.py`, `src/detection/detector.py`, `src/detection/model_loader.py`
- Create: `src/tracking/__init__.py`, `src/tracking/byte_tracker.py`, `src/tracking/track_manager.py`
- Create: `src/roi/__init__.py`, `src/roi/roi_selector.py`, `src/roi/roi_manager.py`, `src/roi/roi_validator.py`
- Create: `src/features/__init__.py`, `src/features/feature_extractor.py`, `src/features/speed_estimator.py`, `src/features/density_calculator.py`, `src/features/flow_analyzer.py`
- Create: `src/temporal/__init__.py`, `src/temporal/temporal_aggregator.py`, `src/temporal/window_manager.py`
- Create: `src/clustering/__init__.py`, `src/clustering/dbscan_clustering.py`, `src/clustering/grid_search.py`, `src/clustering/cluster_evaluator.py`
- Create: `src/analytics/__init__.py`, `src/analytics/traffic_state_analyzer.py`, `src/analytics/statistics_calculator.py`
- Create: `src/visualization/__init__.py`, `src/visualization/traffic_visualizer.py`, `src/visualization/cluster_visualizer.py`, `src/visualization/heatmap_generator.py`, `src/visualization/video_renderer.py`, `src/visualization/paper_plotter.py`
- Create: `src/evaluation/__init__.py`, `src/evaluation/metrics_calculator.py`, `src/evaluation/result_analyzer.py`
- Create: `src/utils/__init__.py`, `src/utils/file_handler.py`, `src/utils/video_reader.py`, `src/utils/data_validator.py`, `src/utils/progress_tracker.py`

**Interfaces:**
- Consumes: Workspace state from Task 3
- Produces: Empty root level code files and modules inside `/src/`

- [ ] **Step 1: Create all src directories**
  Run:
  ```bash
  mkdir -p src/detection src/tracking src/roi src/features src/temporal src/clustering src/analytics src/visualization src/evaluation src/utils
  ```
- [ ] **Step 2: Create files**
  Run:
  ```bash
  touch src/__init__.py src/config.py src/logger.py src/pipeline.py
  touch src/detection/__init__.py src/detection/detector.py src/detection/model_loader.py
  touch src/tracking/__init__.py src/tracking/byte_tracker.py src/tracking/track_manager.py
  touch src/roi/__init__.py src/roi/roi_selector.py src/roi/roi_manager.py src/roi/roi_validator.py
  touch src/features/__init__.py src/features/feature_extractor.py src/features/speed_estimator.py src/features/density_calculator.py src/features/flow_analyzer.py
  touch src/temporal/__init__.py src/temporal/temporal_aggregator.py src/temporal/window_manager.py
  touch src/clustering/__init__.py src/clustering/dbscan_clustering.py src/clustering/grid_search.py src/clustering/cluster_evaluator.py
  touch src/analytics/__init__.py src/analytics/traffic_state_analyzer.py src/analytics/statistics_calculator.py
  touch src/visualization/__init__.py src/visualization/traffic_visualizer.py src/visualization/cluster_visualizer.py src/visualization/heatmap_generator.py src/visualization/video_renderer.py src/visualization/paper_plotter.py
  touch src/evaluation/__init__.py src/evaluation/metrics_calculator.py src/evaluation/result_analyzer.py
  touch src/utils/__init__.py src/utils/file_handler.py src/utils/video_reader.py src/utils/data_validator.py src/utils/progress_tracker.py
  ```
- [ ] **Step 3: Verify src files**
  Run:
  ```bash
  find src/ -type f | while read -r f; do
    if [ -s "$f" ]; then
      echo "$f: FAIL (not empty)"
    fi
  done
  echo "src/ verification complete"
  ```
  Expected output: `src/ verification complete` (no files outputting FAIL)
- [ ] **Step 4: Commit src files**
  Run:
  ```bash
  git add src/
  git commit -m "style: rebuild empty src/ modules"
  ```

---

### Task 5: Rebuild Execution Scripts, Outputs, and Tests
**Files:**
- Create: `scripts/run_pipeline.py`
- Create: `scripts/run_roi_selector.py`
- Create: `scripts/run_grid_search.py`
- Create: `scripts/run_evaluation.py`
- Create: `scripts/generate_paper_figures.py`
- Create: `tests/test_detection.py`
- Create: `tests/test_tracking.py`
- Create: `tests/test_features.py`
- Create: `tests/test_clustering.py`
- Create: Output directories: `outputs/csv/raw_features/`, `outputs/csv/temporal_features/`, `outputs/csv/clusters/`, `outputs/csv/evaluation/`, `outputs/plots/density/`, `outputs/plots/occupancy/`, `outputs/plots/speed/`, `outputs/plots/flow/`, `outputs/plots/clusters/`, `outputs/plots/heatmaps/`, `outputs/clusters/assignments/`, `outputs/clusters/statistics/`, `outputs/videos/tracking/`, `outputs/videos/roi/`, `outputs/videos/traffic_state/`, `outputs/logs/pipeline_runs/`

**Interfaces:**
- Consumes: Workspace state from Task 4
- Produces: Empty execution scripts, test files, and output folder structures.

- [ ] **Step 1: Create directories**
  Run:
  ```bash
  mkdir -p scripts tests
  mkdir -p outputs/csv/raw_features outputs/csv/temporal_features outputs/csv/clusters outputs/csv/evaluation
  mkdir -p outputs/plots/density outputs/plots/occupancy outputs/plots/speed outputs/plots/flow outputs/plots/clusters outputs/plots/heatmaps
  mkdir -p outputs/clusters/assignments outputs/clusters/statistics
  mkdir -p outputs/videos/tracking outputs/videos/roi outputs/videos/traffic_state
  mkdir -p outputs/logs/pipeline_runs
  ```
- [ ] **Step 2: Create files**
  Run:
  ```bash
  touch scripts/run_pipeline.py scripts/run_roi_selector.py scripts/run_grid_search.py scripts/run_evaluation.py scripts/generate_paper_figures.py
  touch tests/test_detection.py tests/test_tracking.py tests/test_features.py tests/test_clustering.py
  ```
- [ ] **Step 3: Verify scripts and tests**
  Run:
  ```bash
  for f in scripts/* tests/*; do
    [ -f "$f" ] && [ ! -s "$f" ] && echo "$f: OK" || echo "$f: FAIL"
  done
  ```
  Expected output:
  ```
  scripts/generate_paper_figures.py: OK
  scripts/run_evaluation.py: OK
  scripts/run_grid_search.py: OK
  scripts/run_pipeline.py: OK
  scripts/run_roi_selector.py: OK
  tests/test_clustering.py: OK
  tests/test_detection.py: OK
  tests/test_features.py: OK
  tests/test_tracking.py: OK
  ```
- [ ] **Step 4: Commit new files**
  Run:
  ```bash
  git add scripts/ tests/ outputs/
  git commit -m "style: create empty execution scripts, test files, and output directories"
  ```
