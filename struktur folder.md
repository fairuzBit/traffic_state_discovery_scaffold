traffic-state-discovery/
│
├── configs/ # Konfigurasi
│ ├── default_config.yaml
│ ├── roi_config.json
│ └── pipeline_config.yaml
│
├── datasets/ # Dataset (kosong untuk data user)
│ └── README.md
│
├── videos/ # Video input
│ └── README.md
│
├── models/ # Model YOLO
│ └── README.md
│
├── weights/ # Bobot model custom
│ └── README.md
│
├── src/ # Source code utama
│ ├── **init**.py
│ ├── config.py
│ ├── logger.py
│ ├── pipeline.py # Main pipeline orchestrator
│ │
│ ├── detection/ # Modul deteksi
│ │ ├── **init**.py
│ │ ├── detector.py
│ │ └── model_loader.py
│ │
│ ├── tracking/ # Modul tracking
│ │ ├── **init**.py
│ │ ├── byte_tracker.py
│ │ └── track_manager.py
│ │
│ ├── roi/ # Modul ROI
│ │ ├── **init**.py
│ │ ├── roi_selector.py
│ │ ├── roi_manager.py
│ │ └── roi_validator.py
│ │
│ ├── features/ # Ekstraksi fitur
│ │ ├── **init**.py
│ │ ├── feature_extractor.py
│ │ ├── speed_estimator.py
│ │ ├── density_calculator.py
│ │ └── flow_analyzer.py
│ │
│ ├── temporal/ # Agregasi temporal
│ │ ├── **init**.py
│ │ ├── temporal_aggregator.py
│ │ └── window_manager.py
│ │
│ ├── clustering/ # Clustering
│ │ ├── **init**.py
│ │ ├── dbscan_clustering.py
│ │ ├── grid_search.py
│ │ └── cluster_evaluator.py
│ │
│ ├── analytics/ # Analitik
│ │ ├── **init**.py
│ │ ├── traffic_state_analyzer.py
│ │ └── statistics_calculator.py
│ │
│ ├── visualization/ # Visualisasi
│ │ ├── **init**.py
│ │ ├── traffic_visualizer.py
│ │ ├── cluster_visualizer.py
│ │ ├── heatmap_generator.py
│ │ ├── video_renderer.py
│ │ └── paper_plotter.py
│ │
│ ├── evaluation/ # Evaluasi
│ │ ├── **init**.py
│ │ ├── metrics_calculator.py
│ │ └── result_analyzer.py
│ │
│ └── utils/ # Utilitas
│ ├── **init**.py
│ ├── file_handler.py
│ ├── video_reader.py
│ ├── data_validator.py
│ └── progress_tracker.py
│
├── outputs/ # Output otomatis
│ ├── csv/ # Data CSV
│ │ ├── raw_features/
│ │ ├── temporal_features/
│ │ ├── clusters/
│ │ └── evaluation/
│ │
│ ├── plots/ # Grafik
│ │ ├── density/
│ │ ├── occupancy/
│ │ ├── speed/
│ │ ├── flow/
│ │ ├── clusters/
│ │ └── heatmaps/
│ │
│ ├── clusters/ # Hasil clustering
│ │ ├── assignments/
│ │ └── statistics/
│ │
│ ├── videos/ # Video output
│ │ ├── tracking/
│ │ ├── roi/
│ │ └── traffic_state/
│ │
│ └── logs/ # Log detail
│ └── pipeline_runs/
│
├── paper/ # Output untuk paper
│ ├── tables/ # Tabel LaTeX
│ │ ├── cluster_statistics.tex
│ │ ├── evaluation_metrics.tex
│ │ └── dataset_summary.tex
│ │
│ └── figures/ # Gambar paper-ready
│ ├── pipeline_overview.png
│ ├── cluster_distribution.png
│ ├── temporal_patterns.png
│ ├── congestion_heatmap.png
│ └── state_transitions.png
│
├── scripts/ # Script eksekusi
│ ├── run_pipeline.py # Pipeline lengkap
│ ├── run_roi_selector.py # Tool ROI interaktif
│ ├── run_grid_search.py # Grid search clustering
│ ├── run_evaluation.py # Evaluasi model
│ └── generate_paper_figures.py # Generate semua figure
│
├── tests/ # Unit tests
│ ├── test_detection.py
│ ├── test_tracking.py
│ ├── test_features.py
│ └── test_clustering.py
│
├── README.md
├── requirements.txt
├── setup.py
├── .gitignore
├── LICENSE
└── .env.example
