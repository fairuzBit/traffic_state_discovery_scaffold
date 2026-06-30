"""
Unit tests for vehicle detection module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from src.config import DetectionConfig
from src.detection.detector import VehicleDetector, Detection
from src.detection.model_loader import ModelLoader


class TestVehicleDetector:
    """Test suite for VehicleDetector."""
    
    @pytest.fixture
    def config(self):
        """Create test detection config."""
        return DetectionConfig(
            model_path="models/yolov8n.pt",
            confidence_threshold=0.5,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_frame(self):
        """Create a sample frame."""
        return np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    def test_initialization(self, config):
        """Test detector initialization."""
        detector = VehicleDetector(config)
        assert detector.model is not None
        assert detector.config.classes == [2, 3, 5, 7]
    
    def test_detection_on_frame(self, config, sample_frame):
        """Test detection on a single frame."""
        detector = VehicleDetector(config)
        detections = detector.detect(sample_frame)
        
        assert isinstance(detections, list)
        for det in detections:
            assert isinstance(det, Detection)
            assert len(det.bbox) == 4
            assert 0 <= det.confidence <= 1
            assert det.class_name in ['car', 'motorcycle', 'bus', 'truck']
    
    def test_detection_summary(self, config, sample_frame):
        """Test detection summary generation."""
        detector = VehicleDetector(config)
        detections = detector.detect(sample_frame)
        summary = detector.get_detection_summary(detections)
        
        assert 'total' in summary
        assert 'classes' in summary
        assert 'avg_confidence' in summary
    
    def test_filter_by_area(self, config):
        """Test area-based filtering."""
        detector = VehicleDetector(config)
        
        detections = [
            Detection((0, 0, 5, 5), 0.9, 2, 'car'),
            Detection((0, 0, 100, 100), 0.9, 2, 'car'),
        ]
        
        filtered = detector.filter_detections_by_area(detections, min_area=50)
        assert len(filtered) == 1
        assert filtered[0].area == 10000


class TestModelLoader:
    """Test suite for ModelLoader."""
    
    def test_list_models(self):
        """Test listing available models."""
        loader = ModelLoader()
        models = loader.list_available_models()
        assert isinstance(models, list)
    
    def test_invalid_model_name(self):
        """Test handling of invalid model name."""
        loader = ModelLoader()
        with pytest.raises(ValueError):
            loader.download_model("invalid_model.pt")