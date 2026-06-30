"""
Unit tests for feature extraction module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from src.config import FeatureExtractionConfig
from src.features.feature_extractor import FeatureExtractor
from src.features.speed_estimator import SpeedEstimator
from src.features.density_calculator import DensityCalculator
from src.features.flow_analyzer import FlowAnalyzer
from src.roi.roi_manager import ROIManager
from src.tracking.track_manager import Track


class TestSpeedEstimator:
    """Test suite for SpeedEstimator."""
    
    @pytest.fixture
    def estimator(self):
        """Create SpeedEstimator."""
        return SpeedEstimator(pixel_to_meter_ratio=0.05, frame_rate=30.0)
    
    def test_speed_calculation(self, estimator):
        """Test speed estimation."""
        track = Track(
            track_id=1,
            class_id=2,
            class_name='car',
            start_frame=1,
            last_frame=10
        )
        
        # Add positions with known displacement
        from src.detection.detector import Detection
        
        det1 = Detection((0, 0, 10, 10), 0.9, 2, 'car')
        det2 = Detection((10, 0, 20, 10), 0.9, 2, 'car')
        
        track.add_detection(det1, 1)
        track.add_detection(det2, 2)
        
        speed = estimator.estimate_speed(track)
        
        assert speed is not None
        assert speed.speed_kmh > 0
        assert speed.confidence > 0


class TestDensityCalculator:
    """Test suite for DensityCalculator."""
    
    @pytest.fixture
    def calculator(self):
        """Create DensityCalculator."""
        return DensityCalculator(road_length=50.0, road_width=7.0)
    
    def test_density_calculation(self, calculator):
        """Test density calculation."""
        density = calculator.calculate_density(vehicle_count=10, roi_area_pixels=50000)
        assert density > 0
        
        # 10 vehicles over 50m = 200 vehicles/km
        expected = (10 / 50) * 1000
        assert abs(density - expected) < 0.1
    
    def test_level_of_service(self, calculator):
        """Test LOS classification."""
        assert calculator.get_level_of_service(5) == "A"
        assert calculator.get_level_of_service(10) == "B"
        assert calculator.get_level_of_service(15) == "C"
        assert calculator.get_level_of_service(20) == "D"
        assert calculator.get_level_of_service(25) == "E"
        assert calculator.get_level_of_service(35) == "F"


class TestFlowAnalyzer:
    """Test suite for FlowAnalyzer."""
    
    @pytest.fixture
    def analyzer(self):
        """Create FlowAnalyzer."""
        return FlowAnalyzer(queue_threshold_speed=5.0, jam_density=150.0)
    
    def test_queue_estimation(self, analyzer):
        """Test queue length estimation."""
        queue_length = analyzer.estimate_queue_length(
            stopped_vehicles=5,
            average_speed=2.0  # Below threshold
        )
        
        # 5 vehicles * (5m + 2m gap) = 35m
        assert queue_length == 35.0
    
    def test_no_queue_when_moving(self, analyzer):
        """Test no queue when vehicles are moving."""
        queue_length = analyzer.estimate_queue_length(
            stopped_vehicles=5,
            average_speed=50.0  # Above threshold
        )
        
        assert queue_length == 0.0