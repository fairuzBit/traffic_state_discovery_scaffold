"""
Unit tests for vehicle tracking module.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.config import TrackingConfig
from src.tracking.track_manager import Track, TrackManager
from src.tracking.byte_tracker import ByteTrackTracker
from src.detection.detector import Detection


class TestTrackManager:
    """Test suite for TrackManager."""
    
    @pytest.fixture
    def track_manager(self):
        """Create TrackManager instance."""
        return TrackManager(max_missed_frames=5, min_track_length=3)
    
    @pytest.fixture
    def sample_detection(self):
        """Create sample detection."""
        return Detection(
            bbox=(100, 100, 200, 200),
            confidence=0.9,
            class_id=2,
            class_name='car'
        )
    
    def test_create_track(self, track_manager, sample_detection):
        """Test track creation."""
        track = track_manager.create_track(sample_detection, 1)
        
        assert track.track_id == 1
        assert track.class_name == 'car'
        assert track.is_active
        assert len(track.positions) == 1
    
    def test_update_track(self, track_manager, sample_detection):
        """Test track updating."""
        track = track_manager.create_track(sample_detection, 1)
        
        new_detection = Detection((110, 110, 210, 210), 0.95, 2, 'car')
        track_manager.update_track(track.track_id, new_detection, 2, velocity=5.0)
        
        assert len(track.positions) == 2
        assert len(track.velocities) == 1
        assert track.velocities[0] == 5.0
    
    def test_deactivate_stale(self, track_manager, sample_detection):
        """Test deactivation of stale tracks."""
        track = track_manager.create_track(sample_detection, 1)
        assert track.is_active
        
        deactivated = track_manager.deactivate_stale_tracks(10)
        assert len(deactivated) == 1
        assert not track.is_active
    
    def test_valid_tracks(self, track_manager, sample_detection):
        """Test filtering of valid tracks."""
        # Create track with insufficient length
        track1 = track_manager.create_track(sample_detection, 1)
        
        # Create track with sufficient length
        track2 = track_manager.create_track(sample_detection, 1)
        for i in range(5):
            track_manager.update_track(
                track2.track_id,
                Detection((100+i, 100+i, 200+i, 200+i), 0.9, 2, 'car'),
                i + 2
            )
        
        valid = track_manager.get_valid_tracks()
        assert len(valid) == 1
        assert valid[0].track_id == track2.track_id


class TestTrack:
    """Test suite for Track dataclass."""
    
    def test_track_properties(self):
        """Test track property calculations."""
        track = Track(
            track_id=1,
            class_id=2,
            class_name='car',
            start_frame=1,
            last_frame=10
        )
        
        assert track.age == 9
        assert track.length == 0
    
    def test_total_distance(self):
        """Test distance calculation."""
        track = Track(
            track_id=1,
            class_id=2,
            class_name='car',
            start_frame=1,
            last_frame=10
        )
        
        # Add positions forming a path
        detections = [
            Detection((0, 0, 10, 10), 0.9, 2, 'car'),
            Detection((3, 4, 13, 14), 0.9, 2, 'car'),
        ]
        
        for det in detections:
            track.add_detection(det, 1)
        
        # Distance should be sqrt(3^2 + 4^2) = 5
        assert abs(track.total_distance - 5.0) < 0.01