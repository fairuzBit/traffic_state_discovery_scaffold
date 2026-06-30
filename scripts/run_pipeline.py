#!/usr/bin/env python3
"""
Main execution script for Traffic State Discovery Pipeline.
Usage: python scripts/run_pipeline.py --video path/to/video.mp4
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ProjectConfig
from src.pipeline import TrafficStatePipeline
from src.logger import logger


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Unsupervised Traffic State Discovery Pipeline"
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to input video file'
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='configs/default_config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--no-roi',
        action='store_true',
        help='Skip interactive ROI selection'
    )
    
    parser.add_argument(
        '--max-frames', '-n',
        type=int,
        default=None,
        help='Maximum number of frames to process'
    )
    
    parser.add_argument(
        '--no-grid-search',
        action='store_true',
        help='Skip grid search for clustering parameters'
    )
    
    parser.add_argument(
        '--device', '-d',
        type=str,
        default='cuda:0',
        help='Device for inference (cuda:0, cpu)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = ProjectConfig()
    
    # Override device if specified
    config.detection.device = args.device
    
    # If config file exists, load it
    config_path = Path(args.config)
    if config_path.exists():
        config = ProjectConfig.from_yaml(str(config_path))
        config.detection.device = args.device
    
    # Create pipeline
    pipeline = TrafficStatePipeline(config)
    
    # Run pipeline
    video_path = Path(args.video)
    
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
    
    try:
        results = pipeline.run_full_pipeline(
            video_path=video_path,
            select_roi=not args.no_roi,
            max_frames=args.max_frames,
            run_grid_search=not args.no_grid_search
        )
        
        logger.info("\n✅ Pipeline completed successfully!")
        logger.info(f"📊 Results saved to: {config.paths.outputs}")
        logger.info(f"📄 Paper materials saved to: {config.paths.paper}")
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()