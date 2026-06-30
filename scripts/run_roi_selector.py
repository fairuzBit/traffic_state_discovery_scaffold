#!/usr/bin/env python3
"""
Interactive ROI selection tool.
Usage: python scripts/run_roi_selector.py --video path/to/video.mp4
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.roi.roi_selector import ROISelector
from src.roi.roi_manager import ROIManager
from src.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI Selection Tool"
    )
    
    parser.add_argument(
        '--video', '-v',
        type=str,
        required=True,
        help='Path to video file'
    )
    
    parser.add_argument(
        '--frame', '-f',
        type=int,
        default=0,
        help='Frame number to use for selection'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='configs/roi_config.json',
        help='Output path for ROI configuration'
    )
    
    args = parser.parse_args()
    
    video_path = Path(args.video)
    
    if not video_path.exists():
        logger.error(f"Video not found: {video_path}")
        sys.exit(1)
    
    logger.info("Starting ROI Selector...")
    logger.info("Instructions:")
    logger.info("  - LEFT CLICK: Add polygon point")
    logger.info("  - RIGHT CLICK: Complete current polygon")
    logger.info("  - C: Clear current polygon")
    logger.info("  - R: Reset all ROIs")
    logger.info("  - ENTER: Save and exit")
    logger.info("  - Q: Quit without saving")
    
    selector = ROISelector()
    rois = selector.select_roi_from_video(video_path, args.frame)
    
    if rois:
        # Save ROIs
        roi_manager = ROIManager()
        for i, roi_points in enumerate(rois):
            roi_manager.add_roi(f"ROI_{i+1}", roi_points)
        
        roi_manager.save_roi(Path(args.output))
        logger.info(f"✅ Saved {len(rois)} ROIs to {args.output}")
    else:
        logger.info("No ROIs selected")


if __name__ == "__main__":
    main()