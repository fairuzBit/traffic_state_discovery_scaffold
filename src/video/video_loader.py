"""Module: video_loader.py"""
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import cv2
# TODO: implement according to research pipeline
class VideoLoader:
    """
    Video loader utility for CCTV processing.

    Responsibilities:
    - Open video file
    - Extract metadata
    - Read frames sequentially
    - Provide frame generator
    """

    def __init__(self, video_path: str):
        """
        Parameters
        ----------
        video_path : str
            Path to CCTV video.
        """
        video_path
        self.video_path = Path(video_path)

        if not self.video_path.exists():
            raise FileNotFoundError(
                f"Video not found: {self.video_path}"
            )

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Unable to open video: {self.video_path}"
            )

        self.video_info = self._extract_metadata()

    # --------------------------------------------------
    # Metadata
    # --------------------------------------------------

    def _extract_metadata(self) -> Dict:
        """
        Extract video metadata.

        Returns
        -------
        Dict
        """

        fps = self.cap.get(cv2.CAP_PROP_FPS)

        frame_count = int(
            self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )

        duration_sec = (
            frame_count / fps
            if fps > 0
            else 0
        )

        return {
            "video_name": self.video_path.name,
            "video_path": str(self.video_path),
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_seconds": duration_sec,
            "duration_minutes": duration_sec / 60,
        }

    # --------------------------------------------------
    # Public Methods
    # --------------------------------------------------

    def get_video_info(self) -> Dict:
        """
        Return extracted metadata.
        """

        return self.video_info

    def read_frame(
        self
    ) -> Tuple[bool, Optional[cv2.typing.MatLike]]:
        """
        Read single frame.

        Returns
        -------
        success : bool
        frame : ndarray
        """

        success, frame = self.cap.read()

        return success, frame

    def frame_generator(
        self
    ) -> Generator[
        Tuple[int, cv2.typing.MatLike],
        None,
        None
    ]:
        """
        Yield frames sequentially.

        Yields
        ------
        frame_idx
        frame
        """

        frame_idx = 0

        while True:

            success, frame = self.cap.read()

            if not success:
                break

            yield frame_idx, frame

            frame_idx += 1

    def set_frame_position(
        self,
        frame_number: int
    ) -> None:
        """
        Jump to specific frame.

        Useful for ROI selection.
        """

        self.cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number
        )

    def get_current_timestamp(
        self
    ) -> float:
        """
        Current timestamp in seconds.
        """

        return (
            self.cap.get(
                cv2.CAP_PROP_POS_MSEC
            ) / 1000.0
        )

    def release(self):
        """
        Release video resource.
        """

        if self.cap is not None:
            self.cap.release()

    def __del__(self):
        """
        Auto release.
        """

        self.release()