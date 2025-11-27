"""Video frame extraction for embedding."""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image


class FrameSelectionMethod(Enum):
    """Methods for selecting frames from video."""

    UNIFORM = "uniform"  # Evenly spaced frames
    KEYFRAME = "keyframe"  # Scene change detection
    MOTION = "motion"  # High motion frames
    REPRESENTATIVE = "representative"  # Cluster representatives


@dataclass
class VideoFrame:
    """A frame extracted from video."""

    image: Image.Image
    frame_number: int
    timestamp: float  # Seconds
    metadata: dict = field(default_factory=dict)


@dataclass
class VideoEmbedding:
    """Embedding for a video frame."""

    embedding: np.ndarray
    frame_number: int
    timestamp: float
    video_id: str = ""
    metadata: dict = field(default_factory=dict)


class VideoFrameExtractor:
    """Extract frames from video for embedding."""

    def __init__(
        self,
        method: FrameSelectionMethod = FrameSelectionMethod.UNIFORM,
        target_frames: int = 10,
        min_frame_interval: float = 0.5,  # Minimum seconds between frames
    ):
        """
        Initialize frame extractor.

        Args:
            method: Frame selection method
            target_frames: Target number of frames to extract
            min_frame_interval: Minimum interval between frames in seconds
        """
        self.method = method
        self.target_frames = target_frames
        self.min_frame_interval = min_frame_interval

    def extract_frames(
        self,
        frames: list[np.ndarray],
        fps: float = 30.0,
    ) -> list[VideoFrame]:
        """
        Extract representative frames from video.

        Args:
            frames: List of video frames as numpy arrays
            fps: Video frame rate

        Returns:
            List of selected VideoFrame objects
        """
        total_frames = len(frames)
        duration = total_frames / fps

        if self.method == FrameSelectionMethod.UNIFORM:
            selected_indices = self._uniform_selection(total_frames)
        elif self.method == FrameSelectionMethod.KEYFRAME:
            selected_indices = self._keyframe_selection(frames)
        elif self.method == FrameSelectionMethod.MOTION:
            selected_indices = self._motion_selection(frames)
        elif self.method == FrameSelectionMethod.REPRESENTATIVE:
            selected_indices = self._representative_selection(frames)
        else:
            selected_indices = self._uniform_selection(total_frames)

        # Convert to VideoFrame objects
        video_frames = []
        for idx in selected_indices:
            if idx < len(frames):
                frame_array = frames[idx]
                image = Image.fromarray(frame_array)

                video_frames.append(
                    VideoFrame(
                        image=image,
                        frame_number=idx,
                        timestamp=idx / fps,
                        metadata={
                            "total_frames": total_frames,
                            "duration": duration,
                            "selection_method": self.method.value,
                        },
                    )
                )

        return video_frames

    def _uniform_selection(self, total_frames: int) -> list[int]:
        """Select uniformly spaced frames."""
        if total_frames <= self.target_frames:
            return list(range(total_frames))

        # Calculate interval
        interval = total_frames / self.target_frames
        indices = [int(i * interval) for i in range(self.target_frames)]

        return indices

    def _keyframe_selection(self, frames: list[np.ndarray]) -> list[int]:
        """Select frames at scene changes."""
        if len(frames) <= self.target_frames:
            return list(range(len(frames)))

        # Compute frame differences
        differences = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i - 1].astype(float)))
            differences.append((i, diff))

        # Sort by difference (highest = scene change)
        differences.sort(key=lambda x: x[1], reverse=True)

        # Select top scene changes
        selected = [0]  # Always include first frame
        for idx, _ in differences:
            if len(selected) >= self.target_frames:
                break

            # Check minimum interval
            too_close = any(abs(idx - s) < self.min_frame_interval * 30 for s in selected)
            if not too_close:
                selected.append(idx)

        return sorted(selected)

    def _motion_selection(self, frames: list[np.ndarray]) -> list[int]:
        """Select frames with high motion."""
        if len(frames) <= self.target_frames:
            return list(range(len(frames)))

        # Compute motion scores
        motion_scores = []
        for i in range(1, len(frames) - 1):
            # Motion = difference between consecutive frames
            diff_prev = np.mean(np.abs(frames[i].astype(float) - frames[i - 1].astype(float)))
            diff_next = np.mean(np.abs(frames[i + 1].astype(float) - frames[i].astype(float)))
            motion = (diff_prev + diff_next) / 2
            motion_scores.append((i, motion))

        # Sort by motion
        motion_scores.sort(key=lambda x: x[1], reverse=True)

        # Select high-motion frames with spacing
        selected = []
        for idx, _ in motion_scores:
            if len(selected) >= self.target_frames:
                break

            too_close = any(abs(idx - s) < 15 for s in selected)  # Min 15 frames apart
            if not too_close:
                selected.append(idx)

        # Add first and last if not present
        if 0 not in selected:
            selected.append(0)
        if len(frames) - 1 not in selected:
            selected.append(len(frames) - 1)

        return sorted(selected)[: self.target_frames]

    def _representative_selection(self, frames: list[np.ndarray]) -> list[int]:
        """Select representative frames via clustering."""
        if len(frames) <= self.target_frames:
            return list(range(len(frames)))

        # Simple k-means-like selection
        # Compute mean color per frame as simple feature
        features = []
        for frame in frames:
            mean_color = np.mean(frame, axis=(0, 1))
            features.append(mean_color)
        features = np.array(features)

        # Select diverse frames using greedy farthest point
        selected = [0]  # Start with first frame

        while len(selected) < self.target_frames:
            # Find frame farthest from all selected frames
            max_min_dist = -1
            best_idx = -1

            for i in range(len(frames)):
                if i in selected:
                    continue

                # Minimum distance to any selected frame
                min_dist = min(np.linalg.norm(features[i] - features[s]) for s in selected)

                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i

            if best_idx >= 0:
                selected.append(best_idx)
            else:
                break

        return sorted(selected)


class VideoEmbedder:
    """Generate embeddings for video frames."""

    def __init__(
        self,
        encoder,
        extractor: VideoFrameExtractor | None = None,
        target_size: int = 224,
    ):
        """
        Initialize video embedder.

        Args:
            encoder: Image embedding model
            extractor: Frame extractor (optional)
            target_size: Model input size
        """
        self.encoder = encoder
        self.extractor = extractor or VideoFrameExtractor()
        self.target_size = target_size

    def embed_video(
        self,
        frames: list[np.ndarray],
        video_id: str = "",
        fps: float = 30.0,
    ) -> list[VideoEmbedding]:
        """
        Extract and embed video frames.

        Args:
            frames: Video frames as numpy arrays
            video_id: Video identifier
            fps: Frame rate

        Returns:
            List of VideoEmbedding objects
        """
        # Extract representative frames
        video_frames = self.extractor.extract_frames(frames, fps)

        # Generate embeddings
        embeddings = []
        for vf in video_frames:
            # Resize for model
            resized = vf.image.resize(
                (self.target_size, self.target_size),
                Image.Resampling.LANCZOS,
            )

            embedding = self.encoder.encode(resized)

            embeddings.append(
                VideoEmbedding(
                    embedding=embedding,
                    frame_number=vf.frame_number,
                    timestamp=vf.timestamp,
                    video_id=video_id,
                    metadata=vf.metadata,
                )
            )

        return embeddings


def aggregate_video_embeddings(
    embeddings: list[VideoEmbedding],
    method: str = "mean",
) -> np.ndarray:
    """
    Aggregate frame embeddings to video-level embedding.

    Args:
        embeddings: List of frame embeddings
        method: Aggregation method ('mean', 'max', 'first_last')

    Returns:
        Video-level embedding
    """
    vectors = np.array([e.embedding for e in embeddings])

    if method == "mean":
        return np.mean(vectors, axis=0)
    elif method == "max":
        return np.max(vectors, axis=0)
    elif method == "first_last":
        # Concatenate first and last frame embeddings
        return np.concatenate([vectors[0], vectors[-1]])
    else:
        raise ValueError(f"Unknown aggregation method: {method}")
