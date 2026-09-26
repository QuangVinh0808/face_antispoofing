"""Protocol-safe evaluation helpers for Stage 4."""

from .protocol import (
    aggregate_frame_rows,
    attach_video_results,
    evaluate_rows,
    tune_development_policy,
)

__all__ = [
    "aggregate_frame_rows",
    "attach_video_results",
    "evaluate_rows",
    "tune_development_policy",
]
