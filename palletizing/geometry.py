from __future__ import annotations

from math import pi

TOL = 1e-9


def oriented_size(length: float, width: float, yaw: float) -> tuple[float, float]:
    """Return (size_x, size_y) after applying a yaw rotation in the pallet plane."""
    if abs(yaw - pi / 2) < TOL:
        return width, length
    return length, width


def normalized_allowed_yaws(
    length: float, width: float, allowed_yaws: tuple[float, ...]
) -> tuple[float, ...]:
    """Return deduplicated yaw options, dropping pi/2 when the box is square."""
    yaws = [0.0]
    if abs(length - width) > TOL and any(abs(yaw - pi / 2) < TOL for yaw in allowed_yaws):
        yaws.append(pi / 2)
    return tuple(yaws)


def rect_corners(size_x: float, size_y: float) -> list[tuple[float, float]]:
    """Return the four corners of a rectangle anchored at origin."""
    return [
        (0.0, 0.0),
        (size_x, 0.0),
        (size_x, size_y),
        (0.0, size_y),
    ]


def bottom_alignment_corners(
    local_corners: list[tuple[float, float]], anchor_index: int
) -> list[tuple[float, float]]:
    """Return the 3 corners that are *not* at anchor_index (used for bottom-on-bottom alignment)."""
    return [corner for index, corner in enumerate(local_corners) if index != anchor_index]
