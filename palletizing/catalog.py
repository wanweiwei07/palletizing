from __future__ import annotations

from dataclasses import asdict, dataclass


HIGH_FREQUENCY_BOX_DIMS = (
    (0.16, 0.12, 0.08),
    (0.18, 0.14, 0.10),
    (0.20, 0.15, 0.10),
    (0.20, 0.16, 0.12),
    (0.22, 0.16, 0.14),
    (0.24, 0.18, 0.12),
    (0.24, 0.20, 0.16),
    (0.26, 0.18, 0.14),
    (0.28, 0.20, 0.16),
    (0.30, 0.22, 0.18),
    (0.30, 0.24, 0.18),
    (0.32, 0.24, 0.20),
    (0.34, 0.24, 0.18),
    (0.34, 0.26, 0.20),
    (0.36, 0.28, 0.20),
    (0.38, 0.28, 0.22),
    (0.40, 0.30, 0.22),
    (0.42, 0.30, 0.24),
    (0.44, 0.32, 0.24),
    (0.45, 0.35, 0.25),
)

MEDIUM_FREQUENCY_BOX_DIMS = (
    (0.25, 0.20, 0.10),
    (0.27, 0.22, 0.12),
    (0.29, 0.23, 0.14),
    (0.31, 0.24, 0.16),
    (0.33, 0.25, 0.18),
    (0.35, 0.26, 0.20),
    (0.37, 0.28, 0.18),
    (0.39, 0.29, 0.20),
    (0.41, 0.30, 0.18),
    (0.43, 0.31, 0.20),
    (0.45, 0.32, 0.22),
    (0.47, 0.33, 0.24),
    (0.49, 0.35, 0.24),
    (0.50, 0.36, 0.26),
    (0.52, 0.38, 0.28),
    (0.54, 0.40, 0.28),
    (0.55, 0.42, 0.30),
    (0.56, 0.38, 0.32),
    (0.58, 0.40, 0.34),
    (0.60, 0.45, 0.35),
)

LOW_FREQUENCY_BOX_DIMS = (
    (0.32, 0.20, 0.10),
    (0.34, 0.22, 0.12),
    (0.36, 0.24, 0.14),
    (0.38, 0.26, 0.16),
    (0.40, 0.28, 0.18),
    (0.42, 0.30, 0.20),
    (0.44, 0.32, 0.18),
    (0.46, 0.34, 0.20),
    (0.48, 0.36, 0.22),
    (0.50, 0.38, 0.24),
    (0.52, 0.40, 0.26),
    (0.54, 0.42, 0.28),
    (0.56, 0.44, 0.30),
    (0.58, 0.46, 0.32),
    (0.60, 0.48, 0.34),
    (0.62, 0.40, 0.20),
    (0.64, 0.42, 0.22),
    (0.66, 0.44, 0.24),
    (0.68, 0.46, 0.26),
    (0.70, 0.50, 0.30),
)

DEFAULT_FREQUENCY_WEIGHTS = {
    "high": 0.6,
    "medium": 0.3,
    "low": 0.1,
}


@dataclass(frozen=True)
class BoxType:
    box_type_id: str
    length: float
    width: float
    height: float
    frequency_group: str
    sampling_weight: float
    allowed_yaws: tuple[float, float] = (0.0, 1.5707963267948966)

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height

    def as_dict(self) -> dict:
        payload = asdict(self)
        payload["volume"] = self.volume
        return payload


def _build_group(prefix: str, dims_group: tuple[tuple[float, float, float], ...], frequency_group: str, weight: float) -> list[BoxType]:
    return [
        BoxType(
            box_type_id=f"{prefix}{index:02d}",
            length=length,
            width=width,
            height=height,
            frequency_group=frequency_group,
            sampling_weight=weight,
        )
        for index, (length, width, height) in enumerate(dims_group)
    ]


def build_box_type_catalog() -> list[BoxType]:
    return [
        *_build_group("H", HIGH_FREQUENCY_BOX_DIMS, "high", DEFAULT_FREQUENCY_WEIGHTS["high"]),
        *_build_group("M", MEDIUM_FREQUENCY_BOX_DIMS, "medium", DEFAULT_FREQUENCY_WEIGHTS["medium"]),
        *_build_group("L", LOW_FREQUENCY_BOX_DIMS, "low", DEFAULT_FREQUENCY_WEIGHTS["low"]),
    ]


BOX_TYPE_CATALOG = tuple(build_box_type_catalog())
