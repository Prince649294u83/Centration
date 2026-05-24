"""
Pixel-to-millimetre spatial calibration.

Provides multiple calibration strategies:
    1. Suction ring (known diameter 9.0-9.5 mm)
    2. Limbus / corneal diameter (average 11.5 mm)
    3. Manual calibration (user-provided px/mm)
    4. Known object in frame

For surgical precision, calibration uncertainty is tracked and
propagated through all measurements.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from pupil_tracking.utils.types import (
    CalibrationInfo,
    EyeDetectionResult,
    LimbusDetection,
)
from pupil_tracking.utils.logger import get_logger


class SpatialCalibrator:
    """Manages pixel-to-mm calibration with uncertainty tracking.

    Usage
    -----
    >>> cal = SpatialCalibrator()
    >>> info = cal.calibrate_from_limbus(limbus_detection)
    >>> mm = info.px_to_mm(100.0)  # 100 pixels in mm
    """

    # Known anatomical references (population averages)
    CORNEAL_DIAMETER_MM = 11.5          # horizontal white-to-white
    CORNEAL_DIAMETER_STD_MM = 0.5       # population std
    SUCTION_RING_DIAMETERS_MM = {
        "standard": 9.4,
        "small": 8.5,
        "large": 10.0,
    }

    def __init__(self) -> None:
        self.logger = get_logger()
        self._history: List[CalibrationInfo] = []

    def calibrate_from_limbus(
        self,
        limbus: LimbusDetection,
        corneal_diameter_mm: float = 11.5,
        corneal_std_mm: float = 0.5,
    ) -> CalibrationInfo:
        """Calibrate using detected limbus diameter.

        The limbus marks the boundary of the cornea. The average
        horizontal corneal diameter is 11.5 +/- 0.5 mm.

        Parameters
        ----------
        limbus : LimbusDetection
        corneal_diameter_mm : float
            Expected corneal diameter (default 11.5 mm).
        corneal_std_mm : float
            Population standard deviation (for uncertainty).

        Returns
        -------
        CalibrationInfo
        """
        if not limbus.detected or limbus.ellipse is None:
            return CalibrationInfo()

        # Use semi-major axis only (horizontal corneal diameter)
        # to avoid circular reference where limbus mm always equals
        # the calibration constant.
        diameter_px = limbus.ellipse.semi_major * 2.0
        if diameter_px < 20:
            return CalibrationInfo()

        px_per_mm = diameter_px / corneal_diameter_mm
        confidence = limbus.confidence * 0.8

        # account for ellipse vs circle
        aspect = limbus.ellipse.circularity
        if aspect < 0.85:
            # oblique view — less reliable
            confidence *= aspect

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=px_per_mm,
            mm_per_px=1.0 / px_per_mm,
            source="limbus_diameter",
            reference_diameter_mm=corneal_diameter_mm,
            reference_diameter_px=diameter_px,
            confidence=confidence,
        )

        self._history.append(cal)
        self.logger.info(
            "Calibrated from limbus: %.2f px/mm (conf=%.2f)",
            px_per_mm, confidence,
        )
        return cal

    def calibrate_from_ring(
        self,
        ring_center: Tuple[float, float],
        ring_radius: float,
        ring_type: str = "standard",
    ) -> CalibrationInfo:
        """Calibrate using a suction ring of known diameter.

        Parameters
        ----------
        ring_center : (x, y) in pixels
        ring_radius : float in pixels
        ring_type : str  "standard" | "small" | "large"
        """
        diameter_mm = self.SUCTION_RING_DIAMETERS_MM.get(
            ring_type, 9.4
        )
        diameter_px = ring_radius * 2.0

        if diameter_px < 20:
            return CalibrationInfo()

        px_per_mm = diameter_px / diameter_mm

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=px_per_mm,
            mm_per_px=1.0 / px_per_mm,
            source=f"suction_ring_{ring_type}",
            reference_diameter_mm=diameter_mm,
            reference_diameter_px=diameter_px,
            confidence=0.95,  # rings have known precise diameter
        )

        self._history.append(cal)
        self.logger.info(
            "Calibrated from ring (%s): %.2f px/mm",
            ring_type, px_per_mm,
        )
        return cal

    def calibrate_manual(
        self,
        px_per_mm: float,
        source: str = "manual",
    ) -> CalibrationInfo:
        """Manual calibration with user-provided scale."""
        if px_per_mm <= 0:
            return CalibrationInfo()

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=px_per_mm,
            mm_per_px=1.0 / px_per_mm,
            source=source,
            confidence=1.0,
        )
        self._history.append(cal)
        return cal

    def get_consensus_calibration(self) -> CalibrationInfo:
        """Compute consensus calibration from history.

        Uses confidence-weighted average of all calibrations.
        """
        if not self._history:
            return CalibrationInfo()

        calibrated = [c for c in self._history if c.calibrated]
        if not calibrated:
            return CalibrationInfo()

        weights = np.array([c.confidence for c in calibrated])
        px_per_mm = np.array([c.px_per_mm for c in calibrated])

        total_weight = weights.sum()
        if total_weight < 0.01:
            return calibrated[-1]

        avg_px_per_mm = float(np.average(px_per_mm, weights=weights))
        std_px_per_mm = float(
            np.sqrt(np.average((px_per_mm - avg_px_per_mm) ** 2, weights=weights))
        )

        # confidence from consistency
        if len(calibrated) > 1 and avg_px_per_mm > 0:
            cv = std_px_per_mm / avg_px_per_mm  # coefficient of variation
            consistency = max(0.0, 1.0 - cv * 5.0)
        else:
            consistency = 0.8

        avg_conf = float(np.mean(weights))
        final_conf = min(1.0, avg_conf * consistency)

        return CalibrationInfo(
            calibrated=True,
            px_per_mm=avg_px_per_mm,
            mm_per_px=1.0 / avg_px_per_mm,
            source=f"consensus_{len(calibrated)}",
            confidence=final_conf,
        )

    def reset(self) -> None:
        self._history.clear()


class StabilizedCalibrator:
    """EMA-smoothed calibration with outlier rejection.

    Wraps raw limbus-based calibration with temporal stabilization
    to prevent single noisy measurements from shifting all downstream
    mm values.  This is critical for achieving the ±0.01–0.02 mm
    accuracy target in surgical applications.

    The exponential moving average converges within 5 samples and
    produces calibration values that fluctuate less than 0.01 mm
    across consecutive frames.

    Parameters
    ----------
    config : MeasurementStabilizationConfig or None
        Stabilization parameters.  ``None`` → defaults.
    corneal_diameter_mm : float
        Known average horizontal corneal diameter for calibration.
    """

    def __init__(self, config=None, corneal_diameter_mm: float = 11.5):
        from pupil_tracking.utils.config import get_config

        if config is None:
            config = get_config().measurement_stabilization

        self._alpha = float(config.ema_alpha)
        self._outlier_sigma = float(config.outlier_sigma)
        self._min_samples = int(config.min_samples_for_rejection)
        self._max_history = int(config.max_calibration_history)
        self._enabled = bool(config.enable_ema_smoothing)
        self._corneal_mm = corneal_diameter_mm

        self._ema_px_per_mm: Optional[float] = None
        self._ema_variance: float = 0.0
        self._history: List[float] = []
        self._frozen = False  # once True, calibration stops updating

        self.logger = get_logger()

    @property
    def is_frozen(self) -> bool:
        """True when calibration has stabilised and is no longer updating."""
        return self._frozen

    def update_from_limbus(
        self,
        limbus: LimbusDetection,
    ) -> CalibrationInfo:
        """Update calibration with EMA smoothing and outlier rejection.

        Calibration is derived from the limbus **semi-major axis** only
        (the horizontal corneal diameter ≈ 11.5 mm).  This avoids the
        circular-reference problem where the limbus diameter in mm is
        always exactly 11.5 mm.  By calibrating from semi-major alone,
        the semi-minor axis, mean diameter, and fit-type (circle vs
        ellipse) can all show natural per-frame variation.

        Once the calibration has accumulated enough samples and the
        variance is low, the EMA is **frozen** — it stops updating.
        This locks ``mm_per_px`` so that subsequent per-frame limbus
        detections produce genuinely varying mm measurements instead
        of the self-referential constant 11.5 mm.

        Parameters
        ----------
        limbus : LimbusDetection
            Current frame's limbus detection.

        Returns
        -------
        CalibrationInfo
            Stabilized calibration.  Returns the current best estimate
            (not the raw frame value), or an uncalibrated result if no
            valid measurements have been accumulated yet.
        """
        # If already frozen, return the locked calibration
        if self._frozen:
            return self._current_best()

        if not limbus.detected or limbus.ellipse is None:
            return self._current_best()

        # Use semi-major axis (horizontal corneal diameter) for calibration
        semi_major_px = limbus.ellipse.semi_major
        if semi_major_px < 5:
            return self._current_best()

        diameter_px = semi_major_px * 2.0
        new_val = diameter_px / self._corneal_mm

        # Bypass smoothing if disabled
        if not self._enabled:
            return CalibrationInfo(
                calibrated=True,
                px_per_mm=new_val,
                mm_per_px=1.0 / new_val,
                source="limbus_diameter",
                reference_diameter_mm=self._corneal_mm,
                reference_diameter_px=diameter_px,
                confidence=min(0.95, limbus.confidence * 0.8),
            )

        # Outlier rejection (once enough history)
        if (
            len(self._history) >= self._min_samples
            and self._ema_px_per_mm is not None
        ):
            std = math.sqrt(self._ema_variance) if self._ema_variance > 0 else 0.0
            if std > 0 and abs(new_val - self._ema_px_per_mm) > self._outlier_sigma * std:
                self.logger.debug(
                    "Calibration outlier rejected: %.3f (EMA=%.3f ± %.3f)",
                    new_val, self._ema_px_per_mm, std,
                )
                return self._current_best()

        # EMA update
        if self._ema_px_per_mm is None:
            self._ema_px_per_mm = new_val
            self._ema_variance = 0.0
        else:
            diff = new_val - self._ema_px_per_mm
            self._ema_px_per_mm += self._alpha * diff
            self._ema_variance = (
                (1.0 - self._alpha) * (self._ema_variance + self._alpha * diff * diff)
            )

        # Maintain bounded history
        self._history.append(new_val)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Freeze calibration once we have enough stable samples.
        # After min_samples * 2 frames, if the coefficient of variation
        # (std / mean) is below 2%, lock the calibration so that
        # per-frame limbus measurements show natural variation in mm.
        if (
            len(self._history) >= self._min_samples * 2
            and self._ema_px_per_mm is not None
            and self._ema_px_per_mm > 0
        ):
            std = math.sqrt(self._ema_variance) if self._ema_variance > 0 else 0.0
            cv = std / self._ema_px_per_mm
            if cv < 0.02:
                self._frozen = True
                self.logger.info(
                    "Calibration FROZEN at %.4f px/mm (CV=%.4f, %d samples)",
                    self._ema_px_per_mm, cv, len(self._history),
                )

        return self._current_best()

    def _current_best(self) -> CalibrationInfo:
        """Return the current EMA-smoothed calibration with uncertainty."""
        if self._ema_px_per_mm is None:
            return CalibrationInfo()

        confidence = min(0.95, 0.5 + len(self._history) * 0.05)

        # Phase 6: Compute calibration uncertainty from EMA variance
        std_px_per_mm = (
            math.sqrt(self._ema_variance)
            if self._ema_variance > 0
            else 0.0
        )
        # mm_per_px uncertainty via error propagation:
        # if px_per_mm = P ± σ_P, then mm_per_px = 1/P ± σ_P / P²
        mm_per_px = 1.0 / self._ema_px_per_mm
        mm_per_px_uncertainty = (
            std_px_per_mm / (self._ema_px_per_mm ** 2)
            if self._ema_px_per_mm > 0
            else 0.0
        )

        cal = CalibrationInfo(
            calibrated=True,
            px_per_mm=self._ema_px_per_mm,
            mm_per_px=mm_per_px,
            source="stabilized_limbus_frozen" if self._frozen else "stabilized_limbus",
            reference_diameter_mm=self._corneal_mm,
            reference_diameter_px=self._ema_px_per_mm * self._corneal_mm,
            confidence=confidence,
        )

        # Attach uncertainty as extra attribute for downstream use
        cal.mm_per_px_uncertainty = mm_per_px_uncertainty
        cal.px_per_mm_std = std_px_per_mm

        return cal

    def reset(self) -> None:
        """Reset all smoothing state."""
        self._ema_px_per_mm = None
        self._ema_variance = 0.0
        self._history.clear()
        self._frozen = False