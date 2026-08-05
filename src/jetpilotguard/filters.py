"""Signal-conditioning filters for noisy sensor streams."""

from __future__ import annotations


class KalmanFilter1D:
    """Scalar Kalman filter for smoothing a single noisy sensor channel.

    This is the standard 1-D constant-position model: we assume the true value
    drifts slowly (governed by ``process_variance``) and each measurement is
    corrupted by noise (``measurement_variance``). Increasing the process
    variance makes the filter trust new measurements more (faster, noisier);
    increasing the measurement variance makes it smoother but laggier.

    Args:
        process_variance: Q, expected variance of the underlying true value
            between steps.
        measurement_variance: R, expected variance of sensor noise.
    """

    def __init__(
        self,
        process_variance: float = 1e-3,
        measurement_variance: float = 1e-1,
    ) -> None:
        if process_variance <= 0 or measurement_variance <= 0:
            raise ValueError("variances must be positive")
        self._q = process_variance
        self._r = measurement_variance
        self._x = 0.0          # posterior state estimate
        self._p = 1.0          # posterior error covariance
        self._initialised = False

    @property
    def value(self) -> float:
        """Current state estimate."""
        return self._x

    def reset(self) -> None:
        """Return the filter to its uninitialised state."""
        self._x = 0.0
        self._p = 1.0
        self._initialised = False

    def update(self, measurement: float) -> float:
        """Incorporate one measurement and return the new estimate.

        The first measurement seeds the state directly (avoids a long
        convergence transient from the arbitrary initial estimate).
        """
        if not self._initialised:
            self._x = measurement
            self._initialised = True
            return self._x

        # Predict: state is assumed static, covariance grows by process noise.
        self._p += self._q

        # Update: blend prediction with measurement via the Kalman gain.
        k = self._p / (self._p + self._r)
        self._x += k * (measurement - self._x)
        self._p *= (1.0 - k)

        return self._x
