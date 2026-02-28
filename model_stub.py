from __future__ import annotations

import numpy as np


class _StubModel:
    """
    Replace this with your real model.

    Expected API: model.predict(x) where x is a (N,6) float array
    and returns a score in [1..5].
    """

    def predict(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != 6 or x.shape[0] < 10:
            return 3.0

        # Very simple heuristic so you can see the pipeline working:
        # higher motion energy => higher score.
        accel = x[:, 0:3]
        gyro = x[:, 3:6]
        energy = float(np.mean(np.linalg.norm(accel, axis=1)) + 0.25 * np.mean(np.linalg.norm(gyro, axis=1)))

        # Map energy to 1..5 (rough).
        if energy < 2.5:
            return 5.0
        if energy < 4.0:
            return 4.0
        if energy < 6.0:
            return 3.0
        if energy < 8.5:
            return 2.0
        return 1.0


model = _StubModel()

