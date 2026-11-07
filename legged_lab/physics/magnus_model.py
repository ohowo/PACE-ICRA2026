# SPDX-License-Identifier: MIT
"""Speed/spin-dependent Magnus coefficient for a nominal 40 mm ball."""

import torch


class ContiMagnusModel:
    """Evaluate kappa = 8/3 C_M from Conti et al., arXiv:2606.28805v2.

    Eq. (5), Tables II-III supply curves at six speeds. PACE evaluates the
    two neighboring curves at current spin magnitude and interpolates C_M
    linearly in speed. Input units are m/s and rad/s. Speed is clamped to
    [2, 17] m/s; spin used for coefficient lookup to [0, 1000] rad/s.
    The latter is a PACE guard at the paper's stated spin envelope. Actual
    velocity/spin vectors still enter the force; outside this envelope that
    force is an unvalidated extension, not a fitted prediction.
    """

    # speed, linear slope, intercept, spin breakpoint, quadratic a, b, c
    _ROWS = (
        (2.0, 0.0, 0.08, 150.0, -1.852e-7, -1.296e-4, 0.0983),
        (3.5, -1.1e-3, 0.31, 200.0, -1.667e-7, -3.333e-5, 0.1),
        (7.5, -8.0e-4, 0.37, 350.0, -2.000e-7, 1.700e-4, 0.0587),
        (10.5, -6.58e-4, 0.375, 440.0, -2.604e-7, 3.646e-4, -0.0225),
        (13.5, -5.6e-4, 0.383, 550.0, -3.571e-7, 5.357e-4, -0.0893),
        (17.0, -4.48e-4, 0.371, 650.0, -1.000e-7, 2.300e-4, -0.0375),
    )

    def __init__(self):
        self._table = None
        self._speeds = None

    @staticmethod
    def _curve(rows, spin):
        linear = rows[..., 1] * spin + rows[..., 2]
        quadratic = (rows[..., 4] * spin + rows[..., 5]) * spin + rows[..., 6]
        return torch.where(spin <= rows[..., 3], linear, quadratic).clamp_min(0.0)

    def __call__(self, speed: torch.Tensor, spin: torch.Tensor) -> torch.Tensor:
        """Return per-ball kappa, preserving speed's shape, device and dtype."""
        if self._table is None or self._table.device != speed.device or self._table.dtype != speed.dtype:
            self._table = speed.new_tensor(self._ROWS)
            self._speeds = self._table[:, 0].contiguous()
        bounded_speed = speed.clamp(2.0, 17.0)
        bounded_spin = spin.clamp(0.0, 1000.0)
        upper = torch.searchsorted(self._speeds, bounded_speed.contiguous(), right=True).clamp(1, 5)
        lower = upper - 1
        lo, hi = self._table[lower], self._table[upper]
        weight = (bounded_speed - lo[..., 0]) / (hi[..., 0] - lo[..., 0])
        cm = torch.lerp(self._curve(lo, bounded_spin), self._curve(hi, bounded_spin), weight)
        return (8.0 / 3.0) * cm
