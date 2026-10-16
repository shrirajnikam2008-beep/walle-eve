"""Bayesian Probabilistic Belief Tracker (Location & EVE Operational Status) + EIG Calculator.

Mathematical Model:
-------------------
1. Spatial Location Belief B_t(c) = P(EVE at c | z_{1:t}):
   - Uniform prior over all valid non-obstacle candidate cells.
   - Physical visit without discovery sets B_t(x) = 0 and renormalizes.
   - Radio scan reading `s` updates beliefs via Bayes' rule in log-space:
         log B_t(c) = log B_{t-1}(c) + log P(s | EVE at c, sensor at x)
     stabilized via log-sum-exp and normalized so sum_c B_t(c) = 1.0.

2. EVE Operational Status Belief P(Status | z_{1:t}):
   Maintains a normalized probability distribution over 3 candidate states:
   - `OPERATIONAL_SIGNAL`: Clean, steady beacon carrier.
   - `CRITICAL_LOW_POWER`: Attenuated or fluctuating emergency beacon.
   - `DORMANT_INTERFERENCE`: Heavy electromagnetic interference / decoy distortion.
   Updated via Bayes' rule after each scan based on observed signal vs. expected signal
   at the current maximum-a-posteriori (MAP) location hypothesis, and collapsed to 1.0
   upon physical verification.

3. Expected Information Gain (EIG):
   H(B) = - sum_{c: B(c) > 0} B(c) ln B(c)
   EIG(x) = max(0, H(B) - E_s[H(B_{t+1} | s, x)])
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Set, Tuple
import numpy as np

from app.environment.sensors import expected_signal_from_target, observation_likelihood
from app.schemas.mission import Coordinate, EveOperationalStatus, PublicMissionConfig


class BeliefTracker:
    """Maintains normalized probability distributions over EVE's location and operational status."""

    def __init__(self, config: PublicMissionConfig) -> None:
        self.width: int = config.grid_width
        self.height: int = config.grid_height
        self.decay_rate: float = config.signal_decay_rate
        self.noise_std: float = config.sensor_noise_std
        self.decoy_count: int = config.decoy_count

        self.impossible_cells: Set[Coordinate] = set(tuple(c) for c in config.obstacles)
        self.verified_target: Coordinate | None = None

        self.grid: np.ndarray = np.zeros((self.width, self.height), dtype=np.float64)
        self.status_beliefs: Dict[str, float] = {
            EveOperationalStatus.OPERATIONAL_SIGNAL.value: 1.0 / 3.0,
            EveOperationalStatus.CRITICAL_LOW_POWER.value: 1.0 / 3.0,
            EveOperationalStatus.DORMANT_INTERFERENCE.value: 1.0 / 3.0,
        }
        self.reset_uniform(self.impossible_cells)

    def reset_uniform(self, known_obstacles: Iterable[Coordinate]) -> None:
        """Initialize a uniform prior over all valid candidate cells and status hypotheses."""
        self.impossible_cells = set(tuple(c) for c in known_obstacles)
        self.verified_target = None
        self.grid.fill(1.0)
        for ox, oy in self.impossible_cells:
            if 0 <= ox < self.width and 0 <= oy < self.height:
                self.grid[ox, oy] = 0.0
        self.status_beliefs = {
            EveOperationalStatus.OPERATIONAL_SIGNAL.value: 1.0 / 3.0,
            EveOperationalStatus.CRITICAL_LOW_POWER.value: 1.0 / 3.0,
            EveOperationalStatus.DORMANT_INTERFERENCE.value: 1.0 / 3.0,
        }
        self._normalize()

    def _normalize(self) -> None:
        """Normalize `self.grid` so probabilities sum to 1.0, handling underflow/zero-mass."""
        if self.verified_target is not None:
            vx, vy = self.verified_target
            self.grid.fill(0.0)
            self.grid[vx, vy] = 1.0
            return

        for ix, iy in self.impossible_cells:
            if 0 <= ix < self.width and 0 <= iy < self.height:
                self.grid[ix, iy] = 0.0

        total = float(np.sum(self.grid))
        if total <= 1e-15 or math.isnan(total):
            valid_count = 0
            for x in range(self.width):
                for y in range(self.height):
                    if (x, y) not in self.impossible_cells:
                        self.grid[x, y] = 1.0
                        valid_count += 1
                    else:
                        self.grid[x, y] = 0.0
            if valid_count > 0:
                self.grid /= float(valid_count)
            return

        self.grid /= total

        valid_mask = np.ones((self.width, self.height), dtype=bool)
        for ix, iy in self.impossible_cells:
            if 0 <= ix < self.width and 0 <= iy < self.height:
                valid_mask[ix, iy] = False

        if np.any(valid_mask):
            self.grid[valid_mask] = np.maximum(self.grid[valid_mask], 1e-9)
            self.grid[~valid_mask] = 0.0
            self.grid /= float(np.sum(self.grid))

    def mark_obstacles(self, obstacles: Iterable[Coordinate]) -> None:
        """Zero out newly discovered obstacle cells and renormalize."""
        changed = False
        for ox, oy in obstacles:
            c = (int(ox), int(oy))
            if c not in self.impossible_cells and 0 <= c[0] < self.width and 0 <= c[1] < self.height:
                self.impossible_cells.add(c)
                self.grid[c[0], c[1]] = 0.0
                changed = True
        if changed:
            self._normalize()

    def mark_cell_visited(
        self,
        cell: Coordinate,
        target_found: bool,
        verified_status: EveOperationalStatus | None = None,
    ) -> None:
        """Update belief when WALL-E physically visits `cell`."""
        cx, cy = int(cell[0]), int(cell[1])
        if target_found:
            self.verified_target = (cx, cy)
            self.grid.fill(0.0)
            self.grid[cx, cy] = 1.0
            if verified_status is not None:
                for k in self.status_beliefs:
                    self.status_beliefs[k] = 1.0 if k == verified_status.value else 0.0
            return

        if (cx, cy) not in self.impossible_cells:
            self.impossible_cells.add((cx, cy))
        if 0 <= cx < self.width and 0 <= cy < self.height:
            self.grid[cx, cy] = 0.0
        self._normalize()

    def update_from_scan(
        self,
        sensor_pos: Coordinate,
        signal_reading: float | None,
        modulation_hint: str | None = None,
    ) -> None:
        """Perform a Bayesian belief update in log-space given a scalar radio scan reading."""
        if self.verified_target is not None:
            return
        if signal_reading is None or math.isnan(signal_reading) or not (0.0 <= signal_reading <= 1.0):
            return

        log_post = np.full((self.width, self.height), -np.inf, dtype=np.float64)
        for x in range(self.width):
            for y in range(self.height):
                prior = float(self.grid[x, y])
                if prior <= 0.0 or (x, y) in self.impossible_cells:
                    continue
                like = observation_likelihood(
                    observed_signal=signal_reading,
                    sensor_pos=sensor_pos,
                    candidate_cell=(x, y),
                    decay_rate=self.decay_rate,
                    noise_std=self.noise_std,
                    decoy_count=self.decoy_count,
                )
                log_post[x, y] = math.log(max(prior, 1e-12)) + math.log(max(like, 1e-12))

        finite_mask = np.isfinite(log_post)
        if not np.any(finite_mask):
            self._normalize()
            return

        max_log = float(np.max(log_post[finite_mask]))
        stabilized = np.zeros((self.width, self.height), dtype=np.float64)
        stabilized[finite_mask] = np.exp(log_post[finite_mask] - max_log)
        self.grid = stabilized
        self._normalize()

        # Also update EVE operational status belief using Bayes' rule
        self._update_status_belief(sensor_pos, signal_reading, modulation_hint)

    def _update_status_belief(
        self,
        sensor_pos: Coordinate,
        signal_reading: float,
        modulation_hint: str | None = None,
    ) -> None:
        top_cell, _ = self.most_likely_cell()
        expected_sig = expected_signal_from_target(sensor_pos, top_cell, self.decay_rate)
        residual = abs(signal_reading - expected_sig)

        # Likelihoods for each operational status given signal strength and residual
        like_op = 0.65 if (residual < 0.08 and signal_reading > 0.35) else 0.20
        like_low = 0.65 if (0.12 <= signal_reading <= 0.55) else 0.25
        like_dorm = 0.70 if (residual >= 0.11 or self.decoy_count > 1) else 0.20

        if modulation_hint == "STEADY_CARRIER":
            like_op *= 1.8
        elif modulation_hint == "PULSED_LOW_VOLTAGE":
            like_low *= 1.8
        elif modulation_hint == "MULTIPATH_INTERFERENCE":
            like_dorm *= 1.8

        raw = {
            EveOperationalStatus.OPERATIONAL_SIGNAL.value: self.status_beliefs[
                EveOperationalStatus.OPERATIONAL_SIGNAL.value
            ] * like_op,
            EveOperationalStatus.CRITICAL_LOW_POWER.value: self.status_beliefs[
                EveOperationalStatus.CRITICAL_LOW_POWER.value
            ] * like_low,
            EveOperationalStatus.DORMANT_INTERFERENCE.value: self.status_beliefs[
                EveOperationalStatus.DORMANT_INTERFERENCE.value
            ] * like_dorm,
        }
        s_total = sum(raw.values())
        if s_total > 0:
            self.status_beliefs = {k: round(v / s_total, 4) for k, v in raw.items()}

    def most_likely_status(self) -> Tuple[str, float]:
        """Return (status_name, probability) for EVE's inferred operational status."""
        best_k = max(self.status_beliefs, key=lambda k: self.status_beliefs[k])
        return best_k, float(self.status_beliefs[best_k])

    def entropy(self) -> float:
        """Calculate Shannon entropy H(B) = -sum p(x) ln p(x) in nats."""
        return self._entropy_of_distribution(self.grid)

    @staticmethod
    def _entropy_of_distribution(dist: np.ndarray) -> float:
        mask = dist > 1e-12
        if not np.any(mask):
            return 0.0
        probs = dist[mask]
        return float(-np.sum(probs * np.log(probs)))

    def expected_information_gain(self, scan_pos: Coordinate) -> float:
        """Estimate Expected Information Gain (expected entropy reduction) of scanning at `scan_pos`."""
        if self.verified_target is not None:
            return 0.0

        current_h = self.entropy()
        if current_h <= 1e-6:
            return 0.0

        valid_coords: List[Coordinate] = []
        priors: List[float] = []
        mus: List[float] = []

        for x in range(self.width):
            for y in range(self.height):
                p = float(self.grid[x, y])
                if p > 1e-6 and (x, y) not in self.impossible_cells:
                    valid_coords.append((x, y))
                    priors.append(p)
                    mus.append(expected_signal_from_target(scan_pos, (x, y), self.decay_rate))

        if len(valid_coords) <= 1:
            return 0.0

        prior_arr = np.array(priors, dtype=np.float64)
        prior_arr /= np.sum(prior_arr)
        mu_arr = np.array(mus, dtype=np.float64)
        sigma = max(self.noise_std, 0.02)

        mean_signal = float(np.sum(prior_arr * mu_arr))
        var_signal = float(np.sum(prior_arr * ((mu_arr - mean_signal) ** 2 + sigma ** 2)))
        std_signal = math.sqrt(max(var_signal, 1e-4))

        sample_offsets = (-1.2, -0.5, 0.0, 0.5, 1.2)
        sample_signals = [float(np.clip(mean_signal + z * std_signal, 0.0, 1.0)) for z in sample_offsets]

        expected_post_h = 0.0
        total_weight = 0.0

        for s in sample_signals:
            diff = s - mu_arr
            gauss = np.exp(-0.5 * (diff / sigma) ** 2) / (sigma * math.sqrt(2.0 * math.pi))
            likes = 0.96 * gauss + 0.04
            unnorm_post = prior_arr * likes
            marginal_s = float(np.sum(unnorm_post))
            if marginal_s <= 1e-12:
                continue
            post = unnorm_post / marginal_s
            h_s = self._entropy_of_distribution(post)
            expected_post_h += marginal_s * h_s
            total_weight += marginal_s

        if total_weight <= 1e-12:
            return 0.0

        expected_post_h /= total_weight
        eig = max(0.0, current_h - expected_post_h)
        return float(eig)

    def most_likely_cell(self) -> Tuple[Coordinate, float]:
        """Return (coordinate, probability) of the highest-belief cell."""
        idx = int(np.argmax(self.grid))
        x, y = divmod(idx, self.height)
        return (int(x), int(y)), float(self.grid[x, y])

    def to_nested_list(self) -> List[List[float]]:
        """Return 2D list [x][y] rounded for clean JSON serialization."""
        return [
            [round(float(self.grid[x, y]), 6) for y in range(self.height)]
            for x in range(self.width)
        ]
