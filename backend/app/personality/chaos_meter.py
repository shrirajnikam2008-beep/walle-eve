"""Five-Star Humorous Mission-Stress Indicator ('Chaos Meter').

Documented Scoring Rules:
-------------------------
The Chaos Meter is a purely observational, game-inspired stress indicator modeled
after 5-star open-world wanted levels. It never affects planning or beliefs.

Point Deltas by Verified Mission Event:
- `FALSE_ALARM` (visited a cell near a decoy beacon): +25 points
- `LOW_BATTERY` (battery drops <= 20% of capacity): +20 points
- `BLOCKED_OR_FAILED_ACTION` (bumped obstacle or invalid action): +20 points
- `WEAK_SCAN` (radio scan returns < 0.22 signal): +8 points
- `REPEATED_INEFFICIENT_ACTION` (re-scanning same cell or visiting cell >= 3 times): +15 points
- `PLAN_CHANGED` (Bayesian posterior shifts top hypothesis by >= 3 cells): +10 points
- `MISSION_FAILURE` (battery depleted, timeout, or unreachable target): +50 points
- `RECHARGE_RELIEF` (successful solar recharge): -12 points (calms stress slightly)

Star Thresholds:
- 1 Star (0 - 19 pts):   "Everything is fine. Statistically speaking, probably."
- 2 Stars (20 - 39 pts): "Minor turbulence. Sensors are guessing, WALL-E is coping."
- 3 Stars (40 - 64 pts): "Three suspicious signals and one battery crisis. The mission is now a workplace incident."
- 4 Stars (65 - 89 pts): "DEFCON TOASTER. Navigation is arguing with Bayes' theorem over the radio."
- 5 Stars (90+ pts):     "MISSION CHAOS MAXIMUM. Someone please revoke my explorer certification."
"""
from __future__ import annotations

from typing import List, Tuple

from app.schemas.mission import ChaosEventDetail, ChaosState


STAR_DESCRIPTIONS: dict[int, Tuple[str, str]] = {
    1: (
        "NOMINAL (1 STAR)",
        "Everything is fine. Statistically speaking, probably.",
    ),
    2: (
        "ELEVATED (2 STARS)",
        "Minor turbulence. Sensors are guessing, WALL-E is coping.",
    ),
    3: (
        "WORKPLACE INCIDENT (3 STARS)",
        "Three suspicious signals and one battery crisis. The mission is now a workplace incident.",
    ),
    4: (
        "DEFCON TOASTER (4 STARS)",
        "High operational chaos. Navigation is arguing with Bayes' theorem over the radio.",
    ),
    5: (
        "MAXIMUM CHAOS (5 STARS)",
        "MISSION CHAOS MAXIMUM. Someone please revoke my explorer certification.",
    ),
}


class ChaosMeter:
    """Tracks deterministic chaos points and 1-to-5 star levels with an explicit audit log."""

    def __init__(self) -> None:
        self.raw_score: int = 0
        self.events: List[ChaosEventDetail] = []
        self._low_battery_triggered: bool = False

    def reset(self) -> None:
        self.raw_score = 0
        self.events.clear()
        self._low_battery_triggered = False

    @staticmethod
    def score_to_stars(score: int) -> int:
        if score >= 90:
            return 5
        if score >= 65:
            return 4
        if score >= 40:
            return 3
        if score >= 20:
            return 2
        return 1

    def record_event(self, step: int, event_code: str, delta: int, reason: str) -> None:
        """Apply a deterministic score change and log the reason."""
        self.raw_score = max(0, min(150, self.raw_score + delta))
        self.events.append(
            ChaosEventDetail(
                step=step,
                event_code=event_code,
                delta_points=delta,
                reason=reason,
            )
        )

    def evaluate_step(
        self,
        step: int,
        battery: int,
        battery_capacity: int,
        action_succeeded: bool,
        is_false_alarm: bool,
        weak_scan: bool,
        repeated_action: bool,
        plan_changed: bool,
        recharged: bool,
        mission_failed: bool,
    ) -> ChaosState:
        """Update the Chaos Meter from verified step events and return current `ChaosState`."""
        if not action_succeeded and not mission_failed:
            self.record_event(
                step,
                "BLOCKED_ACTION",
                20,
                "Action blocked or failed in environment execution (+20 chaos).",
            )

        if is_false_alarm:
            self.record_event(
                step,
                "FALSE_ALARM",
                25,
                "Investigated high-signal coordinates that turned out to be a decoy beacon (+25 chaos).",
            )

        if battery <= int(0.22 * battery_capacity) and not self._low_battery_triggered:
            self._low_battery_triggered = True
            self.record_event(
                step,
                "LOW_BATTERY",
                20,
                f"Battery reserve dropped to {battery}/{battery_capacity} (+20 chaos).",
            )
        elif battery > int(0.40 * battery_capacity):
            self._low_battery_triggered = False

        if weak_scan:
            self.record_event(
                step,
                "WEAK_SCAN",
                8,
                "Sensor scan returned background static (<0.22 signal strength) (+8 chaos).",
            )

        if repeated_action:
            self.record_event(
                step,
                "REPEATED_ACTION",
                15,
                "Repeated scan or redundant cell traversal detected (+15 chaos).",
            )

        if plan_changed:
            self.record_event(
                step,
                "PLAN_CHANGED",
                10,
                "New sensor evidence forced a major Bayesian hypothesis shift (+10 chaos).",
            )

        if recharged:
            self.record_event(
                step,
                "RECHARGE_RELIEF",
                -12,
                "Docked at solar station and restored battery reserves (-12 chaos).",
            )

        if mission_failed:
            self.record_event(
                step,
                "MISSION_FAILURE",
                50,
                "Mission terminated without locating EVE (+50 chaos).",
            )

        stars = self.score_to_stars(self.raw_score)
        label, quote = STAR_DESCRIPTIONS[stars]
        return ChaosState(
            stars=stars,
            raw_score=self.raw_score,
            status_label=label,
            headline_quote=quote,
            recent_reasons=list(reversed(self.events[-6:])),
        )

    def get_state(self) -> ChaosState:
        stars = self.score_to_stars(self.raw_score)
        label, quote = STAR_DESCRIPTIONS[stars]
        return ChaosState(
            stars=stars,
            raw_score=self.raw_score,
            status_label=label,
            headline_quote=quote,
            recent_reasons=list(reversed(self.events[-6:])),
        )
