"""GTA-Inspired Satirical Radio Commentary Engine for WALL-E.

Design Guarantees:
------------------
1. Purely reactive: consumes verified mission events and returns humorous dialogue.
2. Zero side effects on navigation, beliefs, sensor readings, energy costs, or metrics.
3. Cooldown & rotation tracking prevents repetitive messages.
4. Works 100% locally without any API key, with an optional safe fallback hook if
   an environment LLM key is ever configured.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from app.schemas.mission import RadioMessage


DIALOGUE_BANK: Dict[str, List[Tuple[str, str]]] = {
    "MISSION_START": [
        (
            "Dispatch, I need a map, a battery, and a miracle. Apparently, only one of those is included.",
            "Copy that, WALL-E. Budget cuts hit miracles first. Start scanning.",
        ),
        (
            "Systems online. Uniform prior loaded, which is math for 'I have literally no idea where EVE is.'",
            "Just follow the Bayesian gradient and try not to scratch the chassis.",
        ),
        (
            "Twelve by twelve grid of existential dread. Let's go rescue someone way more advanced than me.",
            "Telemetry locked. Good luck out there, little compactor.",
        ),
    ],
    "SIGNAL_DETECTED": [
        (
            "Signal detected. Either that's EVE or the universe is doing stand-up comedy again.",
            "Posterior probabilities updated. Follow the hot zone, carefully.",
        ),
        (
            "Whoa, solid radio spike! My antenna is vibrating and for once it's not a loose bolt.",
            "Confirmed high likelihood region. Converging route now.",
        ),
        (
            "Picking up a strong transponder ping! Bayes' theorem says we're getting warm.",
            "Keep an eye out for decoy beacons in that sector.",
        ),
    ],
    "FALSE_ALARM": [
        (
            "We followed that signal across half the map. It was a decoy. I'm filing a complaint with physics.",
            "Scratch that sector from the prior. Re-normalizing belief map.",
        ),
        (
            "Bamboozled by a toaster-grade decoy beacon! Whoever planted that owes me six watt-hours.",
            "Zeroing out local cells. Don't take it personally, WALL-E.",
        ),
        (
            "Plot twist: not EVE. Just a counterfeit radio ping mocking my career choices.",
            "Updating likelihoods. Let's move to the secondary probability peak.",
        ),
    ],
    "WEAK_SCAN": [
        (
            "Scan complete. We found cosmic background static and my own echoing disappointment.",
            "Negative evidence is still evidence! You just ruled out an entire neighborhood.",
        ),
        (
            "Zero point nothing on the transceiver. At least we know where EVE definitely isn't.",
            "Entropy reduced. Keep pushing toward unexplored cells.",
        ),
    ],
    "OBSTACLE": [
        (
            "Fantastic. A wall. My greatest enemy since the invention of walls.",
            "Recalculating A* path around the debris field.",
        ),
        (
            "Impassable terrain detected. Who left a giant pile of industrial rubble in the middle of my shortest path?",
            "Adjusting heuristic route. Go around it, hero.",
        ),
    ],
    "LOW_BATTERY": [
        (
            "Battery at critical levels. If I survive this mission, I'm switching to a desk job.",
            "Resource manager overriding search: route to nearest solar dock immediately!",
        ),
        (
            "Low voltage warning flashing! My life is flashing before my optical sensors and it's mostly trash cubes.",
            "Hold onto your capacitors, solar station is on the A* path.",
        ),
    ],
    "CHARGING": [
        (
            "Recharging. Please respect my privacy during this difficult electrical period.",
            "Solar arrays soaking up watts. Back on search protocol shortly.",
        ),
        (
            "Sipping premium artisanal photons at the charging dock. Give me five seconds.",
            "Battery reserves climbing. Belief map standing by.",
        ),
    ],
    "PLAN_CHANGED": [
        (
            "New data just Nuked our old hypothesis. Ripping up the route and pivoting across the sector!",
            "That's Bayesian updating in action. New argmax coordinates locked.",
        ),
        (
            "Hold the phone, Dispatch—the probability peak just migrated across the grid. Rerouting!",
            "Confirmed shift in posterior distribution. Updating A* waypoints.",
        ),
    ],
    "INEFFICIENT_ROUTE": [
        (
            "That route was so inefficient it deserves its own government investigation.",
            "Obstacles and decoys forced a detour. Stay focused on the posterior.",
        ),
        (
            "Another scan? At this point, the rocks know more about EVE than we do.",
            "Information gain is dropping here—time to physically inspect the top candidate.",
        ),
    ],
    "MISSION_SUCCESS": [
        (
            "EVE located. Rescue successful. Somebody put this toaster in charge of the next mission.",
            "Target verified at 100% confidence! Outstanding work, WALL-E. Bring her home.",
        ),
        (
            "Visual confirmation on EVE! Take that, sensor noise, decoys, and basic probability!",
            "Mission accomplished! Extraction shuttle inbound.",
        ),
    ],
    "MISSION_FAILURE": [
        (
            "Mission unsuccessful. EVE remains missing. My confidence has also left the building.",
            "Power reserves depleted. Dispatching recovery drone to tow you back.",
        ),
    ],
    "MISSION_TIMEOUT": [
        (
            "We hit the action limit! I spent half the shift arguing with radio static and now overtime is denied.",
            "Mission clock expired. Logging final Bayesian estimate for debrief.",
        ),
    ],
    "MISSION_UNREACHABLE": [
        (
            "She's behind a solid containment wall! Unless you installed a plasma torch while I was sleeping, we're blocked.",
            "Confirmed zero reachable paths to remaining candidate cells. Safe abort approved.",
        ),
    ],
}


class CommentaryEngine:
    """Selects context-appropriate satirical radio dialogue with cooldowns and rotation."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled: bool = enabled
        self._variation_Indices: Dict[str, int] = {}
        self._last_step_by_event: Dict[str, int] = {}
        self.cooldown_steps: int = 3

    def reset(self, enabled: Optional[bool] = None) -> None:
        if enabled is not None:
            self.enabled = enabled
        self._variation_Indices.clear()
        self._last_step_by_event.clear()

    def generate(self, step: int, event_type: str, force: bool = False) -> Optional[RadioMessage]:
        """Return a `RadioMessage` for `event_type` if humor is enabled and cooldown permits."""
        if not self.enabled:
            return None

        if event_type not in DIALOGUE_BANK:
            return None

        # Terminal events or mission start always bypass cooldown
        is_terminal_or_start = event_type in (
            "MISSION_START",
            "MISSION_SUCCESS",
            "MISSION_FAILURE",
            "MISSION_TIMEOUT",
            "MISSION_UNREACHABLE",
        )

        if not force and not is_terminal_or_start:
            last_step = self._last_step_by_event.get(event_type, -999)
            if (step - last_step) < self.cooldown_steps:
                return None

        options = DIALOGUE_BANK[event_type]
        idx = self._variation_Indices.get(event_type, 0) % len(options)
        self._variation_Indices[event_type] = idx + 1
        self._last_step_by_event[event_type] = step

        line, reply = options[idx]
        return RadioMessage(
            step=step,
            event_type=event_type,
            speaker="WALL-E",
            dispatcher_reply=reply,
            message=line,
        )
