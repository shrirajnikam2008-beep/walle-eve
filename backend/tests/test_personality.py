"""Unit tests for the GTA-inspired CommentaryEngine and 5-star ChaosMeter."""
from __future__ import annotations

from app.personality.chaos_meter import ChaosMeter
from app.personality.commentary import CommentaryEngine


def test_commentary_selection_cooldown_and_toggle() -> None:
    engine = CommentaryEngine(enabled=True)
    msg1 = engine.generate(step=1, event_type="SIGNAL_DETECTED")
    assert msg1 is not None
    assert "WALL-E" in msg1.speaker

    # Cooldown suppresses immediate repeat on step 2
    msg2 = engine.generate(step=2, event_type="SIGNAL_DETECTED")
    assert msg2 is None

    # After cooldown (step 5), returns rotated dialogue variation
    msg3 = engine.generate(step=5, event_type="SIGNAL_DETECTED")
    assert msg3 is not None
    assert msg3.message != msg1.message

    # Disabling humor returns None even for forced events
    engine.enabled = False
    assert engine.generate(step=10, event_type="MISSION_SUCCESS", force=True) is None


def test_chaos_meter_explicit_rules() -> None:
    chaos = ChaosMeter()
    state0 = chaos.get_state()
    assert state0.stars == 1
    assert state0.raw_score == 0

    # Trigger false alarm (+25) and low battery (+20) -> 45 pts -> 3 stars
    state1 = chaos.evaluate_step(
        step=4,
        battery=15,
        battery_capacity=100,
        action_succeeded=True,
        is_false_alarm=True,
        weak_scan=False,
        repeated_action=False,
        plan_changed=False,
        recharged=False,
        mission_failed=False,
    )
    assert state1.raw_score == 45
    assert state1.stars == 3
    assert len(state1.recent_reasons) == 2

    # Trigger mission failure (+50) -> 95 pts -> 5 stars
    state2 = chaos.evaluate_step(
        step=5,
        battery=0,
        battery_capacity=100,
        action_succeeded=False,
        is_false_alarm=False,
        weak_scan=False,
        repeated_action=False,
        plan_changed=False,
        recharged=False,
        mission_failed=True,
    )
    assert state2.stars == 5
    assert "MAXIMUM" in state2.headline_quote
