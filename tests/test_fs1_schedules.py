"""Warm-start schedules for the opponent-relative weight and the win term."""
from src.fs1.reward import alpha_from_progress, ramp_with_hold


def test_ramp_with_hold_reproduces_the_plain_ramp_when_start_and_hold_are_zero():
    for p in (0.0, 0.1, 0.3, 0.9):
        assert abs(ramp_with_hold(p, 0.0, 0.0, 0.3) - alpha_from_progress(p, 0.3)) < 1e-12
    assert ramp_with_hold(1e-4, 0.0, 0.0, 1e-6) == 1.0            # the iteration-1 "1 from update 1" setting
    assert ramp_with_hold(0.0, 0.0, 0.0, 1e-6) == 0.0


def test_ramp_with_hold_holds_then_ramps():
    f = lambda p: ramp_with_hold(p, 0.3, 0.4, 0.7)                # noqa: E731
    assert f(0.0) == 0.3 and f(0.2) == 0.3 and f(0.4) == 0.3
    assert abs(f(0.55) - 0.65) < 1e-12
    assert f(0.7) == 1.0 and f(0.95) == 1.0
    assert ramp_with_hold(0.5, 0.3, 0.6, 0.6) == 0.3 and ramp_with_hold(0.61, 0.3, 0.6, 0.6) == 1.0   # step
