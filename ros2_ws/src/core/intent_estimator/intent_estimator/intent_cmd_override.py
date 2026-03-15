"""Helpers for timed intent-driven locomotion commands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TimedIntentXVelocity:
    """Map discrete intent labels to a timed x-velocity command."""

    hold_s: float = 1.0
    backward_x_vel: float = -0.5
    forward_x_vel: float = 0.5
    idle_intent_label: int = 0
    backward_intent_label: int = 1
    forward_intent_label: int = 2
    latest_intent: int = 0
    active_x_vel: float = 0.0
    active_until_s: float = 0.0

    def on_intent(self, intent: int, now_s: float) -> None:
        self.latest_intent = int(intent)
        if intent == self.backward_intent_label:
            self._activate(self.backward_x_vel, now_s)
            return
        if intent == self.forward_intent_label:
            self._activate(self.forward_x_vel, now_s)
            return
        if intent == self.idle_intent_label:
            self.clear()

    def current_x_vel(self, now_s: float) -> float:
        if float(now_s) < self.active_until_s:
            return self.active_x_vel
        return 0.0

    def clear(self) -> None:
        self.active_x_vel = 0.0
        self.active_until_s = 0.0

    def _activate(self, x_vel: float, now_s: float) -> None:
        self.active_x_vel = float(x_vel)
        self.active_until_s = float(now_s) + self.hold_s
