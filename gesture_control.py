import time


class GestureController:
    """Debounce hand state changes and emit one event per fist transition."""

    def __init__(self, debounce_seconds: float = 0.3, cooldown_seconds: float = 3.0):
        if debounce_seconds < 0:
            raise ValueError("debounce_seconds must not be negative")
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")

        self.debounce_seconds = debounce_seconds
        self.cooldown_seconds = cooldown_seconds
        self.stable_closed: bool | None = None
        self._candidate_closed: bool | None = None
        self._candidate_since: float | None = None
        self._last_triggered_at: float | None = None

    def update(self, observed_closed: bool | None, now: float | None = None) -> bool:
        """Return True once for a debounced open-to-closed transition.

        Pass None when no hand is detected. A missing hand cancels a pending
        transition but does not change the last stable state.
        """
        timestamp = time.monotonic() if now is None else now

        if observed_closed is None:
            self._candidate_closed = None
            self._candidate_since = None
            return False

        if self.stable_closed == observed_closed:
            self._candidate_closed = None
            self._candidate_since = None
            return False

        if self._candidate_closed != observed_closed:
            self._candidate_closed = observed_closed
            self._candidate_since = timestamp
            return False

        if timestamp - self._candidate_since < self.debounce_seconds:
            return False

        previous_closed = self.stable_closed
        self.stable_closed = observed_closed
        self._candidate_closed = None
        self._candidate_since = None

        if previous_closed is not False or not observed_closed:
            return False

        if (
            self._last_triggered_at is not None
            and timestamp - self._last_triggered_at < self.cooldown_seconds
        ):
            return False

        self._last_triggered_at = timestamp
        return True

    def cooldown_remaining(self, now: float | None = None) -> float:
        if self._last_triggered_at is None:
            return 0.0
        timestamp = time.monotonic() if now is None else now
        return max(
            0.0,
            self.cooldown_seconds - (timestamp - self._last_triggered_at),
        )
