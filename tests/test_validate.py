"""Pre-flight checks: validate_adapter must catch the silent adapter bugs,
benchmark_bot must measure decision latency."""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from examples.nim_game import NimState, make_state
from mathathon_kit import MinimaxBot, RandomBot, benchmark_bot, validate_adapter


def test_validate_accepts_a_correct_adapter():
    report = validate_adapter(NimState(pile=21), games=5)
    assert report.ok, report.summary()
    assert report.states_checked > 0


@dataclass
class _MutatingState:
    """Broken adapter: apply() mutates self instead of returning a new state."""

    pile: int = 10
    current_index: int = 0
    players: Tuple[int, int] = (0, 1)

    @property
    def current_player(self) -> int:
        return self.players[self.current_index]

    def legal_actions(self, player: Optional[int] = None) -> Sequence[int]:
        return () if self.is_terminal() else (1, 2)

    def apply(self, action: int) -> "_MutatingState":
        self.pile -= action  # BUG: mutation
        self.current_index = 1 - self.current_index
        return self

    def is_terminal(self) -> bool:
        return self.pile <= 0

    def score(self, player: int) -> float:
        return 0.0


def test_validate_flags_mutation():
    report = validate_adapter(_MutatingState(), games=3)
    assert not report.ok
    assert any("mutated" in e for e in report.errors)


@dataclass(frozen=True)
class _NeverEndingState:
    """Broken adapter: the game never reaches a terminal state."""

    players: Tuple[int, int] = (0, 1)
    current_index: int = 0

    @property
    def current_player(self) -> int:
        return self.players[self.current_index]

    def legal_actions(self, player: Optional[int] = None) -> Sequence[int]:
        return (0,)

    def apply(self, action: int) -> "_NeverEndingState":
        return _NeverEndingState(self.players, 1 - self.current_index)

    def is_terminal(self) -> bool:
        return False

    def score(self, player: int) -> float:
        return 0.0


def test_validate_flags_nontermination():
    report = validate_adapter(_NeverEndingState(), games=1, max_turns=50)
    assert not report.ok
    assert any("terminate" in e for e in report.errors)


def test_benchmark_reports_latency():
    result = benchmark_bot(MinimaxBot(depth=6), make_state, positions=20, time_limit=0.2)
    assert result.samples == 20
    assert result.worst >= result.mean >= 0.0
    assert result.ok  # a depth-6 minimax on tiny Nim is well under 0.2s
    assert "latency" in result.summary()


def test_benchmark_detects_overrun():
    result = benchmark_bot(RandomBot(), make_state, positions=10, time_limit=0.0)
    # A zero-second limit forces every move to count as an overrun.
    assert result.overruns == result.samples
    assert not result.ok


def test_benchmark_works_on_simultaneous_games():
    from examples.figgie import MarketMakerBot
    from examples.figgie import make_state as figgie_state

    result = benchmark_bot(MarketMakerBot(), figgie_state, positions=20, time_limit=0.2)
    assert result.samples == 20
    assert result.ok
