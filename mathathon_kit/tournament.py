from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field
from math import erfc, sqrt
from typing import Callable, Dict, Generic, List, Mapping, Optional, Sequence, Tuple

from .core import (
    Action,
    Bot,
    GameResult,
    GameState,
    Player,
    SimultaneousSimulator,
    SimultaneousState,
    Simulator,
)


InitialStateFactory = Callable[[int], GameState[Player, Action]]
SimultaneousFactory = Callable[[int], SimultaneousState[Player, Action]]


# ---------------------------------------------------------------------------
# Significance helpers -- "is this win rate real, or just noise?"
# ---------------------------------------------------------------------------


def wilson_interval(successes: float, total: float, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score confidence interval for a win rate (default 95%).

    Robust at small samples and near 0/1, unlike the naive normal interval.
    ``successes`` may be fractional (draws count as 0.5).
    """
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    centre = (p + z2 / (2.0 * total)) / denom
    half = (z * sqrt(p * (1.0 - p) / total + z2 / (4.0 * total * total))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def binomial_p_value(successes: float, total: float, p0: float = 0.5) -> float:
    """Two-sided p-value (normal approximation) that the observed win rate
    differs from ``p0``. Use this to answer "I won 28 of 50 vs my old bot --
    is the new version actually better?": ``binomial_p_value(28, 50)``.

    A p-value below 0.05 means the difference is unlikely to be chance.
    """
    if total <= 0:
        return 1.0
    se = sqrt(p0 * (1.0 - p0) / total)
    if se == 0.0:
        return 1.0
    z = abs(successes / total - p0) / se
    return min(1.0, erfc(z / sqrt(2.0)))  # two-sided tail


@dataclass
class MatchStats:
    games: int = 0
    wins: float = 0.0
    score_sum: float = 0.0
    errors: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def avg_score(self) -> float:
        return self.score_sum / self.games if self.games else 0.0

    def win_rate_ci(self, z: float = 1.96) -> Tuple[float, float]:
        """95% Wilson confidence interval for ``win_rate``."""
        return wilson_interval(self.wins, self.games, z)

    def p_value(self, baseline: float = 0.5) -> float:
        """Two-sided p-value that ``win_rate`` differs from ``baseline``."""
        return binomial_p_value(self.wins, self.games, baseline)


_ERROR_SAMPLE_CAP = 50


def _collect_errors(result: "GameResult", sink: List[str]) -> None:
    """Accumulate distinct error messages (timeout / crash / illegal action)
    so they survive even when ``keep_results=False``."""
    for err in result.errors:
        if err not in sink and len(sink) < _ERROR_SAMPLE_CAP:
            sink.append(err)


@dataclass
class TournamentReport(Generic[Player, Action]):
    standings: Dict[str, MatchStats]
    matrix: Dict[Tuple[str, ...], MatchStats]
    elo: Dict[str, float]
    results: List[GameResult[Player, Action]] = field(default_factory=list)
    seats: int = 2
    error_samples: List[str] = field(default_factory=list)

    def ranked(self) -> List[Tuple[str, MatchStats]]:
        return sorted(
            self.standings.items(),
            key=lambda item: (item[1].win_rate, item[1].avg_score),
            reverse=True,
        )

    def summary(self) -> str:
        lines = ["bot, games, win_rate, avg_score, errors, elo"]
        for name, stats in self.ranked():
            lines.append(
                f"{name}, {stats.games}, {stats.win_rate:.3f}, "
                f"{stats.avg_score:.3f}, {stats.errors}, {self.elo.get(name, 1500.0):.1f}"
            )
        return "\n".join(lines)

    def error_report(self, limit: int = 10) -> str:
        """Sample of the distinct error messages behind the ``errors`` counts
        (timeouts, crashes, illegal actions) -- visible without ``keep_results``."""
        if not self.error_samples:
            return "no errors"
        shown = self.error_samples[:limit]
        lines = [f"{len(self.error_samples)} distinct error(s) -- first {len(shown)}:"]
        lines.extend(f"  {msg}" for msg in shown)
        return "\n".join(lines)

    def significance(self, baseline: Optional[float] = None) -> str:
        """Standings with a 95% CI and a p-value vs ``baseline``.

        ``baseline`` defaults to the chance win rate ``1 / seats`` (0.5 for a
        2-player game, 0.25 for 4-player) -- so the test compares against luck,
        not a hard-coded 50%. Pass it explicitly to override.

        ``verdict`` is ``significant`` only when the whole confidence interval
        sits clear of the baseline -- i.e. the result is unlikely to be noise.
        Run a candidate against your previous version in the same round-robin,
        then read this to decide whether a tweak genuinely helped.
        """
        if baseline is None:
            baseline = 1.0 / self.seats if self.seats else 0.5
        lines = ["bot, games, win_rate, ci95_low, ci95_high, p_vs_%.2f, verdict" % baseline]
        for name, stats in self.ranked():
            lo, hi = stats.win_rate_ci()
            p = stats.p_value(baseline)
            decisive = p < 0.05 and (lo > baseline or hi < baseline)
            lines.append(
                f"{name}, {stats.games}, {stats.win_rate:.3f}, "
                f"{lo:.3f}, {hi:.3f}, {p:.4f}, "
                f"{'significant' if decisive else 'noise'}"
            )
        return "\n".join(lines)


class RoundRobin(Generic[Player, Action]):
    """Round-robin tournament for sequential turn-based games.

    Generalised to N players: every k-tuple of distinct bots (k = number of
    seats) plays ``games_per_pair`` games with rotated seat assignment.
    Set ``simultaneous=True`` together with a ``SimultaneousState`` factory
    to use the simultaneous-move simulator instead.
    """

    def __init__(
        self,
        players: Sequence[Player],
        initial_state_factory: InitialStateFactory[Player, Action],
        games_per_pair: int = 20,
        max_turns: int = 1_000,
        time_limit_per_move: float = 0.2,
        keep_results: bool = False,
        simultaneous: bool = False,
    ) -> None:
        self.players = list(players)
        self.initial_state_factory = initial_state_factory
        self.games_per_pair = games_per_pair
        self.max_turns = max_turns
        self.time_limit_per_move = time_limit_per_move
        self.keep_results = keep_results
        self.simultaneous = simultaneous

    def run(
        self,
        bots: Mapping[str, Bot[Player, Action]],
        seed: int = 0,
    ) -> TournamentReport[Player, Action]:
        names = list(bots.keys())
        n_seats = len(self.players)
        if len(names) < n_seats:
            raise ValueError(
                f"need at least {n_seats} bots for a {n_seats}-seat game, got {len(names)}"
            )

        standings: Dict[str, MatchStats] = defaultdict(MatchStats)
        matrix: Dict[Tuple[str, ...], MatchStats] = defaultdict(MatchStats)
        elo = {name: 1500.0 for name in names}
        results: List[GameResult[Player, Action]] = []
        error_samples: List[str] = []

        if self.simultaneous:
            simulator: object = SimultaneousSimulator(
                players=self.players,
                max_rounds=self.max_turns,
                time_limit_per_move=self.time_limit_per_move,
            )
        else:
            simulator = Simulator(
                players=self.players,
                max_turns=self.max_turns,
                time_limit_per_move=self.time_limit_per_move,
            )

        game_id = 0
        for combo in itertools.combinations(names, n_seats):
            base = list(combo)
            for repeat in range(self.games_per_pair):
                # Rotate seats so every bot plays each seat roughly equally.
                shift = repeat % n_seats
                order = base[shift:] + base[:shift]
                assigned = {self.players[i]: bots[order[i]] for i in range(n_seats)}
                state = self.initial_state_factory(seed + game_id)
                result = simulator.play(state, assigned, seed=seed + game_id, record=False)  # type: ignore[attr-defined]
                self._record_game(order, result, standings, matrix)
                self._update_elo(order, result, elo)
                _collect_errors(result, error_samples)
                if self.keep_results:
                    results.append(result)
                game_id += 1

        return TournamentReport(
            dict(standings), dict(matrix), elo, results,
            seats=n_seats, error_samples=error_samples,
        )

    def _record_game(
        self,
        order: Sequence[str],
        result: GameResult[Player, Action],
        standings: Dict[str, MatchStats],
        matrix: Dict[Tuple[str, ...], MatchStats],
    ) -> None:
        player_to_name = {self.players[i]: order[i] for i in range(len(self.players))}
        winning_names = {player_to_name[player] for player in result.winners}

        for player, name in player_to_name.items():
            opponents = tuple(n for n in order if n != name)
            win_value = 1.0 / len(winning_names) if name in winning_names else 0.0
            matrix_key = (name,) + opponents
            for bucket in (standings[name], matrix[matrix_key]):
                bucket.games += 1
                bucket.wins += win_value
                bucket.score_sum += result.scores[player]
                bucket.errors += len(result.errors)

    def _update_elo(
        self,
        order: Sequence[str],
        result: GameResult[Player, Action],
        elo: Dict[str, float],
        k: float = 24.0,
    ) -> None:
        # Multi-player Elo: each pair updates against each other using the
        # binary outcome "did player i finish strictly above player j?".
        player_to_name = {self.players[i]: order[i] for i in range(len(self.players))}
        scores = result.scores

        # Convert to per-name actual score in [0, 1] based on rank.
        n = len(order)
        ranks = sorted(
            ((scores[self.players[i]], order[i]) for i in range(n)),
            key=lambda x: x[0],
            reverse=True,
        )
        # Standard Elo for K players uses pairwise duels: 1 if higher score, 0.5 if equal, 0 otherwise.
        for i in range(n):
            for j in range(i + 1, n):
                left_name, right_name = order[i], order[j]
                ls, rs = scores[self.players[i]], scores[self.players[j]]
                if ls > rs:
                    left_score = 1.0
                elif ls < rs:
                    left_score = 0.0
                else:
                    left_score = 0.5
                expected_left = 1.0 / (
                    1.0 + 10 ** ((elo[right_name] - elo[left_name]) / 400.0)
                )
                # Smaller K so multi-pair updates per game don't explode.
                k_eff = k / max(1, n - 1)
                elo[left_name] += k_eff * (left_score - expected_left)
                elo[right_name] += k_eff * ((1.0 - left_score) - (1.0 - expected_left))
