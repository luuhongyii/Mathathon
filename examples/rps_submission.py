"""Adaptive Rock-Paper-Scissors submission for the Mathathon platform.

Protocol: each round the judge writes one line on stdin; the bot writes one of
``Rock`` / ``Paper`` / ``Scissors`` on stdout. Round 1's input is empty (no
history yet); from round 2 on the input line carries the opponent's last move.

Strategy: a multi-predictor meta-bot in the spirit of "Iocaine Powder".
Several predictors each guess the opponent's next move; for every prediction we
spawn three counter-agents (beat it / beat the beat / original) to cover an
opponent that anticipates us. Each agent is scored online by how it *would*
have done, and the best-scoring agent plays. A net-loss guardrail falls back to
uniform random so a hard-countering opponent cannot pull us below ~33%.
"""

from __future__ import annotations

import random
import sys

NAMES = ["Rock", "Paper", "Scissors"]


def beat(m: int) -> int:
    """Return the move that beats ``m``."""
    return (m + 1) % 3


def outcome(a: int, b: int) -> int:
    """+1 if move ``a`` beats ``b``, -1 if it loses, 0 on a tie."""
    if a == b:
        return 0
    return 1 if a == beat(b) else -1


def parse_move(line: str) -> int | None:
    """Extract a move index from a judge input line, or None if absent."""
    low = line.strip().lower()
    if not low:
        return None
    if "scissors" in low:
        return 2
    if "paper" in low:
        return 1
    if "rock" in low:
        return 0
    for ch in low:
        if ch in "r0":
            return 0
        if ch in "p1":
            return 1
        if ch in "s2":
            return 2
    return None


def _pattern_predict(seq: list[int], maxlen: int = 20) -> int | None:
    """Longest-matching-suffix predictor: find the most recent earlier spot
    where the current suffix occurred and return what followed it."""
    n = len(seq)
    if n < 2:
        return None
    for length in range(min(maxlen, n - 1), 0, -1):
        suffix = seq[n - length:]
        for i in range(n - length - 1, -1, -1):
            if seq[i:i + length] == suffix:
                return seq[i + length]
    return None


class RPSBot:
    """Online adaptive RPS player. ``record_opponent`` then ``next_move``."""

    DECAY = 0.9  # exponential forgetting on agent scores

    def __init__(self, rng: random.Random | None = None) -> None:
        self.rng = rng or random.Random()
        self.opp: list[int] = []      # opponent moves, one per finished round
        self.mine: list[int] = []     # our moves, one per finished round
        self.score: dict = {}         # (predictor, level) -> decayed score
        self.last_recos: dict = {}    # agent -> move it recommended last round
        self.net = 0                  # our cumulative result (win +1 / loss -1)

    # -- observation -------------------------------------------------------
    def record_opponent(self, opp_move: int) -> None:
        """Register the opponent's move from the round just played and update
        every agent's score against it."""
        self.opp.append(opp_move)
        if self.mine:
            self.net += outcome(self.mine[-1], opp_move)
        for key, mv in self.last_recos.items():
            self.score[key] = self.score.get(key, 0.0) * self.DECAY + outcome(mv, opp_move)

    # -- prediction --------------------------------------------------------
    def _predictions(self) -> dict:
        """Each predictor's guess of the opponent's next move."""
        preds: dict = {}
        opp = self.opp
        n = len(opp)
        if n == 0:
            return preds
        m = min(len(self.opp), len(self.mine))

        preds["repeat"] = opp[-1]

        cnt = [opp.count(0), opp.count(1), opp.count(2)]
        preds["freq"] = cnt.index(max(cnt))

        rec = opp[-8:]
        rc = [rec.count(0), rec.count(1), rec.count(2)]
        preds["recent"] = rc.index(max(rc))

        if n >= 2:  # order-1 Markov on opponent moves
            last = opp[-1]
            nxt = [0, 0, 0]
            for i in range(n - 1):
                if opp[i] == last:
                    nxt[opp[i + 1]] += 1
            if sum(nxt):
                preds["markov1"] = nxt.index(max(nxt))

        if n >= 3:  # order-2 Markov on opponent moves
            key2 = (opp[-2], opp[-1])
            nxt = [0, 0, 0]
            for i in range(n - 2):
                if (opp[i], opp[i + 1]) == key2:
                    nxt[opp[i + 2]] += 1
            if sum(nxt):
                preds["markov2"] = nxt.index(max(nxt))

        patt = _pattern_predict(opp)
        if patt is not None:
            preds["patt_opp"] = patt

        if m >= 2:  # pattern match on the interleaved (mine, opp) stream
            combo = [self.mine[i] * 3 + self.opp[i] for i in range(m)]
            cp = _pattern_predict(combo)
            if cp is not None:
                preds["patt_combo"] = cp % 3

        if m >= 2:  # opponent reacts to our previous move
            trigger = self.mine[-1]
            nxt = [0, 0, 0]
            for i in range(1, m):
                if self.mine[i - 1] == trigger:
                    nxt[self.opp[i]] += 1
            if sum(nxt):
                preds["react"] = nxt.index(max(nxt))

        return preds

    def _agents(self) -> dict:
        """Map every (predictor, level) agent to its recommended move."""
        agents: dict = {}
        for name, pm in self._predictions().items():
            agents[(name, 0)] = beat(pm)             # beat the prediction
            agents[(name, 1)] = pm                   # opponent expects level 0
            agents[(name, 2)] = beat(beat(pm))       # opponent expects level 1
        return agents

    # -- decision ----------------------------------------------------------
    def next_move(self) -> int:
        """Pick this round's move and remember each agent's pick for scoring."""
        agents = self._agents()
        rounds = len(self.opp)

        # Guardrail: if we are being clearly out-played, go unexploitable.
        if rounds >= 20 and self.net < -0.18 * rounds:
            self.last_recos = {}
            move = self.rng.randint(0, 2)
        elif not agents:
            self.last_recos = {}
            move = self.rng.randint(0, 2)
        else:
            best = max(agents, key=lambda k: (self.score.get(k, 0.0), self.rng.random()))
            self.last_recos = dict(agents)
            move = agents[best]

        self.mine.append(move)
        return move


def main(stdin=None, stdout=None) -> None:
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    bot = RPSBot()
    round_no = 0
    for raw in stdin:
        line = raw.rstrip("\n")
        sys.stderr.write(f"RAW[{round_no}]: {line!r}\n")
        sys.stderr.flush()
        if round_no > 0:
            opp_move = parse_move(line)
            if opp_move is not None:
                bot.record_opponent(opp_move)
        move = bot.next_move()
        stdout.write(NAMES[move] + "\n")
        stdout.flush()
        round_no += 1


if __name__ == "__main__":
    main()
