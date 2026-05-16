"""Kuhn Poker — the canonical 3-card imperfect-information game.

Rules:
- Deck = {1, 2, 3}. Each player antes 1 chip and gets one private card.
- Player 0 acts first: check (c) or bet (b).
- If 0 checks, 1 can check (showdown) or bet (then 0 can call/fold).
- If 0 bets, 1 can call (showdown) or fold (0 wins pot).
- Showdown: higher card wins the pot.

The optimal Nash policy for player 0 is well-known (alpha in [0, 1/3]):
- Hold J: bet with prob alpha, else check
- Hold Q: always check (and call with prob alpha + 1/3)
- Hold K: bet with prob 3*alpha, else check

We use this as a smoke test for ISMCTSBot. The bot doesn't need to find the
exact Nash policy — it just needs to substantially beat random play, which
is a low bar that any working ISMCTS implementation should clear.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathathon_kit import (
    GameResult,
    ISMCTSBot,
    RandomBot,
    Simulator,
    TimeBudget,
)


CHECK, BET, CALL, FOLD = "check", "bet", "call", "fold"
DECK = (1, 2, 3)


@dataclass(frozen=True)
class KuhnState:
    """Full state including both hidden cards (used in determinizations)."""

    cards: Tuple[int, int]
    history: Tuple[str, ...] = ()
    players: Tuple[int, int] = (0, 1)

    @property
    def current_player(self) -> int:
        return len(self.history) % 2

    def legal_actions(self, player: Optional[int] = None) -> Sequence[str]:
        if self.is_terminal():
            return ()
        if not self.history:
            return (CHECK, BET)
        last = self.history[-1]
        if last == CHECK and len(self.history) == 1:
            return (CHECK, BET)
        if last == BET:
            return (CALL, FOLD)
        return ()

    def apply(self, action: str) -> "KuhnState":
        return KuhnState(cards=self.cards, history=self.history + (action,))

    def is_terminal(self) -> bool:
        h = self.history
        if not h:
            return False
        if h == (CHECK, CHECK):
            return True
        if len(h) >= 2 and h[-1] in (CALL, FOLD):
            return True
        return False

    def score(self, player: int) -> float:
        if not self.is_terminal():
            return 0.0
        h = self.history
        # Pot = 2 antes plus any bet calls.
        if h == (CHECK, CHECK):
            winner = 0 if self.cards[0] > self.cards[1] else 1
            return 1.0 if player == winner else -1.0
        if h[-1] == FOLD:
            # Last actor folded; the other actor wins the pot of 2 antes.
            winner = (len(h) - 1 + 1) % 2  # actor who didn't fold
            # Actually: bettor wins. If history is (check, bet, fold), bettor=1, folder=0.
            # If history is (bet, fold), bettor=0, folder=1.
            folder = (len(h) - 1) % 2
            winner = 1 - folder
            return 1.0 if player == winner else -1.0
        if h[-1] == CALL:
            # Showdown for 2 chips after a call.
            winner = 0 if self.cards[0] > self.cards[1] else 1
            return 2.0 if player == winner else -2.0
        return 0.0


@dataclass(frozen=True)
class KuhnPublicState:
    """What a single player can observe."""

    my_card: int
    history: Tuple[str, ...] = ()
    me: int = 0
    players: Tuple[int, int] = (0, 1)

    @property
    def current_player(self) -> int:
        return len(self.history) % 2

    def legal_actions(self, player: Optional[int] = None) -> Sequence[str]:
        return KuhnState(cards=(self.my_card, 0), history=self.history).legal_actions()

    def is_terminal(self) -> bool:
        return KuhnState(cards=(0, 0), history=self.history).is_terminal()


def determinize(public: KuhnPublicState, player: int, rng: random.Random) -> KuhnState:
    """Sample the opponent's card uniformly from the deck minus my card."""
    opp_pool = [c for c in DECK if c != public.my_card]
    opp_card = rng.choice(opp_pool)
    cards = (public.my_card, opp_card) if player == 0 else (opp_card, public.my_card)
    return KuhnState(cards=cards, history=public.history)


def make_full_state(seed: int) -> KuhnState:
    r = random.Random(seed)
    deck = list(DECK)
    r.shuffle(deck)
    return KuhnState(cards=(deck[0], deck[1]))


@dataclass
class _ISMCTSWrapper:
    """Adapts ISMCTSBot to the simulator (which passes the full KuhnState).

    On each turn we project the full state to the bot's information set and
    let ISMCTS sample determinizations.
    """

    bot: ISMCTSBot
    name: str = "ismcts"

    def choose_action(self, state: KuhnState, player: int, budget: TimeBudget, rng: random.Random):
        public = KuhnPublicState(
            my_card=state.cards[player], history=state.history, me=player
        )
        # Wrap the determinize so it preserves whose move it is.
        original = self.bot.determinize

        def wrapped(pub, p, r):
            return original(pub, p, r)

        return self.bot.choose_action(public, player, budget, rng)


if __name__ == "__main__":
    sim = Simulator(players=(0, 1), max_turns=20, time_limit_per_move=0.2)
    ismcts_bot = ISMCTSBot(determinize=determinize, simulations=400)
    wrapped = _ISMCTSWrapper(bot=ismcts_bot, name="ismcts")

    # Smoke test 1: ISMCTS should beat random comfortably.
    n_games = 200
    score_ismcts = 0.0
    score_random = 0.0
    for seed in range(n_games):
        for swap in (False, True):
            bots = (
                {0: RandomBot(), 1: wrapped} if swap else {0: wrapped, 1: RandomBot()}
            )
            r = sim.play(make_full_state(seed), bots, seed=seed)
            ismcts_idx = 1 if swap else 0
            random_idx = 0 if swap else 1
            score_ismcts += r.scores[ismcts_idx]
            score_random += r.scores[random_idx]

    total = 2 * n_games
    print(f"games: {total}")
    print(f"ISMCTS avg payoff: {score_ismcts / total:+.3f}")
    print(f"random avg payoff: {score_random / total:+.3f}")
    assert score_ismcts > score_random, "ISMCTS should beat random in Kuhn poker"
    print("PASS: ISMCTSBot is functional.")
