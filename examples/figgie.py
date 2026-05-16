"""Simplified Jane Street **Figgie** — the kit's end-to-end rehearsal game.

Figgie is a 4-player, simultaneous-move, hidden-information game. Real Figgie
is a 4-minute continuous double auction; the kit's adapters are round-based,
so the FIRST design decision is discretisation — here each round is one
simultaneous quote per player, cleared as a call auction.

You win money two ways, and a faithful model needs BOTH:
  1. GOAL CARDS  — at the end each goal card pays $10 from the $200 pot and
     the largest holder takes the bonus.
  2. THE SPREAD  — buying a card below fair value or selling it above pays
     immediately in chips. (An earlier draft of this file priced every trade
     at a flat $10; that silently deleted reason 2 — the whole point of
     trading. validate_adapter cannot catch a modelling error like that;
     only knowing the game can.)

Deck (40 cards, 4 suits): one suit has 12 cards, its SAME-COLOUR partner has
8, the other colour's two suits have 10 each. The 8-card suit (same colour as
the 12-card suit) is the hidden GOAL suit.

HIDDEN-INFO NOTE: the kit hands the full state to every bot, so `twelve_suit`
and other players' hands are technically visible — a well-behaved bot must
not read them. A real submission gets only its own observation (see
figgie_submission.py); for search you would add an ISMCTS `determinize()`.

SIMPLIFICATIONS (documented on purpose): one quote per player per round, and
a single round-by-round call auction rather than a continuous order book.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import List, Mapping, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathathon_kit import RandomBot, RoundRobin, benchmark_bot, validate_adapter

Player = int
Action = Tuple  # ("pass",) | ("bid", suit, price) | ("ask", suit, price)

_PARTNER = {0: 1, 1: 0, 2: 3, 3: 2}  # same-colour suit pairs (0,1 black; 2,3 red)
PRICES = (6, 8, 10, 12, 14)  # quote ladder; 10 is the suit-blind "fair" value
ANTE = 50
GOAL_CARD_VALUE = 10
START_CHIPS = 100
MAX_ROUNDS = 8


@dataclass(frozen=True)
class FiggieState:
    """Round-based simultaneous Figgie with a priced call auction. Frozen +
    tuple fields => hashable, so it is cache-safe and passes validate_adapter."""

    hands: Tuple[Tuple[int, int, int, int], ...]
    chips: Tuple[int, ...]
    round: int
    twelve_suit: int
    max_rounds: int = MAX_ROUNDS
    players: Tuple[Player, ...] = (0, 1, 2, 3)

    def active_players(self) -> Sequence[Player]:
        return () if self.is_terminal() else self.players

    def legal_actions(self, player: Player) -> Sequence[Action]:
        actions = [("pass",)]
        hand = self.hands[player]
        chips = self.chips[player]
        for suit in range(4):
            for price in PRICES:
                if chips >= price:
                    actions.append(("bid", suit, price))
                if hand[suit] > 0:
                    actions.append(("ask", suit, price))
        return tuple(actions)

    def apply_joint(self, actions: Mapping[Player, Action]) -> "FiggieState":
        # Per-suit call auction: best bids vs best asks. A trade clears at the
        # midpoint, so a buyer who bids high pays less than its limit and a
        # seller who asks low earns more than its limit -- that gap is the
        # spread profit. Sorting is index-tie-broken, so this is deterministic.
        hands = [list(h) for h in self.hands]
        chips = list(self.chips)
        for suit in range(4):
            bids = sorted(
                (
                    (a[2], p)
                    for p, a in actions.items()
                    if a[0] == "bid" and a[1] == suit and self.chips[p] >= a[2]
                ),
                key=lambda bp: (-bp[0], bp[1]),
            )
            asks = sorted(
                (
                    (a[2], p)
                    for p, a in actions.items()
                    if a[0] == "ask" and a[1] == suit and self.hands[p][suit] > 0
                ),
                key=lambda ap: (ap[0], ap[1]),
            )
            i = j = 0
            while i < len(bids) and j < len(asks) and bids[i][0] >= asks[j][0]:
                bid_price, buyer = bids[i]
                ask_price, seller = asks[j]
                trade = (bid_price + ask_price) // 2
                hands[buyer][suit] += 1
                hands[seller][suit] -= 1
                chips[buyer] -= trade
                chips[seller] += trade
                i += 1
                j += 1
        return FiggieState(
            hands=tuple(tuple(h) for h in hands),
            chips=tuple(chips),
            round=self.round + 1,
            twelve_suit=self.twelve_suit,
            max_rounds=self.max_rounds,
            players=self.players,
        )

    def is_terminal(self) -> bool:
        return self.round >= self.max_rounds

    def score(self, player: Player) -> float:
        if not self.is_terminal():
            # Mid-game the goal suit is hidden, so chips-on-hand (which already
            # reflects spread profit) is the only honest proxy.
            return float(self.chips[player])
        goal = _PARTNER[self.twelve_suit]
        counts = [self.hands[p][goal] for p in range(4)]
        best = max(counts)
        winners = [p for p in range(4) if counts[p] == best]
        bonus = ANTE * 4 - sum(counts) * GOAL_CARD_VALUE
        payout = self.chips[player] + self.hands[player][goal] * GOAL_CARD_VALUE
        if player in winners:
            payout += bonus / len(winners)
        return float(payout)


def make_state(seed: int) -> FiggieState:
    """Deal a fresh game: pick the 12-card suit, build the 12/8/10/10 deck,
    shuffle, and deal 10 cards to each of the 4 players."""
    rng = random.Random(seed)
    twelve = rng.randrange(4)
    partner = _PARTNER[twelve]
    others = [s for s in range(4) if s not in (twelve, partner)]
    counts = {twelve: 12, partner: 8, others[0]: 10, others[1]: 10}
    deck = [suit for suit, count in counts.items() for _ in range(count)]
    rng.shuffle(deck)
    hands = [[0, 0, 0, 0] for _ in range(4)]
    for index, card in enumerate(deck):
        hands[index % 4][card] += 1
    return FiggieState(
        hands=tuple(tuple(h) for h in hands),
        chips=(START_CHIPS,) * 4,
        round=0,
        twelve_suit=twelve,
    )


# --- The informational edge: infer the hidden goal suit ----------------------
#
# The goal suit is hidden, but your own 10 cards are evidence. The deck has
# suit sizes 12 / 10 / 10 / 8, so your hand is a draw from one of four decks
# (one per "which suit holds 12"). The multivariate-hypergeometric likelihood
# of your hand under each deck gives a posterior over the 12-card suit -- and
# the goal suit is its same-colour partner. Holding FEW of a suit points at it
# being the scarce 8-card goal suit; holding MANY of a suit points the goal at
# that suit's partner. This is private information, fair for a bot to use.


def _deck_counts(twelve_suit: int) -> dict:
    counts = {s: 10 for s in range(4)}
    counts[twelve_suit] = 12
    counts[_PARTNER[twelve_suit]] = 8
    return counts


def goal_belief(hand: Sequence[int]) -> List[float]:
    """Posterior P(goal suit = s) for s in 0..3, from this hand alone."""
    posterior = []
    for twelve in range(4):
        deck = _deck_counts(twelve)
        likelihood = 1.0
        for suit in range(4):
            if hand[suit] > deck[suit]:
                likelihood = 0.0
                break
            likelihood *= comb(deck[suit], hand[suit])
        posterior.append(likelihood)
    total = sum(posterior)
    if total == 0.0:
        return [0.25, 0.25, 0.25, 0.25]
    posterior = [p / total for p in posterior]
    # goal(t) = partner(t), and partner is an involution, so
    # P(goal = g) = P(twelve = partner(g)).
    return [posterior[_PARTNER[g]] for g in range(4)]


def hand_value(hand: Sequence[int], chips: int, belief: Sequence[float]) -> float:
    """Belief-weighted expected end value -- the evaluator a search bot would
    maximise. Goal cards are worth far more than the flat $10 a naive score()
    assigns them: each also buys a shot at the $120 bonus. (Crude bonus model:
    holding >= 3 of the goal suit usually takes the majority.)"""
    value = float(chips)
    bonus = ANTE * 4 - 8 * GOAL_CARD_VALUE
    for suit in range(4):
        cards = hand[suit]
        share = bonus if cards >= 3 else 0.0
        value += belief[suit] * (cards * GOAL_CARD_VALUE + share)
    return value


# --- Bots written for simultaneous play (a policy, no lookahead) -------------


class MarketMakerBot:
    """Quotes a spread: offers inventory dear, then bids for cards cheap. It
    earns the gap whenever an eager counterparty crosses its quote."""

    name = "market_maker"

    def choose_action(self, state, player, budget, rng):
        hand = state.hands[player]
        legal = set(state.legal_actions(player))
        if state.round % 2 == 0:  # sell the suit we hold most of, dear
            suit = max(range(4), key=lambda s: hand[s])
            for price in (14, 12):
                if ("ask", suit, price) in legal:
                    return ("ask", suit, price)
        else:  # buy the suit we hold least of, cheap
            suit = min(range(4), key=lambda s: hand[s])
            for price in (6, 8):
                if ("bid", suit, price) in legal:
                    return ("bid", suit, price)
        return ("pass",)


class EagerBot:
    """Chases the scarce suit and bids the top of the ladder -- it accumulates
    fast but overpays, handing the spread to whoever sells to it."""

    name = "eager"

    def choose_action(self, state, player, budget, rng):
        hand = state.hands[player]
        legal = set(state.legal_actions(player))
        suit = min(range(4), key=lambda s: hand[s])
        for price in (14, 12, 10):
            if ("bid", suit, price) in legal:
                return ("bid", suit, price)
        return ("pass",)


class PassBot:
    """Never trades — the do-nothing baseline."""

    name = "passer"

    def choose_action(self, state, player, budget, rng):
        return ("pass",)


class InferenceBot:
    """The bot with an edge: estimates the hidden goal suit from its own hand,
    then BUYS the believed goal suit (bidding as high as its confidence
    justifies) and ASKS away the suit it is most sure is junk. Pays the
    midpoint, so a high bid wins the match without overpaying."""

    name = "inference"

    def choose_action(self, state, player, budget, rng):
        hand = state.hands[player]
        belief = goal_belief(hand)
        legal = set(state.legal_actions(player))
        goal = max(range(4), key=lambda s: belief[s])
        confidence = belief[goal]
        # The more sure we are, the higher up the ladder we will bid.
        if confidence > 0.45:
            ceiling = 14
        elif confidence > 0.30:
            ceiling = 12
        elif confidence > 0.22:
            ceiling = 10
        else:
            ceiling = 8
        for price in reversed(PRICES):  # bid as high as confidence allows
            if price <= ceiling and ("bid", goal, price) in legal:
                return ("bid", goal, price)
        junk = min(range(4), key=lambda s: belief[s])
        for price in (14, 12, 10):  # offload the believed non-goal suit, dear
            if ("ask", junk, price) in legal:
                return ("ask", junk, price)
        return ("pass",)


if __name__ == "__main__":
    print("=== Step 1: validate the adapter (mutation / hashable / termination) ===")
    print(validate_adapter(make_state(0), games=10).summary())

    print("\n=== Step 2: sample goal-suit belief from one hand ===")
    sample = make_state(0).hands[0]
    print(f"hand {sample}  ->  P(goal=suit) = {[round(p, 2) for p in goal_belief(sample)]}")

    print("\n=== Step 3: 4-player round-robin -- naive bots vs the inference bot ===")
    bots = {
        "inference": InferenceBot(),
        "market_maker": MarketMakerBot(),
        "eager": EagerBot(),
        "random": RandomBot(),
    }
    report = RoundRobin(
        players=(0, 1, 2, 3),
        initial_state_factory=make_state,
        games_per_pair=800,
        simultaneous=True,
        time_limit_per_move=0.02,
    ).run(bots, seed=7)
    print(report.summary())

    print("\n=== Step 4: is the inference bot's edge real, or noise? ===")
    # significance() infers the chance win-rate (1/seats = 0.25 here) itself.
    print(report.significance())
    if report.error_samples:
        print(report.error_report())

    print("\n=== Step 5: per-move latency (works on simultaneous games too) ===")
    print(benchmark_bot(InferenceBot(), make_state, positions=40, time_limit=0.02).summary())
