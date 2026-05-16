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
figgie_submission.py). The honest way to *search* a hidden-info game is a
`determinize()` sampler — `SearchBot` below does exactly this, driving the
kit's `SimultaneousMCTSBot` (decoupled-UCT for simultaneous-move games).
Note `ISMCTSBot` will NOT work here: it is turn-based and needs
`current_player` / `apply`, which a `SimultaneousState` does not have.

WHAT THIS FILE SHOWS, deepest last:
  - goal_belief / hand_value — the static informational edge from one hand.
  - BeliefTracker — refines that belief round by round from observed trades.
  - MarketMaker / Eager / Inference — fixed one-shot policies.
  - determinize_figgie + SearchBot — sample the hidden goal, then search.

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

from mathathon_kit import (
    RandomBot,
    RoundRobin,
    SimultaneousMCTSBot,
    benchmark_bot,
    validate_adapter,
)

Player = int
Action = Tuple  # ("pass",) | ("bid", suit, price) | ("ask", suit, price)
Trade = Tuple[int, int, int, int]  # (suit, buyer, seller, clearing_price)

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
    # Trades cleared in the PREVIOUS round -- public information every player
    # observes. A round-by-round belief tracker mines this; see BeliefTracker.
    last_trades: Tuple[Trade, ...] = ()

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
        trades: List[Trade] = []
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
                trades.append((suit, buyer, seller, trade))
                i += 1
                j += 1
        return FiggieState(
            hands=tuple(tuple(h) for h in hands),
            chips=tuple(chips),
            round=self.round + 1,
            twelve_suit=self.twelve_suit,
            max_rounds=self.max_rounds,
            players=self.players,
            last_trades=tuple(trades),
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


# --- Tracking the goal suit as the game unfolds -----------------------------
#
# goal_belief() reads a static snapshot of one hand. But every round you also
# see which suits TRADED (FiggieState.last_trades). Players who have inferred
# the goal tend to buy it, so trade volume in a suit is (noisy) evidence that
# it is the goal. BeliefTracker folds that evidence into the hand-only prior.


class BeliefTracker:
    """Posterior over the hidden goal suit, refined each round from observed
    trades. Seed it with your own hand, then call ``observe`` once per round
    with ``FiggieState.last_trades``.

    The trade signal is deliberately weak (``trade_weight`` is low): market
    makers also churn the abundant 12-card suit -- the goal's PARTNER, not the
    goal -- so trades are informative but noisy. This is a teaching model of
    Bayesian-style updating, not a calibrated likelihood. A suit our own hand
    has ruled out (probability 0) stays 0: evidence never resurrects it."""

    def __init__(self, hand: Sequence[int], trade_weight: float = 0.12) -> None:
        self.belief: List[float] = goal_belief(hand)
        self.trade_weight = trade_weight

    def observe(self, trades: Sequence[Trade]) -> List[float]:
        if trades:
            bump = [0.0, 0.0, 0.0, 0.0]
            for suit, *_rest in trades:
                bump[suit] += 1.0
            updated = [
                self.belief[s] * (1.0 + self.trade_weight * bump[s]) for s in range(4)
            ]
            total = sum(updated)
            if total > 0.0:
                self.belief = [p / total for p in updated]
        return self.belief

    def most_likely(self) -> int:
        return max(range(4), key=lambda s: self.belief[s])


# --- Search: determinize the hidden state, then run decoupled-UCT ------------


def _sample_index(weights: Sequence[float], rng: random.Random) -> int:
    """Sample an index in proportion to non-negative weights."""
    total = sum(weights)
    if total <= 0.0:
        return rng.randrange(len(weights))
    threshold = rng.random() * total
    cumulative = 0.0
    for index, weight in enumerate(weights):
        cumulative += weight
        if threshold <= cumulative:
            return index
    return len(weights) - 1


def determinize_figgie(
    state: FiggieState,
    player: Player,
    belief: Sequence[float],
    rng: random.Random,
) -> FiggieState:
    """Sample a full FiggieState consistent with what ``player`` knows.

    Keeps the player's own hand and the public chips / round / trade log,
    samples the hidden goal suit from ``belief``, and deals the rest of the
    deck across the other seats. Only private-to-player or public facts are
    read: the player's own hand, and how many cards each opponent holds (a
    public count -- every trade is observed). Opponents' SUITS are sampled,
    never copied off the true state -- so a search bot stays honest."""
    my_hand = state.hands[player]
    twelve = _PARTNER[_sample_index(belief, rng)]
    deck = _deck_counts(twelve)
    remaining: List[int] = []
    for suit in range(4):
        left = deck[suit] - my_hand[suit]
        if left < 0:  # belief should have ruled this out; bail out safely
            return state
        remaining.extend([suit] * left)
    rng.shuffle(remaining)
    hands: List[Tuple[int, ...]] = [()] * 4
    hands[player] = tuple(my_hand)
    cursor = 0
    for seat in range(4):
        if seat == player:
            continue
        drawn = [0, 0, 0, 0]
        for _ in range(sum(state.hands[seat])):  # card count is public
            drawn[remaining[cursor]] += 1
            cursor += 1
        hands[seat] = tuple(drawn)
    return FiggieState(
        hands=tuple(hands),
        chips=state.chips,
        round=state.round,
        twelve_suit=twelve,
        max_rounds=state.max_rounds,
        players=state.players,
        last_trades=state.last_trades,
    )


def _figgie_eval(state: FiggieState, player: Player) -> float:
    """Leaf evaluator for SearchBot's search. Inside a determinization the goal
    suit is fixed, so we can project the end-game payout from the *current*
    holdings -- the score() formula applied before the game ends. This is a
    sharp, low-variance signal: a random rollout to the end would smear one
    move's effect across seven noisy rounds and leave the search guessing."""
    goal = _PARTNER[state.twelve_suit]
    counts = [state.hands[p][goal] for p in range(4)]
    best = max(counts)
    winners = [p for p in range(4) if counts[p] == best]
    bonus = ANTE * 4 - sum(counts) * GOAL_CARD_VALUE
    value = state.chips[player] + state.hands[player][goal] * GOAL_CARD_VALUE
    if player in winners:
        value += bonus / len(winners)
    return float(value)


def _figgie_candidates(state: FiggieState, player: Player) -> List[Action]:
    """Prune the 41-action legal set to ~9 for the search: pass, plus one
    representative bid and one representative ask per suit. The suit-and-side
    choice is what decides the game; the exact price rung is a second-order
    knob the search can afford to drop. SimultaneousMCTSBot needs this -- thin
    MCTS cannot cover 41 arms per player on a per-move time budget."""
    legal = set(state.legal_actions(player))
    candidates: List[Action] = [("pass",)]
    for suit in range(4):
        # Highest affordable bid: midpoint clearing makes a top bid near-free,
        # and the high bidder wins the auction match.
        for price in (14, 12, 10, 8):
            if ("bid", suit, price) in legal:
                candidates.append(("bid", suit, price))
                break
        for price in (8, 10, 12, 14):  # lowest ask -- dump junk readily
            if ("ask", suit, price) in legal:
                candidates.append(("ask", suit, price))
                break
    return candidates


class SearchBot:
    """Decoupled-UCT search over a belief-sampled full game -- the deepest bot
    here. It tracks the goal-suit posterior across rounds (BeliefTracker) and
    each round runs SimultaneousMCTSBot: the determinizer samples the hidden
    goal suit and opponents' hands from that posterior, a pruned candidate set
    keeps the branching tractable, and a leaf evaluator projects the end-game
    payout. Unlike InferenceBot (a fixed one-shot bid) the move is the outcome
    of search against opponents who are themselves modelled as best-responding."""

    name = "search"

    def __init__(self, simulations: int = 1200) -> None:
        # `simulations` is only a cap -- the per-move time budget is the real
        # limiter. Keep it high so a generous budget is never wasted; the
        # search needs a few hundred sims before its edge is reliable.
        self.simulations = simulations
        self._tracker = None

    def choose_action(self, state, player, budget, rng):
        if state.round == 0 or self._tracker is None:
            self._tracker = BeliefTracker(state.hands[player])  # fresh game
        belief = self._tracker.observe(state.last_trades)
        engine = SimultaneousMCTSBot(
            determinize=lambda s, p, r: determinize_figgie(s, p, belief, r),
            evaluator=_figgie_eval,
            action_filter=_figgie_candidates,
            # Exploitative: your profit needs a counterparty who misprices a
            # card, so model opponents as noisy, not as flawless hoarders.
            opponent_policy=lambda s, p, r: r.choice(list(s.legal_actions(p))),
            simulations=self.simulations,
        )
        return engine.choose_action(state, player, budget, rng)


if __name__ == "__main__":
    print("=== Step 1: validate the adapter (mutation / hashable / termination) ===")
    print(validate_adapter(make_state(0), games=10).summary())

    print("\n=== Step 2: belief from one hand, then refined by observed trades ===")
    sample = make_state(0).hands[0]
    print(f"hand {sample}  ->  P(goal=suit) = {[round(p, 2) for p in goal_belief(sample)]}")
    tracker = BeliefTracker(sample)
    # Simulate two rounds where suit 1 is heavily traded -- watch belief shift.
    busy_round = ((1, 0, 2, 10), (1, 3, 0, 12), (1, 2, 1, 8))
    for r in (1, 2):
        tracker.observe(busy_round)
        print(f"after round {r} (suit 1 traded x3) -> "
              f"P(goal=suit) = {[round(p, 2) for p in tracker.belief]}")

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

    print("\n=== Step 5: the search bot -- decoupled-UCT over belief samples ===")
    # SearchBot runs SimultaneousMCTSBot every move: it samples the hidden goal
    # suit from its belief, best-responds against a noisy opponent model, and
    # scores leaves with the projected payout. It needs a few hundred sims per
    # move to be reliable, so this round-robin runs fewer games with a larger
    # per-move budget -- it takes ~45s, far longer than the naive bots above.
    #
    # Expect search to lead the field on AVERAGE SCORE and on Elo, and to run
    # level with the hand-tuned InferenceBot on win-rate. It gets there using
    # only the rules and a belief sampler -- no Figgie-specific bidding policy
    # is hand-written into it, unlike InferenceBot. The move limit is 0.10s;
    # SimultaneousMCTSBot self-limits to 80% of it for jitter headroom.
    search_bots = {
        "search": SearchBot(),
        "inference": InferenceBot(),
        "eager": EagerBot(),
        "random": RandomBot(),
    }
    search_report = RoundRobin(
        players=(0, 1, 2, 3),
        initial_state_factory=make_state,
        games_per_pair=80,
        simultaneous=True,
        time_limit_per_move=0.10,
    ).run(search_bots, seed=11)
    print(search_report.summary())
    print(search_report.significance())
    if search_report.error_samples:
        print(search_report.error_report())

    print("\n=== Step 6: per-move latency -- inference (cheap) vs search (heavy) ===")
    # One overrun forfeits a game: always benchmark the engine you ship.
    print(benchmark_bot(InferenceBot(), make_state, positions=40, time_limit=0.10).summary())
    print(benchmark_bot(SearchBot(), make_state, positions=20,
                        time_limit=0.10).summary())
