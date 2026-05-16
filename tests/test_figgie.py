"""The Figgie rehearsal game must pass the same pre-flight checks any
competition adapter does -- and its priced auction must actually trade."""

import random

from examples.figgie import (
    BeliefTracker,
    EagerBot,
    FiggieState,
    InferenceBot,
    MarketMakerBot,
    SearchBot,
    determinize_figgie,
    goal_belief,
    make_state,
)
from mathathon_kit import RandomBot, RoundRobin, TimeBudget, validate_adapter


def test_figgie_adapter_is_valid():
    report = validate_adapter(make_state(0), games=10)
    assert report.ok, report.summary()


def test_figgie_deck_is_well_formed():
    state = make_state(3)
    assert sum(sum(h) for h in state.hands) == 40
    assert all(sum(h) == 10 for h in state.hands)
    suit_totals = sorted(sum(h[s] for h in state.hands) for s in range(4))
    assert suit_totals == [8, 10, 10, 12]


def test_goal_belief_is_a_probability_distribution():
    belief = goal_belief((2, 3, 2, 3))
    assert abs(sum(belief) - 1.0) < 1e-9
    assert all(0.0 <= p <= 1.0 for p in belief)


def test_goal_belief_reads_a_lopsided_hand():
    # Holding 8 of suit 0 => suit 0 is almost certainly the 12-card suit =>
    # the goal is its same-colour partner, suit 1.
    belief = goal_belief((8, 0, 1, 1))
    assert belief[1] == max(belief)
    assert belief[1] > 0.8


def test_inference_bot_beats_random_head_to_head():
    # Over many games the inference edge should clear chance (0.25) vs noise.
    report = RoundRobin(
        players=(0, 1, 2, 3),
        initial_state_factory=make_state,
        games_per_pair=400,
        simultaneous=True,
        time_limit_per_move=0.02,
    ).run(
        {
            "inference": InferenceBot(),
            "eager": EagerBot(),
            "market_maker": MarketMakerBot(),
            "random": RandomBot(),
        },
        seed=3,
    )
    assert report.standings["inference"].win_rate > report.standings["random"].win_rate


def test_figgie_tournament_runs_clean():
    bots = {
        "inference": InferenceBot(),
        "market_maker": MarketMakerBot(),
        "eager": EagerBot(),
        "random": RandomBot(),
    }
    report = RoundRobin(
        players=(0, 1, 2, 3),
        initial_state_factory=make_state,
        games_per_pair=8,
        simultaneous=True,
        time_limit_per_move=0.02,
    ).run(bots, seed=1)
    assert sum(s.errors for s in report.standings.values()) == 0


def test_figgie_call_auction_executes_a_trade_at_the_midpoint():
    state = make_state(0)
    # Player 0 must own a card to sell; find one of its suits.
    suit = next(s for s in range(4) if state.hands[0][s] > 0)
    nxt = state.apply_joint({
        0: ("ask", suit, 8),    # seller wants >= 8
        1: ("bid", suit, 12),   # buyer willing to pay <= 12
        2: ("pass",),
        3: ("pass",),
    })
    assert nxt.hands[0][suit] == state.hands[0][suit] - 1
    assert nxt.hands[1][suit] == state.hands[1][suit] + 1
    # Trade clears at the (8 + 12) // 2 = 10 midpoint.
    assert nxt.chips[0] == state.chips[0] + 10
    assert nxt.chips[1] == state.chips[1] - 10


def test_figgie_payout_conserves_the_pot_even_with_trading():
    rng = random.Random(5)
    state = make_state(5)
    while not state.is_terminal():
        joint = {p: rng.choice(list(state.legal_actions(p))) for p in state.players}
        state = state.apply_joint(joint)
    # 4 * START_CHIPS ($400) + 8 goal cards * $10 + $120 bonus = $600, always.
    assert sum(state.score(p) for p in range(4)) == 600.0


# --- Deeper Figgie: belief tracking, determinization, search ----------------


def test_belief_tracker_seeds_from_the_opening_hand():
    # Holding 8 of suit 0 => suit 0 is the 12-card suit => goal is suit 1.
    assert BeliefTracker((8, 0, 1, 1)).most_likely() == 1


def test_belief_tracker_shifts_toward_a_traded_suit():
    tracker = BeliefTracker((2, 3, 3, 2))
    before = list(tracker.belief)
    suit = next(s for s in range(4) if before[s] > 0.0)
    tracker.observe([(suit, 0, 1, 10)] * 4)  # that suit traded heavily
    assert tracker.belief[suit] > before[suit]
    assert abs(sum(tracker.belief) - 1.0) < 1e-9


def test_belief_tracker_keeps_ruled_out_suits_at_zero():
    # Holding 9 of suit 0 is impossible if suit 0 were the 8-card goal suit,
    # so suit 0 is ruled out as the goal entirely.
    tracker = BeliefTracker((9, 0, 1, 0))
    zeros = [s for s in range(4) if tracker.belief[s] == 0.0]
    assert zeros  # a lopsided hand rules some goal suits out entirely
    tracker.observe([(s, 0, 1, 10) for s in zeros] * 3)
    assert all(tracker.belief[s] == 0.0 for s in zeros)


def test_determinize_figgie_preserves_knowledge_and_conserves_the_deck():
    state = make_state(0)
    sampled = determinize_figgie(state, 1, [0.25, 0.25, 0.25, 0.25], random.Random(2))
    assert sampled.hands[1] == state.hands[1]  # the player's own hand is kept
    assert sum(sum(h) for h in sampled.hands) == 40  # the full deck is dealt
    for p in range(4):  # each seat keeps its public card count
        assert sum(sampled.hands[p]) == sum(state.hands[p])
    # ...and the sampled deck is a valid 12 / 10 / 10 / 8 deck.
    suit_totals = sorted(sum(h[s] for h in sampled.hands) for s in range(4))
    assert suit_totals == [8, 10, 10, 12]


def test_search_bot_returns_a_legal_action():
    state = make_state(1)
    move = SearchBot().choose_action(state, 0, TimeBudget(0.05), random.Random(0))
    assert move in state.legal_actions(0)


def test_search_bot_round_robin_runs_clean():
    report = RoundRobin(
        players=(0, 1, 2, 3),
        initial_state_factory=make_state,
        games_per_pair=6,
        simultaneous=True,
        time_limit_per_move=0.03,
    ).run(
        {
            "search": SearchBot(),
            "inference": InferenceBot(),
            "market_maker": MarketMakerBot(),
            "random": RandomBot(),
        },
        seed=2,
    )
    assert sum(s.errors for s in report.standings.values()) == 0
