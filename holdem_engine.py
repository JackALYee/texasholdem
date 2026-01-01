#!/usr/bin/env python3
"""
Texas Hold'em (CLI) — Multi-round table
- Supports multiple HUMAN players (pass-and-play privacy): each human only sees their own hole cards + board.
- Competitive AIs with multiple archetypes + occasional 3-bets / 4-bets.
- Cheat/command system accepted at ANY prompt (commands are prioritized).

-----------------------
CHEAT CODES (any prompt)
-----------------------
  reveal
      Reveal AI hole cards ONCE (next render only, then auto-hide).

  add <amount>
      Add chips to YOUR (Hero) stack immediately.

  show out
      Reveal AI hole cards ONCE + show your outs (if applicable) + show win rates for ALL active players.

  explain type
      Show how many AI types exist + counts at the current table + short descriptions.
  explain type <type name>
      Show details for one type.

  <AI name> - <AI type>
      Replace an AI player's type at the table.
      Example:  "AI 2 - Maniac"   or  "AI 5 - Calling Station"

  new player <amount>
      Add a NEW AI with that stack (joins next hand).
  new player <name> <amount>
      Add a named AI with that stack (joins next hand).

  all in   (during your turn)
  help
      Show this help.

-----------------------
RULES YOU REQUESTED
-----------------------
1) If there is NO raise (preflop unopened pot), nobody is allowed to fold (humans + AIs).
   - Practically: before the first raise preflop, players can only call/raise (no fold).

2) AIs will 3-bet / 4-bet occasionally, depending on their type, equity, and randomness.

3) "explain type" tells you how many AI types exist and what they do.

4) You can replace an AI type with "AI name - AI type" at any prompt.

-----------------------
DEPENDENCIES
-----------------------
- treys (required)
- rich  (optional, nicer UI)

Install:
  python3 -m venv .venv
  source .venv/bin/activate     # macOS/Linux
  pip install treys rich

Run:
  python3 holdem_cli_v7.py

Note:
- This is a simplified no-limit engine (single pot; no side pots). All-ins are supported but side pots are not split perfectly.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

# Optional rich UI
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    from rich.table import Table
    RICH_OK = True
except Exception:
    RICH_OK = False

from treys import Card, Evaluator, Deck

EVAL = Evaluator()

SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_NAME = {"s": "Spades", "h": "Hearts", "d": "Diamonds", "c": "Clubs"}
RANK_NAME = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7", "8": "8", "9": "9",
    "T": "10", "J": "Jack", "Q": "Queen", "K": "King", "A": "Ace"
}


def treys_to_rs(card_int: int) -> Tuple[str, str]:
    s = Card.int_to_str(card_int)  # e.g. "Kh"
    return s[0], s[1]


def unicode_card(card_int: int) -> str:
    r, s = treys_to_rs(card_int)
    return f"{r}{SUIT_SYMBOL[s]}"


def card_label(card_int: int) -> str:
    r, s = treys_to_rs(card_int)
    return f"{RANK_NAME[r]} of {SUIT_NAME[s]}"


def cards_short(cards: List[int]) -> str:
    """Compact card strings for logs, like 'Ah Kd 7s'."""
    return " ".join(Card.int_to_str(c) for c in cards)


def is_red_suit(suit: str) -> bool:
    return suit in ("h", "d")


def clear_screen():
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()


# --------------------------
# AI Types (common archetypes)
# --------------------------

@dataclass
class AIType:
    name: str
    tightness: float     # 0..1 (higher = plays fewer hands; folds more vs raises)
    aggression: float    # 0..1 (higher = raises more / bets more)
    bluff: float         # 0..1 (higher = bluffs more)
    risk: float          # 0..1 (higher = accepts thinner edges, bigger variance)
    threebet: float      # baseline tendency to 3-bet
    fourbet: float       # baseline tendency to 4-bet
    description: str


def ai_type_registry() -> Dict[str, AIType]:
    types = [
        AIType("TAG", 0.72, 0.72, 0.18, 0.45, 0.22, 0.10,
               "Tight-aggressive: solid ranges, raises good hands, fewer spews."),
        AIType("LAG", 0.35, 0.85, 0.40, 0.75, 0.35, 0.18,
               "Loose-aggressive: wide ranges, high pressure, more 3-bets and barrels."),
        AIType("Nit", 0.92, 0.55, 0.07, 0.25, 0.12, 0.05,
               "Very tight, risk-averse: folds a lot vs aggression, straightforward value."),
        AIType("Rock", 0.85, 0.30, 0.06, 0.20, 0.10, 0.03,
               "Tight-passive: waits for premiums; rarely raises without strength."),
        AIType("Calling Station", 0.40, 0.18, 0.04, 0.35, 0.06, 0.02,
               "Loose-passive: calls too much, rarely raises, dislikes folding."),
        AIType("Maniac", 0.15, 0.98, 0.70, 0.95, 0.60, 0.35,
               "Ultra loose + ultra aggressive: raises constantly; overbets and re-raises often."),
        AIType("Trapper", 0.70, 0.45, 0.20, 0.40, 0.15, 0.10,
               "Slow-plays strong hands, calls to disguise strength, then raises later."),
        AIType("Grinder", 0.62, 0.55, 0.15, 0.35, 0.18, 0.08,
               "Steady, low-variance: picks spots, avoids huge bluffs, plays fundamentally."),
        AIType("Gambler", 0.32, 0.65, 0.35, 0.85, 0.30, 0.14,
               "Swings for the fences: action-oriented, chasey, calls/raises marginally."),
        AIType("Whale", 0.20, 0.25, 0.10, 0.70, 0.08, 0.04,
               "Very loose and passive: calls wide, gives action, big variance."),
        AIType("OMC", 0.90, 0.40, 0.03, 0.15, 0.08, 0.03,
               "Old Man Coffee: super tight, hates bluffing, big bets mean monsters."),
        AIType("Shark", 0.60, 0.75, 0.28, 0.55, 0.28, 0.14,
               "Strong reg: balanced-ish, pressures in spots, can 3-bet/4-bet bluff."),
        AIType("Tricky", 0.55, 0.60, 0.55, 0.55, 0.25, 0.16,
               "Creative lines: check-raises, bluffs more, mixes sizes and timing."),
    ]
    return {t.name.lower(): t for t in types}


AI_TYPES = ai_type_registry()


def normalize_type_name(s: str) -> Optional[str]:
    key = s.strip().lower()
    aliases = {
        "tight aggressive": "tag",
        "tight-aggressive": "tag",
        "loose aggressive": "lag",
        "loose-aggressive": "lag",
        "old man coffee": "omc",
        "oldmancoffee": "omc",
        "callingstation": "calling station",
        "station": "calling station",
    }
    if key in aliases:
        key = aliases[key]
    return key if key in AI_TYPES else None


# --------------------------
# Game state
# --------------------------

@dataclass
class Player:
    name: str
    is_human: bool
    ai_type_key: Optional[str] = None  # key into AI_TYPES if AI


@dataclass
class HandState:
    players: List[Player]                # seat order
    dealer_idx: int
    sb: int
    bb: int
    stacks: Dict[str, int]               # snapshot stacks
    hole: Dict[str, List[int]]           # name -> [c1, c2]
    board5: List[int]                    # 5 community cards pre-generated
    revealed: List[int] = field(default_factory=list)
    street: str = "preflop"              # preflop/flop/turn/river/showdown
    pot: int = 0
    folded: Set[str] = field(default_factory=set)
    allin: Set[str] = field(default_factory=set)

    # betting-round state
    current_bet: int = 0
    contrib: Dict[str, int] = field(default_factory=dict)
    total_contrib: Dict[str, int] = field(default_factory=dict)  # total across the whole hand (for side pots)
    raise_count: int = 0
    last_raise_size: int = 0
    log: List[str] = field(default_factory=list)
    winner_text: Optional[str] = None
    hand_done: bool = False


@dataclass
class TableState:
    players: List[Player]
    stacks: Dict[str, int]
    hero_name: str
    next_ai_id: int = 1

    # restart control
    starting_stack: int = 500
    sb: int = 5
    initial_specs: List[Tuple[str, bool, Optional[str]]] = field(default_factory=list)
    request_restart: Optional[str] = None  # 'same' or 'shuffle'

    reveal_once: bool = False
    overlay_once: bool = False
    overlay_text: Optional[str] = None


# --------------------------
# Helpers
# --------------------------

def rotate_dealer(dealer: int, n: int) -> int:
    return (dealer + 1) % n


def next_idx(idx: int, n: int) -> int:
    return (idx + 1) % n


def active_nonfolded(h: HandState) -> List[str]:
    return [p.name for p in h.players if p.name not in h.folded and p.name not in h.allin]


# --------------------------
# Dealing / progression
# --------------------------

def post_blinds(h: HandState):
    n = len(h.players)
    sb_idx = next_idx(h.dealer_idx, n)
    bb_idx = next_idx(sb_idx, n)
    sb_name = h.players[sb_idx].name
    bb_name = h.players[bb_idx].name

    sb_amt = min(h.sb, h.stacks[sb_name])
    bb_amt = min(h.bb, h.stacks[bb_name])

    h.stacks[sb_name] -= sb_amt
    h.stacks[bb_name] -= bb_amt
    h.pot += sb_amt + bb_amt

    h.contrib = {p.name: 0 for p in h.players}
    h.total_contrib = {p.name: 0 for p in h.players}
    h.contrib[sb_name] = sb_amt
    h.contrib[bb_name] = bb_amt
    h.total_contrib[sb_name] = sb_amt
    h.total_contrib[bb_name] = bb_amt
    h.current_bet = bb_amt
    h.last_raise_size = h.bb
    h.raise_count = 0

    if h.stacks[sb_name] == 0:
        h.allin.add(sb_name)
    if h.stacks[bb_name] == 0:
        h.allin.add(bb_name)

    h.log.append(f"{sb_name} posts SB {sb_amt}. {bb_name} posts BB {bb_amt}.")


def deal_new_hand(t: TableState, dealer_idx: int, sb: int) -> HandState:
    bb = sb * 2
    deck = Deck()
    hole = {p.name: [deck.draw(1)[0], deck.draw(1)[0]] for p in t.players}
    board5 = [deck.draw(1)[0] for _ in range(5)]
    stacks_snapshot = dict(t.stacks)

    h = HandState(players=t.players, dealer_idx=dealer_idx, sb=sb, bb=bb, stacks=stacks_snapshot, hole=hole, board5=board5)
    post_blinds(h)
    return h


def reveal_next_street(h: HandState):
    if h.street == "preflop":
        h.revealed = h.board5[:3]
        h.street = "flop"
        h.log.append(f"Flop: {cards_short(h.revealed)}")
    elif h.street == "flop":
        h.revealed = h.board5[:4]
        h.street = "turn"
        h.log.append(f"Turn: {cards_short(h.revealed)}")
    elif h.street == "turn":
        h.revealed = h.board5[:5]
        h.street = "river"
        h.log.append(f"River: {cards_short(h.revealed)}")
    elif h.street == "river":
        h.street = "showdown"
        h.log.append(f"Showdown. Board: {cards_short(h.board5[:5])}")

    h.contrib = {p.name: 0 for p in h.players}
    h.current_bet = 0
    h.raise_count = 0
    h.last_raise_size = h.bb




def resolve_showdown(h: HandState):
    # If everyone but one folded, award the whole pot immediately.
    alive = [p for p in h.players if p.name not in h.folded]
    if len(alive) == 1:
        w = alive[0].name
        h.stacks[w] += h.pot
        msg = f"{w} player win because everyone folds"
        h.winner_text = msg
        h.log.append(msg)
        h.pot = 0
        h.hand_done = True
        return

    if h.street != "showdown":
        return

    board = h.board5[:5]

    # Side-pot construction: peel chips from everyone from the bottom up.
    # IMPORTANT: if the top layer has only one contributor, that portion is an uncalled bet
    # and should be returned to that player (not awarded as a pot).
    contrib = dict(h.total_contrib)  # total chips put in by each player this hand
    seat_order = [p.name for p in h.players]

    def best_hand_winners(eligible: List[str]) -> List[str]:
        scores = [(EVAL.evaluate(board, h.hole[nm]), nm) for nm in eligible]
        scores.sort()
        best = scores[0][0]
        return [nm for sc, nm in scores if sc == best]

    payouts: Dict[str, int] = {p.name: 0 for p in h.players}
    refund_total = 0
    pots: List[Tuple[int, List[str]]] = []

    # Work on remaining contributions
    remaining = {nm: amt for nm, amt in contrib.items() if amt > 0}

    while True:
        contributors = [nm for nm, amt in remaining.items() if amt > 0]
        if not contributors:
            break

        if len(contributors) == 1:
            # Uncalled bet: return leftover to that player.
            lone = contributors[0]
            refund = remaining[lone]
            if refund > 0:
                h.stacks[lone] += refund
                refund_total += refund
                remaining[lone] = 0
                h.log.append(f"Uncalled bet returned: {lone} gets {refund}.")
            break

        mmin = min(remaining[nm] for nm in contributors)
        pot_amt = mmin * len(contributors)
        for nm in contributors:
            remaining[nm] -= mmin

        eligible = [nm for nm in contributors if nm not in h.folded]
        # If somehow nobody is eligible (shouldn't happen), skip awarding.
        pots.append((pot_amt, eligible))

    # Award pots in order
    total_awarded = 0
    for pot_index, (pot_amt, eligible) in enumerate(pots):
        if pot_amt <= 0:
            continue
        if not eligible:
            # nobody eligible; leave it (should not happen)
            continue

        winners = best_hand_winners(eligible)
        split = pot_amt // len(winners)
        rem = pot_amt - split * len(winners)

        for w in winners:
            payouts[w] += split
        total_awarded += pot_amt

        if rem > 0:
            ordered = [nm for nm in seat_order if nm in winners]
            for k in range(rem):
                payouts[ordered[k % len(ordered)]] += 1

        # Log pot result briefly
        if len(pots) == 1:
            pot_name = "Main pot"
        else:
            pot_name = "Main pot" if pot_index == 0 else f"Side pot #{pot_index}"
        h.log.append(f"{pot_name} {pot_amt} eligible: {', '.join(eligible)}; winners: {', '.join(winners)}.")

    # Apply payouts
    for nm, amt in payouts.items():
        if amt > 0:
            h.stacks[nm] += amt

    # Sanity: adjust h.pot to reflect refunds (so winner text matches money moved)
    if refund_total > 0:
        h.pot = max(0, h.pot - refund_total)

    # Winner text (summary)
    if payouts:
        top_amt = max(payouts.values())
        top_winners = [nm for nm, amt in payouts.items() if amt == top_amt and amt > 0]
        if len(top_winners) == 1:
            h.winner_text = f"{top_winners[0]} wins the most ({top_amt}) — side pots resolved."
        else:
            h.winner_text = f"Side pots resolved. Top payout: {', '.join(top_winners)} ({top_amt})."
    else:
        h.winner_text = "Side pots resolved."

    # Detailed payout lines
    h.log.append("Payouts:")
    for nm in seat_order:
        if payouts.get(nm, 0) > 0:
            h.log.append(f"  {nm}: +{payouts[nm]}")

    h.log.append(h.winner_text)
    h.pot = 0
    h.hand_done = True

def board_texture(h: HandState) -> float:
    b = h.revealed
    if len(b) < 3:
        return 0.5

    suits = [treys_to_rs(c)[1] for c in b]
    max_suit = max(suits.count(s) for s in set(suits))
    suit_score = 0.9 if max_suit >= 3 else (0.6 if max_suit == 2 else 0.2)

    ranks = []
    for c in b:
        r, _ = treys_to_rs(c)
        ranks.append("23456789TJQKA".index(r))
    ranks = sorted(set(ranks))

    adj = 0.0
    for i in range(1, len(ranks)):
        d = ranks[i] - ranks[i - 1]
        if d == 1:
            adj += 1.0
        elif d == 2:
            adj += 0.5
    conn_score = min(1.0, adj / 2.0)

    return max(0.0, min(1.0, 0.55 * suit_score + 0.45 * conn_score))


def estimate_equity_vs_random(h: HandState, player: str, iters: int = 140) -> float:
    known_board = list(h.revealed)
    my_hand = list(h.hole[player])

    alive = [p.name for p in h.players if p.name not in h.folded and p.name != player]
    n_opponents = max(1, len(alive))

    used = set(known_board + my_hand)
    full_deck = [Card.new(r + s) for r in "23456789TJQKA" for s in "shdc"]
    remain = [c for c in full_deck if c not in used]

    wins = ties = 0
    need = 5 - len(known_board)

    for _ in range(iters):
        random.shuffle(remain)
        idx = 0
        opp_holes = []
        for _k in range(n_opponents):
            opp_holes.append([remain[idx], remain[idx + 1]])
            idx += 2

        runout = known_board + remain[idx: idx + need]

        my_score = EVAL.evaluate(runout, my_hand)
        opp_scores = [EVAL.evaluate(runout, h2) for h2 in opp_holes]
        best = min([my_score] + opp_scores)

        if my_score == best:
            if opp_scores.count(best) == 0:
                wins += 1
            else:
                ties += 1

    return (wins + 0.5 * ties) / iters


def compute_win_rates_true(h: HandState, iters: int = 700) -> Dict[str, float]:
    alive = [p.name for p in h.players if p.name not in h.folded]
    if not alive:
        return {}

    k = len(h.revealed)
    if h.street == "showdown" or k >= 5:
        board = h.board5[:5]
        scores = [(EVAL.evaluate(board, h.hole[p]), p) for p in alive]
        scores.sort()
        best = scores[0][0]
        winners = [p for sc, p in scores if sc == best]
        out = {p: 0.0 for p in alive}
        for w in winners:
            out[w] = 1.0 / len(winners)
        return out

    used = set(h.revealed)
    for p in alive:
        used.update(h.hole[p])

    full_deck = [Card.new(r + s) for r in "23456789TJQKA" for s in "shdc"]
    remain = [c for c in full_deck if c not in used]
    need = 5 - k

    wins = {p: 0.0 for p in alive}
    for _ in range(iters):
        random.shuffle(remain)
        runout = h.revealed + remain[:need]

        scores = [(EVAL.evaluate(runout, h.hole[p]), p) for p in alive]
        scores.sort()
        best = scores[0][0]
        winners = [p for sc, p in scores if sc == best]
        share = 1.0 / len(winners)
        for w in winners:
            wins[w] += share

    for p in wins:
        wins[p] /= iters
    return wins


def compute_outs_text(h: HandState, hero: str) -> str:
    alive = [p.name for p in h.players if p.name not in h.folded]
    if hero not in alive:
        return "Outs: (you folded / not in hand)."

    k = len(h.revealed)
    if h.street in ("river", "showdown") or k >= 5:
        return "Outs: none (no cards left to come)."
    if k == 0:
        return "Outs: (available after the flop)."

    used = set(h.revealed + h.hole[hero])
    for p in alive:
        used.update(h.hole[p])

    full_deck = [Card.new(r + s) for r in "23456789TJQKA" for s in "shdc"]
    remain = [c for c in full_deck if c not in used]

    def hero_not_lose(board5: List[int]) -> bool:
        hero_score = EVAL.evaluate(board5, h.hole[hero])
        for p in alive:
            if p == hero:
                continue
            sc = EVAL.evaluate(board5, h.hole[p])
            if sc < hero_score:
                return False
        return True

    if k == 4:
        outs = []
        base4 = list(h.revealed)
        for river in remain:
            if hero_not_lose(base4 + [river]):
                outs.append(river)
        if not outs:
            return "Outs (river): 0"
        faces = " ".join(unicode_card(c) for c in outs[:28])
        more = f" (+{len(outs)-28} more)" if len(outs) > 28 else ""
        return f"Outs (river): {len(outs)}\n{faces}{more}"

    if k == 3:
        outs_turn = []
        flop3 = list(h.revealed)
        for turn in remain:
            remain2 = [c for c in remain if c != turn]
            found = False
            for river in remain2:
                if hero_not_lose(flop3 + [turn, river]):
                    found = True
                    break
            if found:
                outs_turn.append(turn)
        if not outs_turn:
            return "Outs (turn): 0"
        faces = " ".join(unicode_card(c) for c in outs_turn[:28])
        more = f" (+{len(outs_turn)-28} more)" if len(outs_turn) > 28 else ""
        return f"Outs (turn): {len(outs_turn)}\n{faces}{more}"

    return "Outs: (not available at this stage)."


def hand_combo_label(h: HandState, name: str) -> str:
    """Return a human-readable hand class for a player given the final 5-card board."""
    board = h.board5[:5]
    score = EVAL.evaluate(board, h.hole[name])
    rc = EVAL.get_rank_class(score)
    return EVAL.class_to_string(rc)


# --------------------------
# Rendering
# --------------------------

def _rich_card_panel(card_int: int, hidden: bool) -> Panel:
    if hidden:
        t = Text("🂠", style="bold white on blue")
        return Panel(t, padding=(1, 3), border_style="blue")

    r, s = treys_to_rs(card_int)
    sym = SUIT_SYMBOL[s]
    red = is_red_suit(s)
    style = "bold red" if red else "bold white"
    inner = Text()
    inner.append(f"{r}{sym}\n", style=style)
    inner.append(card_label(card_int), style="dim")
    return Panel(inner, padding=(0, 1), border_style=("red" if red else "white"))


def render_for_viewer(h: HandState, t: TableState, viewer: Optional[str]):
    reveal_ai = t.reveal_once
    overlay = t.overlay_text if t.overlay_once else None

    if RICH_OK:
        console = Console()
        console.clear()

        header = Table.grid(expand=True)
        header.add_column(justify="left")
        header.add_column(justify="right")
        header.add_row(
            f"[bold]Texas Hold'em[/bold]  Street: [bold]{h.street.upper()}[/bold]",
            f"Pot: [bold]{h.pot}[/bold]   Current bet: [bold]{h.current_bet}[/bold]",
        )
        console.print(header)
        if viewer:
            console.print(f"[dim]Viewer:[/dim] [bold]{viewer}[/bold]")
        console.print()

        board_panels = []
        for i in range(5):
            if i < len(h.revealed):
                board_panels.append(_rich_card_panel(h.revealed[i], hidden=False))
            else:
                board_panels.append(Panel(Text(""), height=5, border_style="dim"))
        console.print("[bold]Board[/bold]")
        console.print(Columns(board_panels, equal=True, expand=True))
        console.print()

        console.print("[bold]Players[/bold]")
        is_showdown = (h.street == "showdown") or h.hand_done
        for p in h.players:
            name = p.name
            folded = name in h.folded
            show_cards = is_showdown or (p.is_human and viewer == name) or ((not p.is_human) and reveal_ai)

            tag = ""
            if not p.is_human and p.ai_type_key:
                tag = f" [{AI_TYPES[p.ai_type_key].name}]"
            if folded:
                tag += " (FOLDED)"
            if name in h.allin and not folded:
                tag += " (ALL-IN)"

            if is_showdown:
                tag += f" ({hand_combo_label(h, name)})"

            row = Table.grid(expand=True)
            row.add_column(width=40)
            row.add_column()
            row.add_column(justify="right", width=16)
            row.add_row(f"[bold]{name}{tag}[/bold]", "", f"Stack: [bold]{h.stacks[name]}[/bold]")
            console.print(row)

            c1, c2 = h.hole[name]
            console.print(Columns([
                _rich_card_panel(c1, hidden=not show_cards),
                _rich_card_panel(c2, hidden=not show_cards),
            ], equal=True, expand=False))
            console.print()

        if overlay:
            console.print(Panel(overlay, border_style="yellow"))
        if h.winner_text:
            console.print(Panel(f"[bold green]{h.winner_text}[/bold green]", border_style="green"))

        console.print()
        console.print("[bold]Hand log[/bold]")
        for line in h.log:
            console.print(f" • {line}", style="dim")

    else:
        clear_screen()
        print(f"Texas Hold'em  Street: {h.street.upper()}  Pot: {h.pot}  Current bet: {h.current_bet}")
        if viewer:
            print(f"Viewer: {viewer}")
        print()

        board_slots = []
        for i in range(5):
            board_slots.append(unicode_card(h.revealed[i]) if i < len(h.revealed) else "??")
        print("Board:", "  ".join(board_slots))
        print()

        is_showdown = (h.street == "showdown") or h.hand_done
        for p in h.players:
            name = p.name
            folded = name in h.folded
            show_cards = is_showdown or (p.is_human and viewer == name) or ((not p.is_human) and reveal_ai)

            cards = " ".join(unicode_card(c) for c in h.hole[name]) if show_cards else "🂠 🂠"
            tag = ""
            if not p.is_human and p.ai_type_key:
                tag = f" [{AI_TYPES[p.ai_type_key].name}]"
            if folded:
                tag += " (FOLDED)"
            if name in h.allin and not folded:
                tag += " (ALL-IN)"

            if is_showdown:
                tag += f" ({hand_combo_label(h, name)})"

            print(f"{name}{tag} | Stack: {h.stacks[name]:4d} | {cards}")
        print()

        if overlay:
            print(overlay)
            print()

        if h.winner_text:
            print(">>>", h.winner_text)
            print()

        print("Hand log:")
        for line in h.log:
            print(" -", line)

    if t.reveal_once:
        t.reveal_once = False
    if t.overlay_once:
        t.overlay_once = False
        t.overlay_text = None


# --------------------------
# Cheats / commands
# --------------------------

def cheat_help_text() -> str:
    return (
        "Commands (any prompt):\n"
        "  reveal\n"
        "  add <amount>\n"
        "  show out\n"
        "  explain type [<type>]\n"
        "  <AI name> - <AI type>\n"
        "  new player <amount>\n"
        "  new player <name> <amount>\n"
        "  help\n"
    )


def explain_types(t: TableState, specific: Optional[str] = None) -> str:
    counts: Dict[str, int] = {}
    for p in t.players:
        if not p.is_human and p.ai_type_key:
            nm = AI_TYPES[p.ai_type_key].name
            counts[nm] = counts.get(nm, 0) + 1

    if specific:
        key = normalize_type_name(specific)
        if not key:
            return f"Unknown type '{specific}'. Use: explain type"
        typ = AI_TYPES[key]
        return (
            f"AI Type: {typ.name}\n"
            f"  tightness={typ.tightness:.2f} aggression={typ.aggression:.2f} bluff={typ.bluff:.2f} risk={typ.risk:.2f}\n"
            f"  3-bet tendency={typ.threebet:.2f}  4-bet tendency={typ.fourbet:.2f}\n"
            f"  {typ.description}"
        )

    lines = [f"AI types available: {len(AI_TYPES)}", "Counts at table:"]
    if counts:
        for k in sorted(counts.keys()):
            lines.append(f"  {k}: {counts[k]}")
    else:
        lines.append("  (no AIs at table)")

    lines.append("\nType descriptions:")
    for key in sorted(AI_TYPES.keys(), key=lambda x: AI_TYPES[x].name):
        typ = AI_TYPES[key]
        lines.append(f"  {typ.name}: {typ.description}")
    return "\n".join(lines)


def set_ai_type(t: TableState, ai_name: str, type_name: str) -> str:
    key = normalize_type_name(type_name)
    if not key:
        return f"Unknown AI type '{type_name}'. Try: explain type"
    for p in t.players:
        if p.name == ai_name and not p.is_human:
            p.ai_type_key = key
            return f"Set {ai_name} to type {AI_TYPES[key].name}."
    return f"AI player '{ai_name}' not found at the table."


def handle_command(line: str, h: Optional[HandState], t: TableState) -> Tuple[bool, Optional[str]]:
    s = line.strip()
    if not s:
        return False, None
    low = s.lower()

    if low == "help":
        return True, cheat_help_text()

    if low == "restart":
        t.request_restart = 'same'
        return True, "Restart requested (same players & types)."

    if low in ("restart type", "restart types"):
        t.request_restart = 'shuffle'
        return True, "Restart requested (random AI types)."

    if low == "reveal":
        t.reveal_once = True
        return True, "Revealing AI hole cards (once)."

    if low.startswith("add "):
        parts = low.split()
        if len(parts) != 2 or not parts[1].lstrip("-").isdigit():
            return True, "Usage: add <amount>"
        amt = int(parts[1])
        t.stacks[t.hero_name] = t.stacks.get(t.hero_name, 0) + amt
        if h is not None:
            h.stacks[t.hero_name] = h.stacks.get(t.hero_name, 0) + amt
            h.log.append(f"[CHEAT] {t.hero_name} adds {amt}.")
        return True, f"[CHEAT] Added {amt} to {t.hero_name}."

    if low in ("show out", "show outs"):
        if h is None:
            return True, "No active hand."
        t.reveal_once = True
        t.overlay_once = True
        outs = compute_outs_text(h, t.hero_name)
        rates = compute_win_rates_true(h, iters=700)
        alive = [p.name for p in h.players if p.name not in h.folded]
        lines = ["Win rates (active players):"]
        for name in alive:
            lines.append(f"  {name:14s}: {100.0 * rates.get(name, 0.0):5.1f}%")
        t.overlay_text = outs + "\n\n" + "\n".join(lines)
        return True, "Showing outs + win rates (once)."

    if low.startswith("explain type"):
        parts = s.split(maxsplit=2)
        msg = explain_types(t, specific=(parts[2] if len(parts) == 3 else None))
        t.overlay_once = True
        t.overlay_text = msg
        return True, "Explaining AI types (once)."

    # Replace AI type: "<AI name> - <AI type>"
    if " - " in s:
        left, right = s.split(" - ", 1)
        left = left.strip()
        right = right.strip()
        if left and right:
            msg = set_ai_type(t, left, right)
            t.overlay_once = True
            t.overlay_text = msg
            return True, msg

    if low.startswith("new player"):
        parts = s.split()
        if len(parts) < 3:
            return True, "Usage: new player <amount>  OR  new player <name> <amount>"
        if not parts[-1].isdigit():
            return True, "Usage: new player <amount>  OR  new player <name> <amount>"
        amt = int(parts[-1])
        if len(parts) == 3:
            name = f"AI {t.next_ai_id}"
            t.next_ai_id += 1
        else:
            name = " ".join(parts[2:-1]).strip() or f"AI {t.next_ai_id}"
            t.next_ai_id += 1

        if name in t.stacks and t.stacks[name] > 0:
            return True, f"Player '{name}' already exists with stack {t.stacks[name]}."

        type_key = random.choice(list(AI_TYPES.keys()))
        t.players.append(Player(name=name, is_human=False, ai_type_key=type_key))
        t.stacks[name] = amt
        return True, f"Added '{name}' ({AI_TYPES[type_key].name}) with stack {amt}. Joins next hand."

    return False, None


def input_with_commands(prompt: str, h: Optional[HandState], t: TableState, viewer: Optional[str]) -> str:
    while True:
        s = input(prompt)
        consumed, msg = handle_command(s, h, t)
        if consumed:
            if msg:
                print(msg)
            if h is not None:
                render_for_viewer(h, t, viewer=viewer)
            continue
        return s.strip()

def betting_closed_due_to_allin(h: HandState) -> bool:
    """True if at most one non-folded player still has chips behind.
    When everyone else is all-in, further betting/raising is not allowed.
    """
    live = [p.name for p in h.players if p.name not in h.folded]
    with_chips = [nm for nm in live if h.stacks.get(nm, 0) > 0]
    return len(with_chips) <= 1



# --------------------------
# Betting engine
# --------------------------

def first_to_act_index(h: HandState) -> int:
    n = len(h.players)
    sb_idx = next_idx(h.dealer_idx, n)
    bb_idx = next_idx(sb_idx, n)
    if h.street == "preflop":
        return next_idx(bb_idx, n)  # UTG
    return sb_idx  # postflop


def can_fold_preflop_unopened(h: HandState) -> bool:
    return not (h.street == "preflop" and h.raise_count == 0 and h.current_bet == h.bb)


def min_raise_increment(h: HandState) -> int:
    """Minimum raise increment rule: at least 1/2 of the SMALL BLIND (rounded up).
    Example: SB=5 -> min increment=3.
    """
    return max(1, (h.sb + 1) // 2)


def min_raise_to(h: HandState) -> int:
    """Minimum new total bet level for a raise.
    Rule requested: if you raise, you must increase the bet by at least 1/2 SB.
    (We keep it simple and do NOT enforce the standard 'previous raise size' rule.)
    """
    return h.current_bet + min_raise_increment(h)


def pay_into_pot(h: HandState, name: str, amount: int) -> int:
    amt = min(amount, h.stacks[name])
    h.stacks[name] -= amt
    h.pot += amt
    h.contrib[name] += amt
    h.total_contrib[name] = h.total_contrib.get(name, 0) + amt
    if h.stacks[name] == 0:
        h.allin.add(name)
    return amt


def human_decide(h: HandState, t: TableState, name: str, to_call: int, pot_odds: float, can_raise: bool) -> Tuple[str, int]:
    while True:
        if to_call == 0:
            if not can_raise:
                s = input_with_commands("Choose (c=check) [betting closed: others all-in]: ", h, t, viewer=name).lower()
            else:
                s = input_with_commands("Choose (c=check, b=bet, a=all-in): ", h, t, viewer=name).lower()
            if s in ("a", "all in", "allin"):
                if not can_raise:
                    print("Betting is closed (everyone else is all-in). You can only check.")
                    continue
                max_total = h.contrib[name] + h.stacks[name]
                return ("allin", max_total)
            if s == "c":
                return ("call", 0)
            if s == "b":
                if not can_raise:
                    print("Betting is closed (everyone else is all-in). You can only check.")
                    continue
                default_to = max(min_raise_increment(h), h.bb, int(0.5 * max(1, h.pot)))
                amt_str = input_with_commands(
                    f"Bet to (total for street) [{default_to}] (or 'all in'): ",
                    h, t, viewer=name
                )
                if amt_str.strip().lower() in ("a", "all in", "allin"):
                    max_total = h.contrib[name] + h.stacks[name]
                    return ("allin", max_total)
                new_bet = default_to if not amt_str else (int(amt_str) if amt_str.isdigit() else -1)
                if new_bet <= 0:
                    print("Enter a positive number.")
                    continue
                min_to = max(min_raise_increment(h), h.bb)
                if new_bet < min_to and h.stacks[name] > 0:
                    print(f"Bet too small. Minimum is {min_to}.")
                    continue
                return ("raise_to", new_bet)
            print("Invalid. Use c or b.")
            continue

        no_fold_allowed = (h.street == "preflop" and h.raise_count == 0 and h.current_bet == h.bb)
        if no_fold_allowed:
            s = input_with_commands("Choose (c=call, r=raise, a=all-in) [no fold unopened]: ", h, t, viewer=name).lower()
        else:
            s = input_with_commands("Choose (f=fold, c=call, r=raise, a=all-in): ", h, t, viewer=name).lower()

        if s == "f":
            if no_fold_allowed:
                print("No fold allowed before the first raise preflop.")
                continue
            return ("fold", 0)
        if s in ("a", "all in", "allin"):
            max_total = h.contrib[name] + h.stacks[name]
            return ("allin", max_total)
        if s == "c":
            return ("call", 0)
        if s == "r":
            if not can_raise:
                print("Raising is not allowed (everyone else is all-in). You can only call or go all-in as a call.")
                continue

            if h.street == "preflop":
                default_to = 3 * h.bb if h.raise_count == 0 else (9 * h.bb if h.raise_count == 1 else 22 * h.bb)
            else:
                default_to = h.current_bet + max(h.bb, int(0.6 * max(1, h.pot)))
            amt_str = input_with_commands(f"Raise to (total for street) [{default_to}] (or 'all in'): ", h, t, viewer=name)
            if amt_str.lower() in ("a", "all in", "allin"):
                max_total = h.contrib[name] + h.stacks[name]
                return ("allin", max_total)
            new_bet = default_to if not amt_str else (int(amt_str) if amt_str.isdigit() else -1)
            if new_bet <= 0:
                print("Enter a positive number.")
                continue
            min_to = (min_raise_to(h) if h.current_bet > 0 else max(min_raise_increment(h), h.bb))
            if new_bet < min_to and h.stacks[name] > 0:
                print(f"Raise too small. Minimum is to {min_to}.")
                continue
            return ("raise_to", new_bet)

        print("Invalid input.")


def ai_decide(h: HandState, t: TableState, name: str, to_call: int, pot_odds: float, type_key: Optional[str], can_raise: bool) -> Tuple[str, int]:
    typ = AI_TYPES.get(type_key or "tag", AI_TYPES["tag"])
    if not can_raise:
        # Betting is closed when everyone else is all-in.
        return ("call", 0)
    wet = board_texture(h)

    alive_opps = [p.name for p in h.players if p.name not in h.folded and p.name != name]
    n_opp = max(1, len(alive_opps))

    iters = 120 if h.street == "preflop" else 160
    equity = estimate_equity_vs_random(h, name, iters=iters)

    opp_penalty = min(0.22, 0.06 * (n_opp - 1))
    style_buffer = 0.06 * (typ.tightness - 0.5)
    risk_buffer = -0.06 * (typ.risk - 0.5)
    required = max(0.05, pot_odds + opp_penalty + style_buffer + risk_buffer)

    no_fold_unopened = (h.street == "preflop" and h.raise_count == 0 and h.current_bet == h.bb)
    can_fold = (to_call > 0) and (not no_fold_unopened)

    multiway_damp = max(0.18, 1.0 - 0.17 * (n_opp - 1))
    bluff_chance = max(0.0, min(0.9, typ.bluff * (0.6 + 0.4 * wet) * multiway_damp))
    base_raise = typ.aggression * (0.55 + 0.25 * wet)

    def clamp_to_stack(target_total: int) -> int:
        max_total = h.contrib[name] + h.stacks[name]
        return max(0, min(target_total, max_total))

    def pick_open_size() -> int:
        return int((3.0 + 1.0 * typ.aggression) * h.bb)

    def pick_3bet_size() -> int:
        return int((8.0 + 3.0 * typ.aggression) * h.bb)

    def pick_4bet_size() -> int:
        return int((18.0 + 8.0 * typ.aggression) * h.bb)

    def pick_postflop_bet() -> int:
        if typ.name.lower() == "maniac" and random.random() < 0.25:
            frac = 1.25
        else:
            frac = random.choice([0.33, 0.55, 0.75])
        return max(h.bb, min_raise_increment(h), int(frac * max(1, h.pot)))

    if h.street == "preflop":
        if h.raise_count == 0:
            open_thresh = 0.58 - 0.12 * typ.aggression - 0.04 * (n_opp - 1)
            if equity >= open_thresh or random.random() < (0.15 + 0.35 * typ.aggression):
                new_total = clamp_to_stack(pick_open_size())
                if new_total > h.current_bet and h.stacks[name] > 0:
                    return ("raise_to", new_total)
            return ("call", 0)

        if h.raise_count == 1:
            want_3bet = (equity >= required + 0.18) or (random.random() < typ.threebet * bluff_chance)
            if want_3bet and h.stacks[name] > 0:
                new_total = clamp_to_stack(pick_3bet_size())
                if new_total > h.current_bet:
                    return ("raise_to", new_total)

        if h.raise_count >= 2:
            want_4bet = (equity >= required + 0.22) or (random.random() < typ.fourbet * bluff_chance * 0.7)
            if want_4bet and h.stacks[name] > 0:
                new_total = clamp_to_stack(pick_4bet_size())
                if new_total > h.current_bet:
                    return ("raise_to", new_total)

        if equity >= required or not can_fold:
            return ("call", 0)
        return ("fold", 0)

    if to_call == 0:
        value_thresh = 0.56 - 0.05 * (n_opp - 1) - 0.04 * typ.tightness
        do_value = equity >= value_thresh
        do_bluff = (not do_value) and (random.random() < bluff_chance * (0.5 + 0.6 * typ.aggression))
        if do_value or do_bluff:
            bet_size = pick_postflop_bet()
            new_total = clamp_to_stack(bet_size)
            if new_total > 0:
                return ("raise_to", new_total)
        return ("call", 0)

    raise_thresh = required + 0.16 - 0.07 * typ.aggression - 0.05 * wet
    want_raise = (equity >= raise_thresh and random.random() < base_raise) or (random.random() < bluff_chance * typ.aggression * 0.5)

    if want_raise and h.stacks[name] > 0:
        inc = max(h.bb, int(0.6 * max(1, h.pot)))
        new_total = clamp_to_stack(h.current_bet + inc)
        if new_total > h.current_bet:
            return ("raise_to", new_total)

    if equity >= required or not can_fold:
        return ("call", 0)
    return ("fold", 0)


def betting_round(h: HandState, t: TableState):
    if h.hand_done:
        return

    if len([p for p in h.players if p.name not in h.folded]) <= 1:
        return

    n = len(h.players)
    idx = first_to_act_index(h)

    def advance(i: int) -> int:
        j = i
        for _ in range(n):
            nm = h.players[j].name
            if nm not in h.folded and nm not in h.allin:
                return j
            j = (j + 1) % n
        return i

    idx = advance(idx)
    if not active_nonfolded(h):
        return

    pending: Set[str] = set(active_nonfolded(h))
    raise_rights: Set[str] = set(active_nonfolded(h))  # who still has the right to raise this betting round
    loop_guard = 0

    while pending:
        # Defensive cleanup: players may have become all-in/folded earlier.
        pending -= (h.folded | h.allin)
        if not pending:
            break

        loop_guard += 1
        if loop_guard > 600:
            h.log.append("Loop guard tripped (betting round).")
            break

        p = h.players[idx]
        name = p.name

        if name in h.folded or name in h.allin:
            pending.discard(name)
            idx = advance((idx + 1) % n)
            continue
        if name not in pending:
            idx = advance((idx + 1) % n)
            continue

        to_call = max(0, h.current_bet - h.contrib.get(name, 0))
        pot_odds = (to_call / (h.pot + to_call)) if (h.pot + to_call) > 0 else 0.0

        if p.is_human:
            input(f"\nPass to {name}. Press Enter when ready...")
            render_for_viewer(h, t, viewer=name)
            action = human_decide(h, t, name, to_call, pot_odds, can_raise=((name in raise_rights) and (not betting_closed_due_to_allin(h))))
        else:
            render_for_viewer(h, t, viewer=None)
            action = ai_decide(h, t, name, to_call, pot_odds, p.ai_type_key, can_raise=((name in raise_rights) and (not betting_closed_due_to_allin(h))))

        if action[0] == "fold":
            if not can_fold_preflop_unopened(h):
                paid = pay_into_pot(h, name, to_call)
                h.log.append(f"{name} calls {paid} (forced; no-fold-unopened).")
            else:
                h.folded.add(name)
                h.log.append(f"{name} folds.")
            pending.discard(name)
            raise_rights.discard(name)

        elif action[0] == "call":
            paid = pay_into_pot(h, name, to_call)
            h.log.append(f"{name} checks." if (to_call == 0 and h.current_bet == 0) else (f"{name} checks (already matched)." if to_call == 0 else f"{name} calls {paid}."))
            pending.discard(name)
            raise_rights.discard(name)

        elif action[0] == "allin":
            max_total = action[1]
            # If betting is closed, all-in can only function as a call.
            if betting_closed_due_to_allin(h) and max_total > h.current_bet:
                to_call2 = max(0, h.current_bet - h.contrib.get(name, 0))
                paid = pay_into_pot(h, name, to_call2)
                if to_call2 > 0 and paid < to_call2:
                    h.log.append(f"{name} calls {paid} (all-in short).")
                else:
                    h.log.append(f"{name} calls {paid}." if to_call2 > 0 else f"{name} checks.")
                pending.discard(name)
            else:
                need = max(0, max_total - h.contrib.get(name, 0))
                paid = pay_into_pot(h, name, need)
                new_total = h.contrib[name]
                if new_total > h.current_bet and not betting_closed_due_to_allin(h):
                    inc = new_total - h.current_bet
                    # An all-in that increases the bet level MUST be matched by others to continue.
                    # If the increase is smaller than the minimum raise increment, it is an 'incomplete raise'
                    # and does NOT reopen raising rights for players who already acted (full bet rule).
                    full_raise = inc >= min_raise_increment(h)
                    h.last_raise_size = max(1, inc)
                    h.current_bet = new_total
                    if full_raise:
                        h.raise_count += 1
                        h.log.append(f"{name} goes all-in to {h.current_bet}.")
                        # Full raise: everyone still live regains raise rights
                        raise_rights = set(active_nonfolded(h))
                    else:
                        h.log.append(f"{name} goes all-in to {h.current_bet} (incomplete raise).")
                    # Others must respond to the new bet level
                    pending = set(active_nonfolded(h))
                    pending.discard(name)
                    # Raiser has acted; remove raise right for now
                    raise_rights.discard(name)
                else:
                    # all-in call / bet not exceeding current bet
                    if (h.current_bet - h.contrib.get(name, 0)) > 0 and paid < (h.current_bet - h.contrib.get(name, 0)):
                        h.log.append(f"{name} calls {paid} (all-in short).")
                    else:
                        h.log.append(f"{name} goes all-in for {paid}.")
                    pending.discard(name)
                    raise_rights.discard(name)

        elif action[0] == "raise_to":
            if betting_closed_due_to_allin(h):
                paid = pay_into_pot(h, name, to_call)
                if to_call > 0 and paid < to_call:
                    h.log.append(f"{name} calls {paid} (all-in short).")
                else:
                    h.log.append(f"{name} calls {paid}." if to_call > 0 else f"{name} checks.")
                pending.discard(name)
                idx = advance((idx + 1) % n)
                continue
            new_bet = action[1]
            need = max(0, new_bet - h.contrib[name])
            if need <= 0:
                paid = pay_into_pot(h, name, to_call)
                h.log.append(f"{name} calls {paid}.")
                pending.discard(name)
            else:
                min_to = min_raise_to(h) if h.current_bet > 0 else max(min_raise_increment(h), h.bb)
                if new_bet < min_to and h.stacks[name] > 0:
                    new_bet = min_to
                    need = max(0, new_bet - h.contrib[name])

                paid = pay_into_pot(h, name, need)
                last = new_bet - h.current_bet
                h.last_raise_size = max(1, last)
                h.current_bet = max(h.current_bet, h.contrib[name])
                h.raise_count += 1
                h.log.append(f"{name} raises to {h.current_bet}.")

                pending = set(active_nonfolded(h))
                pending.discard(name)
        # Render after each action so the full log is visible and human hole cards are hidden.
        render_for_viewer(h, t, viewer=None)
        if t.request_restart:
            h.log.append(f"[SYSTEM] Restart requested ({t.request_restart}).")
            h.hand_done = True
            return

        alive = [pp.name for pp in h.players if pp.name not in h.folded]
        if len(alive) == 1:
            w = alive[0]
            h.stacks[w] += h.pot
            msg = f"{w} player win because everyone folds"
            h.winner_text = msg
            h.log.append(msg)
            h.pot = 0
            h.hand_done = True
            return

        idx = advance((idx + 1) % n)


# --------------------------
# Hand loop + table loop
# --------------------------

def restart_table(t: TableState, mode: str = 'same'):
    """Restart the whole game with the same players.
    mode='same' keeps AI types; mode='shuffle' re-rolls AI types randomly.
    Resets stacks to starting_stack for everyone.
    """
    rebuilt: List[Player] = []
    stacks: Dict[str, int] = {}
    for (name, is_human, ai_type_key) in t.initial_specs:
        if is_human:
            rebuilt.append(Player(name=name, is_human=True, ai_type_key=None))
        else:
            if mode == 'shuffle':
                ai_type_key = random.choice(list(AI_TYPES.keys()))
            rebuilt.append(Player(name=name, is_human=False, ai_type_key=ai_type_key))
        stacks[name] = t.starting_stack
    t.players = rebuilt
    t.stacks = stacks
    t.request_restart = None
    t.reveal_once = False
    t.overlay_once = False
    t.overlay_text = None


def cleanup_busted(t: TableState):
    new_players = []
    for p in t.players:
        if not p.is_human and t.stacks.get(p.name, 0) <= 0:
            continue
        new_players.append(p)
    t.players = new_players

    existing = {p.name for p in t.players}
    for name in list(t.stacks.keys()):
        if name not in existing:
            del t.stacks[name]


def sync_hand_to_table(h: HandState, t: TableState):
    for p in t.players:
        t.stacks[p.name] = h.stacks.get(p.name, 0)
    cleanup_busted(t)


def run_hand(h: HandState, t: TableState):
    render_for_viewer(h, t, viewer=None)
    betting_round(h, t)
    if t.request_restart:
        h.hand_done = True
        return
    if h.hand_done:
        return

    while not h.hand_done and h.street != "showdown":
        reveal_next_street(h)
        render_for_viewer(h, t, viewer=None)
        betting_round(h, t)
        if t.request_restart:
            h.hand_done = True
            return
        if h.hand_done:
            return
        if h.street == "river":
            reveal_next_street(h)

    resolve_showdown(h)
    t.reveal_once = True
    render_for_viewer(h, t, viewer=None)


def read_int(prompt: str, default: int, lo: int, hi: int) -> int:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        x = default if not s else (int(s) if s.isdigit() else None)
        if x is not None and lo <= x <= hi:
            return x
        print(f"Enter an integer between {lo} and {hi}.")


def setup_table() -> Tuple[TableState, int]:
    print("Texas Hold'em (CLI) — Multi-round table")
    if not RICH_OK:
        print("Tip: pip install rich  (prettier UI)")
    print("Type 'help' at any prompt for commands.\n")

    hero = input("Your (Hero) name [You]: ").strip() or "You"
    n_humans = read_int("Number of human players (including you)", default=1, lo=1, hi=8)
    human_names = [hero]
    for i in range(2, n_humans + 1):
        nm = input(f"Name for human player #{i}: ").strip() or f"Human{i}"
        if nm in human_names:
            nm = f"{nm}_{i}"
        human_names.append(nm)

    n_ai = read_int("Number of AI opponents (0-7)", default=max(0, 6 - (n_humans - 1)), lo=0, hi=8 - n_humans)
    starting_stack = read_int("Starting stack", default=500, lo=50, hi=1_000_000)
    sb = read_int("Small blind", default=5, lo=1, hi=100_000)

    players: List[Player] = []
    stacks: Dict[str, int] = {}

    for nm in human_names:
        players.append(Player(name=nm, is_human=True))
        stacks[nm] = starting_stack

    next_ai_id = 1
    for _ in range(n_ai):
        nm = f"AI {next_ai_id}"
        next_ai_id += 1
        key = random.choice(list(AI_TYPES.keys()))
        players.append(Player(name=nm, is_human=False, ai_type_key=key))
        stacks[nm] = starting_stack

    initial_specs = [(p.name, p.is_human, p.ai_type_key) for p in players]
    t = TableState(players=players, stacks=stacks, hero_name=hero, next_ai_id=next_ai_id,
                   starting_stack=starting_stack, sb=sb, initial_specs=initial_specs)
    return t, sb


def main():
    t, sb = setup_table()
    dealer = 0
    hand_no = 1

    while True:
        cleanup_busted(t)
        if len(t.players) < 2:
            clear_screen()
            print("Game over: not enough players to continue.")
            print("Final stacks:")
            for p in t.players:
                print(f"  {p.name}: {t.stacks.get(p.name, 0)}")
            break

        h = deal_new_hand(t, dealer_idx=dealer, sb=sb)
        h.log.append(f"Hand #{hand_no} (dealer: {t.players[dealer].name})")
        hand_no += 1

        run_hand(h, t)
        if t.request_restart:
            mode = t.request_restart
            restart_table(t, mode=mode)
            dealer = 0
            hand_no = 1
            continue
        sync_hand_to_table(h, t)

        dealer = rotate_dealer(dealer, len(t.players))

        s = input_with_commands("\nNext hand: Enter=deal, or type 'q' to quit: ", None, t, viewer=t.hero_name).lower()
        if s in ("q", "quit", "exit"):
            print("Good game!")
            break


if __name__ == "__main__":
    main()
