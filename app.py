import random
import streamlit as st
from treys import Card, Evaluator, Deck

# ----------------------------
# Card face (SVG) rendering
# ----------------------------
SUIT_SYMBOL = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}
SUIT_NAME   = {"s": "Spades", "h": "Hearts", "d": "Diamonds", "c": "Clubs"}
RANK_NAME   = {"2":"2","3":"3","4":"4","5":"5","6":"6","7":"7","8":"8","9":"9",
               "T":"10","J":"Jack","Q":"Queen","K":"King","A":"Ace"}

def treys_to_rs(card_int: int):
    s = Card.int_to_str(card_int)   # e.g. "Kh"
    rank, suit = s[0], s[1]
    return rank, suit

def card_label(card_int: int):
    r, s = treys_to_rs(card_int)
    return f"{RANK_NAME[r]} of {SUIT_NAME[s]}"

def card_svg(card_int: int, width=120, height=170):
    r, s = treys_to_rs(card_int)
    sym = SUIT_SYMBOL[s]
    is_red = s in ("h", "d")
    color = "#C1121F" if is_red else "#111111"

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 120 170">
      <rect x="2" y="2" width="116" height="166" rx="12" ry="12" fill="white" stroke="#222" stroke-width="2"/>
      <text x="12" y="24" font-size="20" font-family="Arial" fill="{color}">{r}</text>
      <text x="12" y="46" font-size="22" font-family="Arial" fill="{color}">{sym}</text>

      <text x="60" y="98" text-anchor="middle" font-size="54" font-family="Arial" fill="{color}">{sym}</text>

      <g transform="rotate(180,60,85)">
        <text x="12" y="24" font-size="20" font-family="Arial" fill="{color}">{r}</text>
        <text x="12" y="46" font-size="22" font-family="Arial" fill="{color}">{sym}</text>
      </g>
    </svg>
    """
    return svg.strip().encode("utf-8")

def card_back_svg(width=120, height=170):
    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 120 170">
      <rect x="2" y="2" width="116" height="166" rx="12" ry="12" fill="#0B3D91" stroke="#222" stroke-width="2"/>
      <rect x="10" y="10" width="100" height="150" rx="10" ry="10" fill="none" stroke="white" stroke-width="2"/>
      <text x="60" y="92" text-anchor="middle" font-size="18" font-family="Arial" fill="white">HOLD'EM</text>
    </svg>
    """
    return svg.strip().encode("utf-8")

# ----------------------------
# Game logic (simple betting)
# ----------------------------
evaluator = Evaluator()

STREETS = ["preflop", "flop", "turn", "river", "showdown"]

def init_state():
    if "g" not in st.session_state:
        st.session_state.g = None

def new_hand(hero_name: str, n_ai: int, starting_stack: int, small_blind: int):
    deck = Deck()
    players = [hero_name] + [f"AI {i}" for i in range(1, n_ai + 1)]

    hole = {}
    for p in players:
        hole[p] = [deck.draw(1)[0], deck.draw(1)[0]]

    # 5 community cards will be pre-generated; we reveal by street
    board5 = [deck.draw(1)[0] for _ in range(5)]

    stacks = {p: starting_stack for p in players}
    pot = 0
    big_blind = small_blind * 2

    # post blinds (hero SB, AI1 BB just for simplicity)
    sb_player = players[0]
    bb_player = players[1] if len(players) > 1 else players[0]
    sb_amt = min(small_blind, stacks[sb_player])
    bb_amt = min(big_blind, stacks[bb_player])
    stacks[sb_player] -= sb_amt
    stacks[bb_player] -= bb_amt
    pot += sb_amt + bb_amt

    return {
        "players": players,
        "hero": hero_name,
        "n_ai": n_ai,
        "deck": deck,      # kept for possible future use
        "hole": hole,
        "board5": board5,
        "street": "preflop",
        "revealed_board": [],
        "stacks": stacks,
        "pot": pot,
        "to_call": big_blind,     # simplified: one decision per street
        "last_bet": big_blind,
        "folded": set(),
        "log": [f"New hand. {sb_player} posts SB {sb_amt}. {bb_player} posts BB {bb_amt}."],
        "hand_done": False,
        "winner_text": None,
    }

def reveal_next_street(g):
    if g["street"] == "preflop":
        g["revealed_board"] = g["board5"][:3]
        g["street"] = "flop"
        g["to_call"] = 0
        g["last_bet"] = 0
        g["log"].append("Flop dealt.")
    elif g["street"] == "flop":
        g["revealed_board"] = g["board5"][:4]
        g["street"] = "turn"
        g["to_call"] = 0
        g["last_bet"] = 0
        g["log"].append("Turn dealt.")
    elif g["street"] == "turn":
        g["revealed_board"] = g["board5"][:5]
        g["street"] = "river"
        g["to_call"] = 0
        g["last_bet"] = 0
        g["log"].append("River dealt.")
    elif g["street"] == "river":
        g["street"] = "showdown"
        g["log"].append("Showdown.")
    return g

def active_players(g):
    return [p for p in g["players"] if p not in g["folded"] and g["stacks"][p] >= 0]

def estimate_equity_vs_random(g, player: str, iters=250):
    """
    AI equity estimate vs (N-1) random opponents given current board.
    AI does NOT see hero cards; opponents are random.
    """
    known_board = list(g["revealed_board"])
    my_hand = list(g["hole"][player])
    players_left = [p for p in g["players"] if p not in g["folded"]]
    n_opponents = max(1, len(players_left) - 1)

    # build remaining deck
    used = set(known_board + my_hand)
    for p in players_left:
        if p == player: 
            continue
        # opponents unknown -> do not mark their hole
        pass

    full_deck = [Card.new(r+s) for r in "23456789TJQKA" for s in "shdc"]
    remain = [c for c in full_deck if c not in used]

    wins = 0
    ties = 0
    for _ in range(iters):
        random.shuffle(remain)
        # draw opponents
        idx = 0
        opp_holes = []
        for _k in range(n_opponents):
            opp_holes.append([remain[idx], remain[idx+1]])
            idx += 2

        # complete board
        need = 5 - len(known_board)
        runout = known_board + remain[idx:idx+need]
        idx += need

        my_score = evaluator.evaluate(runout, my_hand)
        opp_scores = [evaluator.evaluate(runout, h) for h in opp_holes]
        best = min([my_score] + opp_scores)

        if my_score == best:
            if opp_scores.count(best) == 0:
                wins += 1
            else:
                ties += 1

    return (wins + 0.5 * ties) / iters

def ai_action(g, ai: str):
    if ai in g["folded"]:
        return
    if g["street"] == "showdown":
        return

    # simple policy based on estimated equity + whether facing a bet
    equity = estimate_equity_vs_random(g, ai, iters=180)
    to_call = g["to_call"]

    # thresholds you can tune
    if to_call > 0:
        if equity < 0.28:
            g["folded"].add(ai)
            g["log"].append(f"{ai} folds (equity≈{equity:.2f}).")
            return
        elif equity < 0.55:
            call_amt = min(to_call, g["stacks"][ai])
            g["stacks"][ai] -= call_amt
            g["pot"] += call_amt
            g["log"].append(f"{ai} calls {call_amt} (equity≈{equity:.2f}).")
            return
        else:
            # raise
            raise_amt = min(max(10, to_call * 2), g["stacks"][ai])
            g["stacks"][ai] -= raise_amt
            g["pot"] += raise_amt
            g["to_call"] = raise_amt  # simplified: sets new price for hero only
            g["last_bet"] = raise_amt
            g["log"].append(f"{ai} raises to {raise_amt} (equity≈{equity:.2f}).")
            return
    else:
        # no bet to call: check or bet
        if equity > 0.60 and g["stacks"][ai] > 0:
            bet = min(20, g["stacks"][ai])
            g["stacks"][ai] -= bet
            g["pot"] += bet
            g["to_call"] = bet
            g["last_bet"] = bet
            g["log"].append(f"{ai} bets {bet} (equity≈{equity:.2f}).")
        else:
            g["log"].append(f"{ai} checks (equity≈{equity:.2f}).")

def resolve_hand_if_needed(g):
    alive = [p for p in g["players"] if p not in g["folded"]]
    if len(alive) == 1:
        winner = alive[0]
        g["stacks"][winner] += g["pot"]
        g["winner_text"] = f"{winner} wins the pot ({g['pot']}) — everyone else folded."
        g["log"].append(g["winner_text"])
        g["pot"] = 0
        g["hand_done"] = True
        return

    if g["street"] == "showdown":
        board = g["board5"][:5]
        scores = []
        for p in alive:
            sc = evaluator.evaluate(board, g["hole"][p])
            scores.append((sc, p))
        scores.sort()
        best_score = scores[0][0]
        winners = [p for sc, p in scores if sc == best_score]

        if len(winners) == 1:
            w = winners[0]
            g["stacks"][w] += g["pot"]
            g["winner_text"] = f"{w} wins the pot ({g['pot']}) at showdown."
        else:
            split = g["pot"] // len(winners)
            for w in winners:
                g["stacks"][w] += split
            g["winner_text"] = f"Split pot: {', '.join(winners)} each get {split} (pot={g['pot']})."

        g["log"].append(g["winner_text"])
        g["pot"] = 0
        g["hand_done"] = True

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Hold'em: You vs AI", layout="wide")
init_state()

st.title("Texas Hold'em — You vs AI")
st.caption("One human player (you) vs computer opponents. Cards shown as faces (SVG).")

with st.sidebar:
    st.header("Game Setup")
    hero_name = st.text_input("Your name", value="You")
    n_ai = st.slider("Number of AI opponents", min_value=1, max_value=7, value=5)
    starting_stack = st.number_input("Starting stack", min_value=100, max_value=100000, value=500, step=50)
    sb = st.number_input("Small blind", min_value=1, max_value=1000, value=5, step=1)
    reveal_ai = st.toggle("Cheat: reveal AI hole cards", value=False)

    st.divider()
    if st.button("Start / New Hand", use_container_width=True):
        st.session_state.g = new_hand(hero_name, n_ai, starting_stack, sb)

g = st.session_state.g

if not g:
    st.info("Click **Start / New Hand** in the sidebar.")
    st.stop()

# Top status
left, mid, right = st.columns([1, 1, 1])
with left:
    st.metric("Street", g["street"].upper())
with mid:
    st.metric("Pot", g["pot"])
with right:
    st.metric("To Call", g["to_call"])

st.divider()

# Board
st.subheader("Board")
board_cols = st.columns(5)
for i in range(5):
    with board_cols[i]:
        if i < len(g["revealed_board"]):
            st.image(card_svg(g["revealed_board"][i]), use_container_width=True)
            st.caption(card_label(g["revealed_board"][i]))
        else:
            st.markdown("—")

st.divider()

# Your hand + action
st.subheader("Your Hand")
h1, h2, _ = st.columns([1, 1, 2])
with h1:
    st.image(card_svg(g["hole"][g["hero"]][0]), use_container_width=True)
    st.caption(card_label(g["hole"][g["hero"]][0]))
with h2:
    st.image(card_svg(g["hole"][g["hero"]][1]), use_container_width=True)
    st.caption(card_label(g["hole"][g["hero"]][1]))

st.write(f"**Your stack:** {g['stacks'][g['hero']]}")

# If hand ended, allow advancing to showdown display/new hand
if g["hand_done"]:
    st.success(g["winner_text"])
else:
    st.markdown("### Your Action (one decision per street)")
    a1, a2, a3 = st.columns(3)

    def hero_fold():
        g["folded"].add(g["hero"])
        g["log"].append(f"{g['hero']} folds.")
        resolve_hand_if_needed(g)

    def hero_call_check():
        to_call = g["to_call"]
        if to_call > 0:
            pay = min(to_call, g["stacks"][g["hero"]])
            g["stacks"][g["hero"]] -= pay
            g["pot"] += pay
            g["log"].append(f"{g['hero']} calls {pay}.")
        else:
            g["log"].append(f"{g['hero']} checks.")

    def hero_bet_raise(amount: int):
        amt = min(amount, g["stacks"][g["hero"]])
        if amt <= 0:
            return
        g["stacks"][g["hero"]] -= amt
        g["pot"] += amt
        g["to_call"] = amt
        g["last_bet"] = amt
        g["log"].append(f"{g['hero']} bets/raises to {amt}.")

    with a1:
        st.button("Fold", on_click=hero_fold, use_container_width=True)

    with a2:
        st.button("Check / Call", on_click=hero_call_check, use_container_width=True)

    with a3:
        raise_amt = st.number_input("Bet/Raise amount", min_value=1, max_value=100000, value=20, step=5)
        st.button("Bet / Raise", on_click=hero_bet_raise, args=(int(raise_amt),), use_container_width=True)

    # After hero acts, AIs act, then we either resolve or move street
    st.divider()
    if st.button("Run AI + Next Street", use_container_width=True):
        # AI action round
        for p in g["players"]:
            if p == g["hero"]:
                continue
            ai_action(g, p)

        resolve_hand_if_needed(g)
        if not g["hand_done"]:
            # If hero folded, might already be resolved; otherwise advance street
            if g["hero"] not in g["folded"]:
                g = reveal_next_street(g)
            resolve_hand_if_needed(g)

# Player table
st.divider()
st.subheader("Players")
cols = st.columns(4)
for i, p in enumerate(g["players"]):
    with cols[i % 4]:
        st.markdown(f"**{p}**{' (FOLDED)' if p in g['folded'] else ''}")
        st.write(f"Stack: {g['stacks'][p]}")
        c1, c2 = st.columns(2)

        show_cards = (p == g["hero"]) or reveal_ai or g["hand_done"]
        if show_cards:
            with c1:
                st.image(card_svg(g["hole"][p][0]), use_container_width=True)
            with c2:
                st.image(card_svg(g["hole"][p][1]), use_container_width=True)
        else:
            with c1:
                st.image(card_back_svg(), use_container_width=True)
            with c2:
                st.image(card_back_svg(), use_container_width=True)

st.divider()
st.subheader("Hand Log")
st.write("\n".join(f"- {x}" for x in g["log"]))
