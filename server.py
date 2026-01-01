#!/usr/bin/env python3
"""Texas Hold'em multiplayer (terminal) — Server

- Runs the game engine and hosts a TCP server.
- Human players connect using client.py. Each human sees only their own hole cards + board.

Protocol: newline-delimited JSON (one object per line).
"""

import argparse
import json
import queue
import socket
import threading
import time
from typing import Dict, Optional, List

import holdem_engine as eng


# -----------------------------
# Networking helpers
# -----------------------------

def send_json(sock: socket.socket, obj: dict):
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)

def recv_json_lines(sock: socket.socket, out_q: "queue.Queue[dict]", stop_evt: threading.Event):
    buf = b""
    try:
        while not stop_evt.is_set():
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line.decode("utf-8"))
                    out_q.put(obj)
                except Exception:
                    out_q.put({"type": "bad_json", "raw": line.decode("utf-8", errors="replace")})
    except Exception as e:
        out_q.put({"type": "net_error", "error": str(e)})


class Client:
    def __init__(self, sock: socket.socket, addr):
        self.sock = sock
        self.addr = addr
        self.inbox: "queue.Queue[dict]" = queue.Queue()
        self.stop_evt = threading.Event()
        self.thread = threading.Thread(target=recv_json_lines, args=(sock, self.inbox, self.stop_evt), daemon=True)
        self.thread.start()
        self.name: Optional[str] = None

    def send(self, obj: dict):
        send_json(self.sock, obj)

    def close(self):
        try:
            self.stop_evt.set()
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass


# -----------------------------
# Game rendering (per viewer)
# -----------------------------

def _card_short(c: int) -> str:
    try:
        return eng.card_label(c)
    except Exception:
        return eng.Card.int_to_str(c)

def build_state(h: "eng.HandState", t: "eng.TableState", viewer: str, prompt: str = "", needs_input: bool = False, info: str = "") -> dict:
    board = [_card_short(c) for c in h.revealed]
    your_hole = [_card_short(c) for c in h.hole.get(viewer, [])]

    players_view: List[dict] = []
    for p in t.players:
        nm = p.name
        if nm in h.folded:
            status = "FOLDED"
        elif nm in h.allin:
            status = "ALL-IN"
        else:
            status = "IN"

        # Hide other players' hole cards unless showdown
        show = (nm == viewer) or (h.street == "showdown")
        cards = [_card_short(c) for c in h.hole.get(nm, [])] if show else ["??", "??"]

        players_view.append({
            "name": nm,
            "stack": h.stacks.get(nm, 0),
            "status": status,
            "is_human": bool(p.is_human),
            "cards": cards,
        })

    return {
        "type": "state",
        "viewer": viewer,
        "street": h.street,
        "pot": h.pot,
        "current_bet": getattr(h, "current_bet", 0),
        "board": board,
        "your_hole": your_hole,
        "players": players_view,
        "recent_log": h.log[-40:],
        "winner_text": h.winner_text or "",
        "prompt": prompt,
        "needs_input": needs_input,
        "info": info,
    }


# -----------------------------
# Multiplayer input adapter
# -----------------------------

class MultiplayerIO:
    def __init__(self, clients: Dict[str, Client], allow_cheats: bool):
        self.clients = clients
        self.allow_cheats = allow_cheats

    def broadcast_state(self, h: "eng.HandState", t: "eng.TableState", acting: Optional[str] = None, prompt: str = "", needs_input: bool = False, info_by_name: Optional[Dict[str, str]] = None):
        for nm, c in self.clients.items():
            info = ""
            if info_by_name and nm in info_by_name:
                info = info_by_name[nm]
            st = build_state(
                h, t, viewer=nm,
                prompt=(prompt if nm == acting else ""),
                needs_input=(needs_input and nm == acting),
                info=info,
            )
            try:
                c.send(st)
            except Exception:
                pass

    def get_action(self, prompt: str, h: "eng.HandState", t: "eng.TableState", viewer: str) -> str:
        self.broadcast_state(h, t, acting=viewer, prompt=prompt, needs_input=True)
        client = self.clients[viewer]

        while True:
            msg = client.inbox.get()
            if not isinstance(msg, dict):
                continue
            if msg.get("type") == "action":
                text = str(msg.get("text", "")).strip()
                if not text:
                    self.broadcast_state(h, t, acting=viewer, prompt=prompt, needs_input=True,
                                         info_by_name={viewer: "Empty input. Try again."})
                    continue

                low = text.lower().strip()
                if low in ("quit", "exit"):
                    raise SystemExit(f"{viewer} quit")

                if not self.allow_cheats:
                    cheat_prefixes = ("reveal", "show out", "add ", "restart", "restart type", "explain type", "new player")
                    if low == "reveal" or low == "show out" or any(low.startswith(p) for p in cheat_prefixes):
                        self.broadcast_state(h, t, acting=viewer, prompt=prompt, needs_input=True,
                                             info_by_name={viewer: "Cheat codes are disabled in multiplayer mode."})
                        continue

                return text

            self.broadcast_state(h, t, acting=viewer, prompt=prompt, needs_input=True,
                                 info_by_name={viewer: "Unexpected message. Please send an action."})


def patch_engine_for_multiplayer(io: MultiplayerIO):
    eng.clear_screen = lambda: None
    eng.render_for_viewer = lambda *args, **kwargs: None

    def input_with_commands(prompt: str, h: Optional["eng.HandState"], t: "eng.TableState", viewer: Optional[str]) -> str:
        assert viewer is not None, "viewer must be acting player's name in multiplayer"
        while True:
            s = io.get_action(prompt, h, t, viewer=viewer)
            consumed, msg = eng.handle_command(s, h, t)
            if consumed:
                info = msg or "Command processed."
                if h is not None:
                    io.broadcast_state(h, t, acting=viewer, prompt=prompt, needs_input=True, info_by_name={viewer: info})
                continue
            return s.strip()

    eng.input_with_commands = input_with_commands


# -----------------------------
# Server main
# -----------------------------

def wait_for_humans(host: str, port: int, humans: int) -> Dict[str, Client]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(8)

    print(f"[SERVER] Listening on {host}:{port} ... waiting for {humans} human players to join.")
    joined: Dict[str, Client] = {}

    while len(joined) < humans:
        sock, addr = srv.accept()
        c = Client(sock, addr)
        print(f"[SERVER] Connection from {addr}")

        join_deadline = time.time() + 30
        name = None
        while time.time() < join_deadline and name is None:
            try:
                msg = c.inbox.get(timeout=0.5)
            except queue.Empty:
                continue
            if msg.get("type") == "join":
                name = str(msg.get("name", "")).strip()
                if not name:
                    c.send({"type": "error", "message": "Name required."})
                    name = None
                elif name in joined:
                    c.send({"type": "error", "message": f"Name '{name}' already taken."})
                    name = None
                else:
                    c.name = name
                    joined[name] = c
                    c.send({"type": "welcome", "name": name, "message": f"Joined as {name}. Waiting for others..."})
                    print(f"[SERVER] Player joined: {name} ({addr})")
            else:
                c.send({"type": "error", "message": "Please send join first."})

        if name is None:
            print(f"[SERVER] Join timeout from {addr}, closing.")
            c.close()

    names = list(joined.keys())
    for nm, c in joined.items():
        c.send({"type": "start", "players": names, "message": "Game starting!"})
    return joined


def build_table(human_names: List[str], n_ai: int, starting_stack: int, sb: int) -> "eng.TableState":
    players: List["eng.Player"] = []
    stacks: Dict[str, int] = {}
    hero = human_names[0]

    for nm in human_names:
        players.append(eng.Player(name=nm, is_human=True))
        stacks[nm] = starting_stack

    next_ai_id = 1
    for _ in range(n_ai):
        nm = f"AI {next_ai_id}"
        next_ai_id += 1
        ai_key = eng.random.choice(list(eng.AI_TYPES.keys()))
        players.append(eng.Player(name=nm, is_human=False, ai_type_key=ai_key))
        stacks[nm] = starting_stack

    initial_specs = [(p.name, p.is_human, getattr(p, "ai_type_key", None)) for p in players]
    t = eng.TableState(players=players, stacks=stacks, hero_name=hero, next_ai_id=next_ai_id, starting_stack=starting_stack, sb=sb, initial_specs=initial_specs)
    return t


def rotate_dealer(dealer: int, n: int) -> int:
    return (dealer + 1) % n if n > 0 else 0


def run_server(args):
    clients = wait_for_humans(args.host, args.port, humans=args.humans)
    io = MultiplayerIO(clients, allow_cheats=args.allow_cheats)
    patch_engine_for_multiplayer(io)

    human_names = list(clients.keys())
    t = build_table(human_names, n_ai=args.ai, starting_stack=args.stack, sb=args.sb)

    dealer = 0
    hand_no = 1

    print("[SERVER] Game loop started.")
    while True:
        eng.cleanup_busted(t)

        if len(t.players) < 2:
            msg = {"type": "gameover", "message": "Not enough players left to continue."}
            for c in clients.values():
                try:
                    c.send(msg)
                except Exception:
                    pass
            print("[SERVER] Game over.")
            break

        h = eng.deal_new_hand(t, dealer_idx=dealer, sb=t.sb)
        h.hand_no = getattr(h, "hand_no", hand_no)
        io.broadcast_state(h, t)

        eng.betting_round(h, t)
        if t.request_restart:
            t.request_restart = None
            continue

        if not h.hand_done:
            while not h.hand_done and h.street != "showdown":
                eng.reveal_next_street(h)
                io.broadcast_state(h, t)
                eng.betting_round(h, t)
                if h.hand_done:
                    break
                if h.street == "river":
                    eng.reveal_next_street(h)
                    break

        if not h.hand_done:
            eng.resolve_showdown(h)

        # Showdown: reveal all hole cards to both humans
        h.street = "showdown"
        io.broadcast_state(h, t)

        t.stacks = dict(h.stacks)
        dealer = rotate_dealer(dealer, len(t.players))
        hand_no += 1
        time.sleep(0.2)


def main():
    ap = argparse.ArgumentParser(description="Texas Hold'em Multiplayer Server")
    ap.add_argument("--host", default="0.0.0.0", help="Host/IP to bind (default 0.0.0.0)")
    ap.add_argument("--port", type=int, default=7777, help="Port to bind (default 7777)")
    ap.add_argument("--humans", type=int, default=2, help="Number of human players to wait for (default 2)")
    ap.add_argument("--ai", type=int, default=0, help="Number of AI opponents to add (default 0)")
    ap.add_argument("--stack", type=int, default=10000, help="Starting stack for each player")
    ap.add_argument("--sb", type=int, default=50, help="Small blind amount")
    ap.add_argument("--allow-cheats", action="store_true", help="Allow cheat codes from clients (NOT fair)")
    args = ap.parse_args()
    run_server(args)


if __name__ == "__main__":
    main()
