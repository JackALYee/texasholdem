#!/usr/bin/env python3
"""Texas Hold'em multiplayer (terminal) — Client

Connects to server.py and plays as a human in your own terminal.
"""

import argparse
import json
import socket
import sys
import threading


def send_json(sock: socket.socket, obj: dict):
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    sock.sendall(data)


def recv_loop(sock: socket.socket):
    buf = b""
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            print("\n[Disconnected]")
            sys.exit(0)
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode("utf-8"))
            except Exception:
                print("[Bad message]", line.decode("utf-8", errors="replace"))
                continue
            handle_msg(sock, msg)


def clear():
    print("\033[2J\033[H", end="")


def fmt_state(st: dict) -> str:
    lines = []
    lines.append("=== Texas Hold'em Multiplayer ===")
    lines.append(f"Street: {st.get('street','')}   Pot: {st.get('pot',0)}   Current bet: {st.get('current_bet',0)}")
    board = st.get("board") or []
    lines.append("Board: " + (" ".join(board) if board else "(preflop)"))
    your_hole = st.get("your_hole") or []
    lines.append("Your hole: " + (" ".join(your_hole) if your_hole else "?? ??"))
    lines.append("")
    lines.append("Players:")
    for p in st.get("players", []):
        nm = p["name"]
        stack = p["stack"]
        status = p["status"]
        cards = " ".join(p.get("cards", ["??","??"]))
        lines.append(f"  - {nm:10s}  stack={stack:6d}  {status:7s}  {cards}")
    lines.append("")
    info = st.get("info") or ""
    if info:
        lines.append(f"[INFO] {info}")
        lines.append("")
    lines.append("Recent log:")
    for ln in st.get("recent_log", [])[-30:]:
        lines.append(" • " + ln)
    lines.append("")
    win = st.get("winner_text") or ""
    if win:
        lines.append("Winner: " + win)
        lines.append("")
    return "\n".join(lines)


def handle_msg(sock: socket.socket, msg: dict):
    t = msg.get("type")
    if t == "welcome":
        print(msg.get("message",""))
        return
    if t == "start":
        print(msg.get("message",""))
        print("Players:", ", ".join(msg.get("players", [])))
        return
    if t == "error":
        print("[ERROR]", msg.get("message",""))
        return
    if t == "gameover":
        print(msg.get("message","Game over."))
        sys.exit(0)
    if t == "state":
        clear()
        print(fmt_state(msg))
        if msg.get("needs_input"):
            prompt = msg.get("prompt") or "Your action: "
            try:
                s = input(prompt).strip()
            except EOFError:
                s = "quit"
            send_json(sock, {"type": "action", "text": s})
        return
    print("[MSG]", msg)


def main():
    ap = argparse.ArgumentParser(description="Texas Hold'em Multiplayer Client")
    ap.add_argument("--host", default="127.0.0.1", help="Server host/IP")
    ap.add_argument("--port", type=int, default=7777, help="Server port")
    ap.add_argument("--name", required=True, help="Your player name")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    send_json(sock, {"type":"join", "name": args.name})

    t = threading.Thread(target=recv_loop, args=(sock,), daemon=True)
    t.start()

    try:
        while True:
            t.join(1)
    except KeyboardInterrupt:
        try:
            send_json(sock, {"type":"action", "text":"quit"})
        except Exception:
            pass
        sock.close()


if __name__ == "__main__":
    main()
