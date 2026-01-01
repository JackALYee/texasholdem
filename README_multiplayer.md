# Texas Hold'em Multiplayer (Terminal)

This is a **2-human** (or N-human) multiplayer version of your terminal Texas Hold'em game.

Each human connects from their own terminal using `client.py`.
The server runs the engine and enforces that each player only sees:
- their own hole cards
- the community board
- public stacks / status / log

## Files
- `holdem_engine.py` — the game engine (copied from your working CLI version)
- `server.py` — hosts the game and runs the engine
- `client.py` — player terminal client
- `requirements.txt` — python deps (`treys`, `rich`)

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run (same machine)
Terminal A (server):
```bash
python3 server.py --port 7777 --humans 2 --ai 0 --stack 10000 --sb 50
```

Terminal B (player 1):
```bash
python3 client.py --host 127.0.0.1 --port 7777 --name Jack
```

Terminal C (player 2):
```bash
python3 client.py --host 127.0.0.1 --port 7777 --name Alice
```

## Run (different machines on same Wi‑Fi)
1) On the server machine, find its LAN IP (example: `192.168.1.50`)
2) Start the server (binds to all interfaces by default):
```bash
python3 server.py --port 7777
```
3) On the other machine:
```bash
python3 client.py --host 192.168.1.50 --port 7777 --name Alice
```

> Playing over the public internet generally requires port-forwarding or a VPN (Tailscale/ZeroTier).

## Notes
- Cheat codes are **disabled by default** in multiplayer (fair play).
- If you really want them (debug only), start server with:
```bash
python3 server.py --allow-cheats
```
