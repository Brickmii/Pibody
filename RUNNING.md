# Running PBAI

PBAI runs as a permanent systemd service on the Raspberry Pi.
SSH in from anywhere to monitor, chat, or control it.

## Services

| Service | What it does |
|---------|-------------|
| `pbai-daemon` | Main PBAI daemon — manifold, environment core, API on port 8420, body WebSocket on 8421 |
| `pbai-heatdrip` | Injects ego=5K + identity=3K every 3 seconds to keep PBAI active |

Both start on boot and auto-restart on failure.

## CLI Reference

The `pbai` command is available system-wide after install.

### `pbai status`
Show daemon state, loop count, and psychology (Identity/Ego/Conscience heat, axes, existence).

```
$ pbai status
State:  running
Loops:  838
Driver: minecraft

  identity     heat=98.240  axes=4  existence=actual
  ego          heat=31.060  axes=3  existence=actual
  conscience   heat=0.059  axes=6  existence=actual

  exploration_rate: 0.433
```

### `pbai chat "your message"`
Send natural language to PBAI. Activates motion verbs that match words in the text, which bias action selection toward those behaviors.

```
$ pbai chat "go explore and find structures"
Activated: bm_explore, bm_find
Bus:
  bm_explore       0.70 ##############
  bm_find          0.70 ##############
```

Verb activations decay over ~7 cycles (0.85 per tick). Chat activations start at 0.7 (strongest source). The 20 verbs are:

| Fire | Verbs |
|------|-------|
| Heat (Magnitude) | take, get |
| Polarity (Differentiation) | easily, quickly, better |
| Existence (Perception) | see, view, visual, visually, visualize |
| Righteousness (Evaluation) | analyze, understand, identify |
| Order (Construction) | create, build, design, make |
| Movement (Navigation) | explore, find, discover |

### `pbai bus`
Show the current MotionBus state — which verbs are active and what action weight boosts they produce.

```
$ pbai bus
Active verbs:
  bm_explore       0.50 ##########
  bm_find          0.50 ##########
  bm_see           0.20 ####

Action boosts:
  move_forward         x2.5
  sprint_forward       x2.5
  explore_left         x2.5
  scout_ahead          x2.5
```

### `pbai heat [amount]`
Manually inject heat into Ego and Identity. Default amount is 5.

```
$ pbai heat 10
Injected ego=10.0 identity=6.0
```

### `pbai manifold`
Show manifold summary — node count, total heat, and the first 50 nodes.

### `pbai logs`
Tail the daemon logs live (via journalctl). Press Ctrl-C to stop.

### `pbai restart`
Restart the daemon service. Requires sudo.

## Service Management

```bash
# Start/stop/restart
sudo systemctl start pbai-daemon
sudo systemctl stop pbai-daemon
sudo systemctl restart pbai-daemon

# Check status
sudo systemctl status pbai-daemon

# View logs
journalctl -u pbai-daemon -f              # live tail
journalctl -u pbai-daemon --since "5m ago" # last 5 minutes
journalctl -u pbai-daemon -n 100          # last 100 lines

# Heat drip (starts/stops with daemon)
sudo systemctl start pbai-heatdrip
sudo systemctl stop pbai-heatdrip

# Disable auto-start on boot
sudo systemctl disable pbai-daemon
sudo systemctl disable pbai-heatdrip
```

## API Endpoints

The REST API runs on port 8420. All endpoints accept JSON.

| Method | Path | Description |
|--------|------|-------------|
| GET | `/status` | Daemon state, loop count, active driver |
| GET | `/psychology` | Identity/Ego/Conscience heat and existence |
| GET | `/manifold` | Node count, total heat, first 50 nodes |
| GET | `/motion_bus` | Active verbs and action weight boosts |
| GET | `/environments` | Registered environments and stats |
| GET | `/thermal` | Thermal manager state |
| POST | `/chat` | Send text, activate motion verbs. Body: `{"text": "..."}` |
| POST | `/inject/heat` | Inject heat. Body: `{"target": "ego", "amount": 5}` |
| POST | `/inject/perception` | Inject a perception event |
| POST | `/body/action` | Send action to PC body (driver-routed or raw motor) |
| POST | `/request_world` | Request world state from PC client |
| POST | `/pause` | Pause daemon |
| POST | `/resume` | Resume daemon |
| POST | `/save` | Force save manifold |

### Chat examples from any machine

```bash
# From another computer on the network
curl -X POST http://<pi-ip>:8420/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "explore and find resources"}'

# Check what verbs are active
curl http://<pi-ip>:8420/motion_bus
```

## Architecture

```
                    SSH / curl
                        |
                   [REST API :8420]
                        |
                  PBAIDaemon
                   /        \
          EnvironmentCore    ThermalManager
           /     |     \
    MotionBus  Driver  Introspector
       |         |         |
   verb acts   MC port   transformer
       |         |       attention
       ↓         ↓
    decide()   body_server ←→ PC Client (WebSocket :8421)
```

## Files

| File | Purpose |
|------|---------|
| `pi/daemon.py` | Main daemon loop |
| `pi/api.py` | REST API handler |
| `pi/pbai` | CLI tool |
| `pi/pbai-daemon.service` | systemd service (daemon) |
| `pi/pbai-heatdrip.service` | systemd service (heat drip) |
| `drivers/environment.py` | EnvironmentCore + MotionBus |
| `core/introspector.py` | Transformer attention introspector |
| `drivers/minecraft/minecraft_driver.py` | Minecraft Bedrock driver |
| `growth/` | Manifold persistence (nodes.json, conscience.json, etc.) |

## Install (one time)

```bash
sudo cp pi/pbai-daemon.service /etc/systemd/system/
sudo cp pi/pbai-heatdrip.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pbai-daemon pbai-heatdrip
sudo ln -sf /home/pbai/pibody/pi/pbai /usr/local/bin/pbai
sudo systemctl start pbai-daemon pbai-heatdrip
```
