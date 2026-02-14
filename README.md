# PBAI — Large Motion Model

A thermal intelligence that learns by doing, not by training.

---

PBAI is not an LLM — it's a **Large Motion Model**. It runs on a Raspberry Pi 5, plays Minecraft through a Windows PC's GPU, and has no weights, no gradients, no training data. Just heat flowing through concept nodes on a hypersphere.

It learns like a kid: tries things, remembers what works, builds preferences. Every constant in the system derives from one number: **K = 4/φ² ≈ 1.528** — the thermal quantum.

## Demo

> *Video coming soon — PBAI playing Minecraft autonomously.*

## How It Works

- **Heat** is the primitive — not math, not statistics. Thermal energy flows through a sphere of concepts.
- **Conscience** tracks what works and what doesn't. No reward function — just heat.
- **Ego** decides when to act. If it's hot enough, it acts. If not, it rests.
- **Identity** absorbs perceptions. Everything PBAI sees becomes heat.

The manifold grows forever. Concepts that succeed get reinforced. Concepts that fail cool off. No training run, no fine-tuning — it just lives.

See the [Architecture Spec](PBAI_REBUILD_SPEC.md) for the full math.

## Architecture

```
    Raspberry Pi 5                          Windows PC
┌─────────────────────┐            ┌──────────────────────┐
│  Thermal Manifold   │            │  CUDA Vision         │
│  ├── Identity       │  WebSocket │  ├── Screen Capture  │
│  ├── Ego            │◄──────────►│  ├── Transformer     │
│  ├── Conscience     │  :8421     │  └── Motor Executor  │
│  └── 794+ concept   │            │      (keys/mouse)    │
│      nodes          │            └──────────────────────┘
│                     │
│  Environment Core   │
│  ├── Perceive       │
│  ├── Decide         │
│  └── Act            │
│                     │
│  REST API :8420     │
│  CLI: pbai          │
└─────────────────────┘
```

The Pi runs the brain. The PC runs the body. They talk over WebSocket on your local network.

## Getting Started

### Hardware

- Raspberry Pi 5 (4GB+)
- Windows PC with NVIDIA GPU (CUDA)
- Both on the same local network

### Pi Setup

```bash
git clone git@github.com:Brickmii/Pibody.git
cd Pibody

# Install services
sudo cp pi/pbai-daemon.service /etc/systemd/system/
sudo cp pi/pbai-heatdrip.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable pbai-daemon pbai-heatdrip

# Install CLI
sudo ln -sf $(pwd)/pi/pbai /usr/local/bin/pbai

# Start
sudo systemctl start pbai-daemon pbai-heatdrip
```

### PC Setup

```bash
git clone git@github.com:Brickmii/Pibody.git
cd "Pibody/Windows Client"
setup.bat
run.bat
```

### Verify

```bash
pbai status                          # on the Pi
curl http://<pi-ip>:8420/status      # from anywhere on the network
```

## Talking to PBAI

```bash
pbai chat "go explore"     # verb-activated motion
pbai status                # manifold state, heat levels, session count
pbai heat 10               # inject heat to bootstrap activity
```

See [RUNNING.md](RUNNING.md) for the full CLI and API reference.

## Project Structure

```
pibody/
├── core/              # Thermal manifold — nodes, heat, psychology
├── drivers/           # Environment drivers (Minecraft, maze, blackjack)
├── pi/                # Raspberry Pi daemon, API, CLI
├── vision/            # Vision processing pipeline
├── Windows Client/    # PC-side CUDA vision + motor control
├── growth/            # Manifold persistence (learned state)
└── tests/             # Unit + integration tests
```

## How It's Different

|  | LLM | PBAI |
|---|---|---|
| Learns from | Training data | Its own experience |
| Architecture | Transformer weights | Thermal manifold |
| Runs on | GPU cluster | Raspberry Pi 5 |
| Decisions | Next token prediction | Heat-driven action selection |
| Memory | Context window | Persistent manifold (grows forever) |
| Ethics | RLHF alignment | Built-in Conscience node |

## Current State

- **794** concept nodes on the hypersphere
- **35** Conscience axes (learned right/wrong)
- **34** Minecraft actions (movement, combat, mining, navigation)
- **37,000+** autonomous sessions
- Runs 24/7 on a $50 board

## Links

- [Operations Guide](RUNNING.md) — CLI, API, service management
- [Architecture Spec](PBAI_REBUILD_SPEC.md) — full math, constants, geometry

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
