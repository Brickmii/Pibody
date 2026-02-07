"""
PBAI Thermal Manifold - Tasks Package

Tasks are environments that PBAI can interact with.
Each task is a complete environment with its own driver.

Available tasks:
- chat_client: GUI chat interface with node visualization
- voice_client: Voice/text interface for PBAI interaction
- gym_runner: Run PBAI agent on Gymnasium environments
- unified_gui: Combined GUI for gym + voice
- bigmaze: Large maze environment
- blackjack: Blackjack card game environment
"""

from .chat_client import run_chat_client

__all__ = ['run_chat_client']
