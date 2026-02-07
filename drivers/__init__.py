"""
PBAI Thermal Manifold - Drivers Package

The drivers package handles all environment interaction.

Usage:
    from drivers import EnvironmentCore, GymDriver, create_gym_driver
    
    # Create manifold and driver
    manifold = get_pbai_manifold()
    driver = create_gym_driver("FrozenLake-v1", manifold)
    
    # Create environment core and register driver
    env_core = EnvironmentCore(manifold)
    env_core.register_driver(driver)
    env_core.activate_driver(driver.DRIVER_ID)
    
    # Run step
    action, result, heat = env_core.step()
"""

from .environment import (
    # Port Protocol
    PortState, PortMessage, Port, NullPort,
    
    # Perception/Action
    Perception, Action, ActionResult,
    
    # Driver Base
    Driver, MockDriver,
    
    # Environment Core
    EnvironmentCore, DriverLoader,
    
    # Factory
    create_environment_core
)

__all__ = [
    # Port Protocol
    'PortState', 'PortMessage', 'Port', 'NullPort',

    # Perception/Action
    'Perception', 'Action', 'ActionResult',

    # Driver Base
    'Driver', 'MockDriver',

    # Environment Core
    'EnvironmentCore', 'DriverLoader',

    # Factory
    'create_environment_core',
]
