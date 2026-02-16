"""
PBAI Pi Deployment Package

Run PBAI continuously on Raspberry Pi 5 with:
- Thermal-regulated tick loop
- Multi-environment choice
- REST API for remote control (:8420)
- Body server for remote bodies (:8421)
- Persistent growth to NVMe

USAGE:
    # On Pi:
    python -m pi.daemon --port 8420 --body-port 8421
    
    # Or as systemd service:
    sudo cp pi/pbai.service /etc/systemd/system/
    sudo systemctl enable pbai
    sudo systemctl start pbai
    
    # Remote control:
    from pi.api import PBAIClient
    client = PBAIClient("192.168.1.100", 8420)
    print(client.get_status())

THERMAL ZONES:
    < 50°C  : Cool    - Maximum tick rate
    50-65°C : Warm    - Normal tick rate  
    65-75°C : Hot     - Reduced tick rate
    > 75°C  : Danger  - Minimum tick rate
    > 80°C  : Critical - Pause entirely
"""

from .thermal import (
    ThermalManager, ThermalState,
    create_thermal_manager, read_cpu_temp, get_zone,
    TEMP_COOL, TEMP_WARM, TEMP_DANGER, TEMP_CRITICAL
)

from .daemon import (
    PBAIDaemon, DaemonState, DaemonStats,
    EnvironmentChooser, EnvironmentStats,
    run_daemon
)

from .api import (
    PBAIClient, PBAIAPIServer,
    create_api_server, run_api_server
)

from .body_server import BodyServer

__all__ = [
    # Thermal
    'ThermalManager', 'ThermalState', 'create_thermal_manager',
    'read_cpu_temp', 'get_zone',
    'TEMP_COOL', 'TEMP_WARM', 'TEMP_DANGER', 'TEMP_CRITICAL',
    
    # Daemon
    'PBAIDaemon', 'DaemonState', 'DaemonStats',
    'EnvironmentChooser', 'EnvironmentStats',
    'run_daemon',
    
    # API
    'PBAIClient', 'PBAIAPIServer',
    'create_api_server', 'run_api_server',
    
    # Body Server
    'BodyServer',
]
