"""
PBAI Thermal Integration - Raspberry Pi 5 (Planck-Grounded)

Maps physical CPU temperature to manifold tick rate.
The heat metaphor becomes literal - silicon heat = cognitive constraint.

════════════════════════════════════════════════════════════════════════════════
PLANCK GROUNDING
════════════════════════════════════════════════════════════════════════════════

BODY TEMPERATURE REFERENCE:
    The system's "comfortable" operating temperature is grounded in
    BODY_TEMPERATURE (K × φ¹¹ ≈ 304 K = 31°C).
    
    This creates a meaningful link between:
    - Silicon temperature (CPU)
    - Manifold heat (psychology)  
    - Biological temperature (body temp)

FIRE HEAT SCALING:
    Temperature zones follow K × φⁿ scaling:
    - Fire 1 zone: K × φ¹ ≈ 2.5 K above baseline
    - Fire 2 zone: K × φ² = 4 K above baseline
    - etc.
    
    This grounds thermal throttling in the same physics as the manifold.

EMBODIMENT:
    - Hot CPU → slower ticks (forced rest)
    - Cool CPU → faster ticks (can think more)
    - Thermal throttling = metabolic throttling
    
TEMPERATURE ZONES (Planck-Grounded):
    Baseline = 40°C (comfortable silicon temp)
    Danger anchor = 80°C (40°C above baseline)

    Zone widths follow φ-ratio scaling (ZONE_UNIT = 5K ≈ 7.6°C):

    < ~48°C  (baseline + 5K)         : Cool     - Maximum tick rate (1/φ²)
    < ~60°C  (baseline + 5K(1+φ))    : Warm     - Normal tick rate (1.0)
    < 80°C   (baseline + 40)         : Hot      - Reduced tick rate (φ)
    < 85°C                           : Danger   - Severe throttle (φ²)
    > 85°C                           : Critical - Pause entirely

════════════════════════════════════════════════════════════════════════════════
"""

import logging
import os
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Import Planck constants
try:
    from core.node_constants import (
        K, PHI,
        BODY_TEMPERATURE,
        FIRE_HEAT,
    )
except ImportError:
    PHI = 1.618033988749895
    K = 4 / (PHI ** 2)
    BODY_TEMPERATURE = K * PHI ** 11  # ≈ 304 K
    FIRE_HEAT = {i: K * PHI ** i for i in range(1, 7)}


# ═══════════════════════════════════════════════════════════════════════════════
# TEMPERATURE ZONES (Planck-Grounded)
# ═══════════════════════════════════════════════════════════════════════════════

# Baseline comfortable temperature (Celsius)
TEMP_BASELINE = 40.0

# Zone thresholds: φ-ratio zone widths anchored to danger = 80°C
# ZONE_UNIT = 5K ≈ 7.64°C — zone widths grow by φ each step
# Proof: 5K(1 + φ + φ²) = 5K·2φ² = 10·Kφ² = 10·4 = 40 (exact)
# So 3 zones fill exactly 40°C from baseline to danger.
ZONE_UNIT = 5 * K                              # ≈ 7.64°C (grounded: 5 thermal quanta)

TEMP_COOL    = TEMP_BASELINE + ZONE_UNIT                    # ~47.6°C
TEMP_WARM    = TEMP_BASELINE + ZONE_UNIT * (1.0 + PHI)      # ~60.0°C
TEMP_DANGER  = TEMP_BASELINE + 40.0                          # 80.0°C (exact)
TEMP_CRITICAL = 85.0                                         # Hard limit - pause

# Tick interval multipliers for each zone
# These use 1/φⁿ scaling (same as existence thresholds)
MULTIPLIER_COOL = 1 / (PHI ** 2)      # ≈ 0.38 - faster
MULTIPLIER_WARM = 1.0                  # Normal
MULTIPLIER_HOT = PHI                   # ≈ 1.62 - slower
MULTIPLIER_DANGER = PHI ** 2           # ≈ 2.62 - much slower
MULTIPLIER_CRITICAL = float('inf')     # Paused


@dataclass
class ThermalState:
    """
    Current thermal state of the Pi (Planck-Grounded).
    
    Includes fire_level indicating which Fire zone the temperature is in.
    """
    temperature: float          # CPU temp in Celsius
    zone: str                   # 'cool', 'warm', 'hot', 'danger', 'critical'
    tick_multiplier: float      # How much to slow down ticks
    fan_speed: Optional[int]    # Fan speed if controllable (0-255)
    throttled: bool             # Is the CPU being throttled by the OS?
    
    # Planck grounding
    fire_level: int = 1         # Which fire zone (1-6) based on temperature
    temp_above_baseline: float = 0.0  # Degrees above comfortable baseline
    
    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "zone": self.zone,
            "tick_multiplier": self.tick_multiplier,
            "fan_speed": self.fan_speed,
            "throttled": self.throttled,
            "fire_level": self.fire_level,
            "temp_above_baseline": self.temp_above_baseline,
        }


def read_cpu_temp() -> float:
    """
    Read CPU temperature from the Pi.
    
    Returns:
        Temperature in Celsius, or -1 if unavailable
    """
    # Primary method: thermal zone (Raspberry Pi)
    thermal_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(thermal_path):
        try:
            with open(thermal_path, 'r') as f:
                # Value is in millidegrees
                return int(f.read().strip()) / 1000.0
        except Exception as e:
            logger.warning(f"Failed to read thermal_zone0: {e}")
    
    # Fallback: vcgencmd (Raspberry Pi specific)
    try:
        import subprocess
        result = subprocess.run(
            ['vcgencmd', 'measure_temp'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            # Output: "temp=45.0'C"
            temp_str = result.stdout.strip()
            temp = float(temp_str.split('=')[1].replace("'C", ""))
            return temp
    except Exception as e:
        logger.debug(f"vcgencmd not available: {e}")
    
    # Not on a Pi or can't read temp
    return -1.0


def read_throttle_state() -> bool:
    """
    Check if the CPU is being throttled.
    
    Returns:
        True if throttled, False otherwise
    """
    try:
        import subprocess
        result = subprocess.run(
            ['vcgencmd', 'get_throttled'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            # Output: "throttled=0x0" (0 = not throttled)
            hex_val = result.stdout.strip().split('=')[1]
            return int(hex_val, 16) != 0
    except Exception:
        pass
    return False


def get_fire_level(temperature: float) -> int:
    """
    Determine which Fire zone the temperature is in.
    
    Uses K × φⁿ scaling from baseline.
    
    Returns:
        Fire level 1-6 (1 = coolest, 6 = hottest)
    """
    if temperature < 0:
        return 1
    
    above_baseline = temperature - TEMP_BASELINE
    # Check each fire level threshold
    for fire in range(6, 0, -1):
        if above_baseline >= FIRE_HEAT[fire]:
            return fire
    
    return 1  # Below Fire 1


def get_zone(temperature: float) -> Tuple[str, float, int]:
    """
    Determine thermal zone, tick multiplier, and fire level from temperature.
    
    Args:
        temperature: CPU temperature in Celsius
        
    Returns:
        (zone_name, tick_multiplier, fire_level)
    """
    fire_level = get_fire_level(temperature)
    
    if temperature < 0:
        # Can't read temp - assume warm
        return ('unknown', MULTIPLIER_WARM, 1)
    elif temperature < TEMP_COOL:
        return ('cool', MULTIPLIER_COOL, fire_level)
    elif temperature < TEMP_WARM:
        return ('warm', MULTIPLIER_WARM, fire_level)
    elif temperature < TEMP_DANGER:
        return ('hot', MULTIPLIER_HOT, fire_level)
    elif temperature < TEMP_CRITICAL:
        return ('danger', MULTIPLIER_DANGER, fire_level)
    else:
        return ('critical', MULTIPLIER_CRITICAL, 6)


def get_thermal_state() -> ThermalState:
    """
    Get complete thermal state of the Pi (Planck-Grounded).
    
    Returns:
        ThermalState with current readings and fire level
    """
    temp = read_cpu_temp()
    zone, multiplier, fire_level = get_zone(temp)
    throttled = read_throttle_state()
    
    # Fan speed would come from GPIO/HAT - None for now
    fan_speed = None
    
    return ThermalState(
        temperature=temp,
        zone=zone,
        tick_multiplier=multiplier,
        fan_speed=fan_speed,
        throttled=throttled,
        fire_level=fire_level,
        temp_above_baseline=max(0, temp - TEMP_BASELINE) if temp > 0 else 0,
    )


class ThermalManager:
    """
    Manages thermal state and provides tick rate adjustments (Planck-Grounded).
    
    Integrates with the Clock to enforce physical constraints.
    Uses Fire heat scaling (K × φⁿ) for zone transitions.
    """
    
    def __init__(self, enable_fan_control: bool = False):
        """
        Initialize thermal manager.
        
        Args:
            enable_fan_control: If True, attempt to control Pi fan via GPIO
        """
        self.enable_fan_control = enable_fan_control
        self.history: list = []  # Recent temperature readings
        self.max_history = 60    # Keep last 60 readings
        
        # Fire level history for trend detection
        self.fire_history: list = []
        
        # Fan control (if enabled)
        self.fan_gpio_pin = 14   # Default fan control pin
        self.fan_pwm = None
        
        if enable_fan_control:
            self._init_fan_control()
        
        logger.info(f"ThermalManager initialized (Planck-grounded, ZONE_UNIT=5K)")
        logger.info(f"  Baseline: {TEMP_BASELINE}°C")
        logger.info(f"  Cool < {TEMP_COOL:.1f}°C")
        logger.info(f"  Warm < {TEMP_WARM:.1f}°C")
        logger.info(f"  Hot  < {TEMP_DANGER:.1f}°C")
        logger.info(f"  Danger < {TEMP_CRITICAL:.1f}°C")
    
    def _init_fan_control(self):
        """Initialize GPIO fan control."""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.fan_gpio_pin, GPIO.OUT)
            self.fan_pwm = GPIO.PWM(self.fan_gpio_pin, 25000)  # 25kHz
            self.fan_pwm.start(0)
            logger.info("Fan control initialized")
        except Exception as e:
            logger.warning(f"Fan control not available: {e}")
            self.fan_pwm = None
    
    def set_fan_speed(self, speed: int):
        """
        Set fan speed (0-100).
        
        Args:
            speed: Fan speed percentage (0 = off, 100 = full)
        """
        if self.fan_pwm:
            self.fan_pwm.ChangeDutyCycle(max(0, min(100, speed)))
    
    def update(self) -> ThermalState:
        """
        Update thermal state and adjust fan if needed.
        
        Returns:
            Current ThermalState with Planck grounding
        """
        state = get_thermal_state()
        
        # Record history
        self.history.append(state.temperature)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Record fire level history
        self.fire_history.append(state.fire_level)
        if len(self.fire_history) > self.max_history:
            self.fire_history.pop(0)
        
        # Auto fan control based on fire level
        if self.fan_pwm:
            # Fan speed scales with fire level
            if state.fire_level <= 1:
                self.set_fan_speed(0)
            elif state.fire_level <= 2:
                self.set_fan_speed(20)
            elif state.fire_level <= 3:
                self.set_fan_speed(40)
            elif state.fire_level <= 4:
                self.set_fan_speed(60)
            elif state.fire_level <= 5:
                self.set_fan_speed(80)
            else:  # Fire 6
                self.set_fan_speed(100)
        
        return state
    
    def get_tick_multiplier(self) -> float:
        """
        Get current tick rate multiplier based on thermal state.
        
        Returns:
            Multiplier to apply to base tick interval
        """
        state = self.update()
        return state.tick_multiplier
    
    def should_pause(self) -> bool:
        """
        Check if system should pause due to critical temperature.
        
        Returns:
            True if too hot to continue
        """
        state = get_thermal_state()
        return state.zone == 'critical'
    
    def get_average_temp(self) -> float:
        """Get average temperature over history."""
        if not self.history:
            return -1.0
        return sum(self.history) / len(self.history)
    
    def get_average_fire_level(self) -> float:
        """Get average fire level over history."""
        if not self.fire_history:
            return 1.0
        return sum(self.fire_history) / len(self.fire_history)
    
    def is_heating_up(self) -> bool:
        """Check if temperature is trending upward."""
        if len(self.history) < 10:
            return False
        recent = self.history[-5:]
        older = self.history[-10:-5]
        return sum(recent) / 5 > sum(older) / 5
    
    def cleanup(self):
        """Clean up GPIO resources."""
        if self.fan_pwm:
            self.fan_pwm.stop()
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup(self.fan_gpio_pin)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION MODE (for testing without a Pi)
# ═══════════════════════════════════════════════════════════════════════════════

class SimulatedThermalManager(ThermalManager):
    """
    Simulated thermal manager for testing without hardware (Planck-Grounded).
    
    Temperature varies based on simulated load, following Fire heat scaling.
    """
    
    def __init__(self):
        super().__init__(enable_fan_control=False)
        self._simulated_temp = TEMP_BASELINE + 5.0  # Start slightly above baseline
        self._load = 0.0  # 0.0 to 1.0
    
    def set_load(self, load: float):
        """Set simulated load (0.0 to 1.0)."""
        self._load = max(0.0, min(1.0, load))
    
    def update(self) -> ThermalState:
        """Update simulated temperature based on load."""
        # Temperature trends toward load-based target
        # Load 0.0 = baseline, Load 1.0 = danger zone
        target = TEMP_BASELINE + (self._load * (TEMP_DANGER - TEMP_BASELINE))
        
        # Gradual change
        diff = target - self._simulated_temp
        self._simulated_temp += diff * 0.1
        
        # Add some noise
        import random
        self._simulated_temp += random.uniform(-0.5, 0.5)
        self._simulated_temp = max(30.0, min(90.0, self._simulated_temp))
        
        # Record
        self.history.append(self._simulated_temp)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        zone, multiplier, fire_level = get_zone(self._simulated_temp)
        
        # Record fire level
        self.fire_history.append(fire_level)
        if len(self.fire_history) > self.max_history:
            self.fire_history.pop(0)
        
        return ThermalState(
            temperature=self._simulated_temp,
            zone=zone,
            tick_multiplier=multiplier,
            fan_speed=None,
            throttled=self._simulated_temp > TEMP_DANGER,
            fire_level=fire_level,
            temp_above_baseline=max(0, self._simulated_temp - TEMP_BASELINE),
        )


def create_thermal_manager(simulated: bool = False) -> ThermalManager:
    """
    Create appropriate thermal manager (Planck-Grounded).
    
    Args:
        simulated: If True, use simulated thermal manager
        
    Returns:
        ThermalManager instance with Fire heat scaling
    """
    if simulated:
        return SimulatedThermalManager()
    
    # Check if we're on a Pi
    if read_cpu_temp() < 0:
        logger.info("Not on Raspberry Pi - using simulated thermal manager")
        return SimulatedThermalManager()
    
    return ThermalManager(enable_fan_control=True)
