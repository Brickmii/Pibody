"""
PBAI Constraints - Diagnostic Technique

Automotive terminology as universal language for any domain.

════════════════════════════════════════════════════════════════════════════════
THE DIAGNOSTIC LOOP
════════════════════════════════════════════════════════════════════════════════

    1. MEASURE    - Observe current state
    2. SPEC       - What should it be? (Q frame if known)
    3. TOLERANCE  - Is deviation acceptable? (R threshold)
    4. SYMPTOM    - What's presenting? (observable problem)
    5. DIAGNOSIS  - What's the cause? (trace cold paths)
    6. PROCEDURE  - What's the fix? (ordered actions)
    7. VERIFY     - Did it work? (re-measure)
    8. LEARN      - Heat successful paths

════════════════════════════════════════════════════════════════════════════════
MOTION FUNCTION MAPPING
════════════════════════════════════════════════════════════════════════════════

    Heat (Σ)          - Accumulated verification strength
    Polarity (+/-)    - In spec (+1) or out of spec (-1)
    Existence (δ)     - Spec actual if heat > 1/φ³
    Righteousness (R) - Deviation from nominal
    Order (Q)         - Procedure steps (Robinson)
    Movement (Lin)    - Execute action (costs heat to output)

════════════════════════════════════════════════════════════════════════════════
HEAT FLOW
════════════════════════════════════════════════════════════════════════════════

    INTERNAL (no cost):
        Observe → heat flows in
        Learn → heat accumulates
        Connect → heat redistributes
        Think → heat moves along paths

    EXTERNAL (costs heat):
        Speak → COST_HEAT
        Act → COST_HEAT
        Move properly → COST_MOVEMENT

    Heat flows freely inside the manifold.
    Heat expends when you output to the environment.

════════════════════════════════════════════════════════════════════════════════
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from time import time

from .node_constants import (
    K, PHI, INV_PHI,
    THRESHOLD_EXISTENCE,
    THRESHOLD_RIGHTEOUSNESS,
    CONFIDENCE_EXPLOIT_THRESHOLD,
    righteousness_weight,
    COST_HEAT,
    # Robinson constraints for measurement types
    ROBINSON_CONSTRAINTS,
    ROBINSON_IDENTITY,
    ROBINSON_SUCCESSOR,
    ROBINSON_ADDITION,
    ROBINSON_MULTIPLICATION,
    get_robinson_constraint,
    CONSTRAINT_DESCRIPTIONS,
)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════

class Outcome(Enum):
    """Result of a procedure."""
    UNKNOWN = "unknown"      # Haven't tried
    SUCCESS = "success"      # Fixed the problem
    PARTIAL = "partial"      # Improved but not fixed
    FAILURE = "failure"      # Didn't work
    DAMAGE = "damage"        # Made it worse


class SpecSource(Enum):
    """How was this spec obtained?"""
    ROBINSON = "robinson"    # Minimum from single observation
    LEARNED = "learned"      # Tightened through verification
    DEFINED = "defined"      # Given (manufacturer spec)


# ═══════════════════════════════════════════════════════════════════════════════
# SPEC - The Torque Spec
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Spec:
    """
    The torque spec - known order parameters.
    
    Heat IS confidence. R IS deviation.
    
    MOTION FUNCTION MAPPING:
        heat      → accumulated verification strength (Σ)
        polarity  → in spec (+1) or out of spec (-1)
        existence → actual if heat > 1/φ³, else dormant
        R         → deviation from nominal (righteousness)
    
    CONFIDENCE THRESHOLD:
        5K heat = 5/6 confidence = exploit threshold
        One K per scalar motion function (heat, polarity, existence, R, order)
        The 6th (movement) IS the action
    
    ROBINSON CONSTRAINT TYPES:
        identity:       "What is this?" - pure existence (R=1)
                        Temperature-like measurements, categorical
        successor:      "Where is this?" - stepping through space (R=4φ/7)
                        Length/distance measurements, position
        addition:       "When is this?" - composing moments (R=4/3)
                        Time/duration measurements, sequences
        multiplication: "How much is this?" - scaling quantities (R=13/10)
                        Mass/weight/amount measurements, quantities
    
    The constraint_type affects how tolerance is interpreted and how
    the spec relates to Planck-scale physics.
    """
    nominal: float
    tolerance_low: float
    tolerance_high: float
    unit: str = ""
    source: SpecSource = SpecSource.ROBINSON
    
    # Robinson constraint type - what kind of measurement is this?
    # identity = existence/categorical, successor = spatial, 
    # addition = temporal, multiplication = quantitative
    constraint_type: str = "identity"
    
    # Heat accumulates through successful verification
    # Heat never subtracts - only proper movement expends it
    heat: float = 0.0
    
    # Polarity: in spec (+1) or out of spec (-1)
    polarity: int = 1
    
    @property
    def robinson_factor(self) -> float:
        """Get the Robinson constraint factor for this spec's type."""
        return get_robinson_constraint(self.constraint_type)
    
    @property
    def constraint_description(self) -> str:
        """Human-readable description of what this constraint measures."""
        return CONSTRAINT_DESCRIPTIONS.get(self.constraint_type, "Unknown measurement type")
    
    def in_tolerance(self, measured: float) -> bool:
        """Is the measurement within spec?"""
        return self.tolerance_low <= measured <= self.tolerance_high
    
    def deviation(self, measured: float) -> float:
        """
        R value - how far out of spec.
        
        R = 0 → aligned (in spec)
        R > 0 → misaligned (out of spec)
        
        Normalized to tolerance range AND scaled by Robinson constraint.
        Different measurement types have different natural variances.
        """
        if measured < self.tolerance_low:
            range_below = self.nominal - self.tolerance_low
            if range_below > 0:
                raw_deviation = (self.tolerance_low - measured) / range_below
            else:
                return float('inf')
        elif measured > self.tolerance_high:
            range_above = self.tolerance_high - self.nominal
            if range_above > 0:
                raw_deviation = (measured - self.tolerance_high) / range_above
            else:
                return float('inf')
        else:
            return 0.0
        
        # Scale by Robinson factor - different constraints have different "natural" deviations
        # Successor (spatial) measurements are most precise (R=0.92)
        # Multiplication (mass) measurements have more variance (R=1.3)
        return raw_deviation * self.robinson_factor
    
    def measure(self, value: float) -> dict:
        """
        Measure against spec. Returns all motion function values.
        
        This is the diagnostic measurement - compare observed to expected.
        """
        in_spec = self.in_tolerance(value)
        R = self.deviation(value)
        polarity = 1 if in_spec else -1
        
        # Confidence from heat
        # 5K = 5/6 threshold (one K per scalar motion function)
        confidence = min(1.0, self.heat / (5 * K))
        should_exploit = confidence >= CONFIDENCE_EXPLOIT_THRESHOLD
        
        # Heat flow factor from R (righteousness_weight)
        r_weight = righteousness_weight(R)
        
        # Existence from heat threshold
        existence = "actual" if self.heat >= THRESHOLD_EXISTENCE else "dormant"
        
        return {
            'value': value,
            'nominal': self.nominal,
            'in_spec': in_spec,
            'polarity': polarity,
            'R': R,
            'r_weight': r_weight,
            'heat': self.heat,
            'confidence': confidence,
            'should_exploit': should_exploit,
            'existence': existence,
            'specified': True,
            'constraint_type': self.constraint_type,
            'robinson_factor': self.robinson_factor,
        }
    
    def verify(self, value: float, success: bool) -> float:
        """
        Verification - observation comes in.
        
        Success: heat flows in (K added)
        Failure: no heat added (heat never subtracts)
        
        Returns heat change.
        """
        if success:
            self.heat += K
            self.polarity = 1
            if self.source == SpecSource.ROBINSON:
                self.source = SpecSource.LEARNED
            return K
        else:
            self.polarity = -1
            return 0.0
    
    @property
    def confidence(self) -> float:
        """Confidence derived from heat. 5K = 5/6 threshold."""
        return min(1.0, self.heat / (5 * K))
    
    @property
    def should_exploit(self) -> bool:
        """Above 5/6 threshold? Use spec. Below? Explore."""
        return self.confidence >= CONFIDENCE_EXPLOIT_THRESHOLD
    
    @property
    def existence(self) -> str:
        """Existence state from heat. Above 1/φ³ = actual."""
        return "actual" if self.heat >= THRESHOLD_EXISTENCE else "dormant"
    
    def to_dict(self) -> dict:
        return {
            "nominal": self.nominal,
            "tolerance_low": self.tolerance_low,
            "tolerance_high": self.tolerance_high,
            "unit": self.unit,
            "source": self.source.value,
            "constraint_type": self.constraint_type,
            "heat": self.heat,
            "polarity": self.polarity,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Spec':
        return cls(
            nominal=data["nominal"],
            tolerance_low=data["tolerance_low"],
            tolerance_high=data["tolerance_high"],
            unit=data.get("unit", ""),
            source=SpecSource(data.get("source", "robinson")),
            constraint_type=data.get("constraint_type", "identity"),
            heat=data.get("heat", 0.0),
            polarity=data.get("polarity", 1),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MEASUREMENT - An observation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Measurement:
    """An observation of current state."""
    name: str
    value: float
    timestamp: float = field(default_factory=time)
    spec: Optional[Spec] = None
    result: Optional[dict] = None  # From spec.measure()
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "result": self.result,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SYMPTOM - Observable indicating a problem
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Symptom:
    """
    An observable indicating a problem.
    
    Symptoms are what you notice. Causes are what's actually wrong.
    Multiple symptoms can share a cause. One cause can have multiple symptoms.
    """
    name: str
    measurement_name: str               # Which measurement shows this
    condition: str                      # "low", "high", "erratic", "absent"
    severity: float = 0.5               # 0 = minor, 1 = critical
    
    # Learning: what causes have produced this symptom?
    known_causes: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "measurement_name": self.measurement_name,
            "condition": self.condition,
            "severity": self.severity,
            "known_causes": self.known_causes,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Symptom':
        return cls(
            name=data["name"],
            measurement_name=data["measurement_name"],
            condition=data["condition"],
            severity=data.get("severity", 0.5),
            known_causes=data.get("known_causes", []),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# PROCEDURE - Ordered sequence of actions (Q frame)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Procedure:
    """
    Ordered sequence of actions - the repair procedure.
    
    This is the Q frame for fixing something.
    Each step has parameters (the torque specs for that step).
    Steps follow Robinson arithmetic: 0, S(0), S(S(0))...
    """
    name: str
    target_symptom: str                 # What this procedure fixes
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    # Learning through heat
    heat: float = 0.0
    attempts: int = 0
    
    def add_step(self, action: str, **params) -> int:
        """
        Add a step. Order is Robinson (successor function).
        Returns the step index.
        """
        idx = len(self.steps)
        self.steps.append({
            "action": action,
            "params": params,
            "order": idx,  # 0, S(0), S(S(0))...
        })
        return idx
    
    def get_step(self, idx: int) -> Optional[dict]:
        """Get step by index."""
        if 0 <= idx < len(self.steps):
            return self.steps[idx]
        return None
    
    def record_outcome(self, outcome: Outcome) -> float:
        """
        Record attempt outcome.
        
        Success: heat flows in (K added)
        Failure: no heat added
        
        Returns heat change.
        """
        self.attempts += 1
        if outcome == Outcome.SUCCESS:
            self.heat += K
            return K
        return 0.0
    
    @property
    def confidence(self) -> float:
        """Confidence derived from heat."""
        return min(1.0, self.heat / (5 * K))
    
    @property
    def should_exploit(self) -> bool:
        """Above 5/6 threshold?"""
        return self.confidence >= CONFIDENCE_EXPLOIT_THRESHOLD
    
    @property
    def success_rate(self) -> float:
        """Historical success rate."""
        if self.attempts == 0:
            return 0.0
        # Derive from heat: each success adds K
        successes = self.heat / K
        return successes / self.attempts
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "target_symptom": self.target_symptom,
            "steps": self.steps,
            "heat": self.heat,
            "attempts": self.attempts,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Procedure':
        return cls(
            name=data["name"],
            target_symptom=data["target_symptom"],
            steps=data.get("steps", []),
            heat=data.get("heat", 0.0),
            attempts=data.get("attempts", 0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC FRAME - Container for a domain
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DiagnosticFrame:
    """
    Diagnostic context for a domain.
    
    Contains specs, symptoms, procedures.
    The driver populates this. The manifold uses heat to navigate it.
    """
    domain: str
    specs: Dict[str, Spec] = field(default_factory=dict)
    symptoms: Dict[str, Symptom] = field(default_factory=dict)
    procedures: Dict[str, Procedure] = field(default_factory=dict)
    measurements: Dict[str, Measurement] = field(default_factory=dict)
    
    def add_spec(self, name: str, nominal: float,
                 tolerance: float = None,
                 tolerance_low: float = None,
                 tolerance_high: float = None,
                 unit: str = "",
                 source: SpecSource = SpecSource.DEFINED,
                 constraint_type: str = None) -> Spec:
        """
        Add a known spec.
        
        Defined specs start at 5K heat (confident).
        
        Args:
            name: Spec name
            nominal: Target value
            tolerance: Symmetric tolerance (+/-)
            tolerance_low: Lower bound
            tolerance_high: Upper bound
            unit: Display unit (e.g., "psi", "mm", "seconds")
            source: How was this spec obtained?
            constraint_type: Robinson constraint type. If None, inferred from unit:
                - "mm", "m", "in", "ft", etc. → "successor" (spatial)
                - "s", "ms", "min", "hr", etc. → "addition" (temporal)
                - "kg", "g", "lb", "oz", etc. → "multiplication" (mass)
                - anything else → "identity" (categorical)
        """
        if tolerance is not None:
            tolerance_low = nominal - tolerance
            tolerance_high = nominal + tolerance
        
        # Infer constraint type from unit if not specified
        if constraint_type is None:
            constraint_type = infer_constraint_from_unit(unit)
        
        spec = Spec(
            nominal=nominal,
            tolerance_low=tolerance_low if tolerance_low is not None else nominal,
            tolerance_high=tolerance_high if tolerance_high is not None else nominal,
            unit=unit,
            source=source,
            constraint_type=constraint_type,
            heat=5 * K if source == SpecSource.DEFINED else 0.0,
        )
        self.specs[name] = spec
        return spec
    
    def add_symptom(self, name: str, measurement_name: str,
                    condition: str, severity: float = 0.5) -> Symptom:
        """Add a known symptom."""
        symptom = Symptom(
            name=name,
            measurement_name=measurement_name,
            condition=condition,
            severity=severity,
        )
        self.symptoms[name] = symptom
        return symptom
    
    def add_procedure(self, name: str, target_symptom: str) -> Procedure:
        """Add a procedure (initially empty, add steps separately)."""
        proc = Procedure(name=name, target_symptom=target_symptom)
        self.procedures[name] = proc
        return proc
    
    def measure(self, name: str, value: float, timestamp: float = None) -> Measurement:
        """
        Record a measurement and compare to spec if known.
        """
        if timestamp is None:
            timestamp = time()
        
        m = Measurement(name=name, value=value, timestamp=timestamp)
        
        if name in self.specs:
            m.spec = self.specs[name]
            m.result = m.spec.measure(value)
        
        self.measurements[name] = m
        return m
    
    def get_active_symptoms(self) -> List[Symptom]:
        """
        Check measurements against symptoms, return active ones.
        
        A symptom is active if its measurement is out of spec
        (R > THRESHOLD_RIGHTEOUSNESS).
        """
        active = []
        for symptom in self.symptoms.values():
            m = self.measurements.get(symptom.measurement_name)
            if m and m.result and m.result.get('R', 0) > THRESHOLD_RIGHTEOUSNESS:
                active.append(symptom)
        return active
    
    def get_procedures_for_symptom(self, symptom_name: str) -> List[Procedure]:
        """Get procedures that target a symptom, sorted by heat (confidence)."""
        procs = [p for p in self.procedures.values() if p.target_symptom == symptom_name]
        return sorted(procs, key=lambda p: p.heat, reverse=True)
    
    def learn_spec(self, name: str, value: float, unit: str = "", 
                   constraint_type: str = None) -> Spec:
        """
        Learn a spec from observation (Robinson minimum).
        
        If we don't know the spec, create one from what we observed.
        Wide tolerance - we're just learning.
        Constraint type is inferred from unit if not specified.
        """
        spec = robinson_spec(value, unit, constraint_type)
        self.specs[name] = spec
        return spec
    
    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "specs": {k: v.to_dict() for k, v in self.specs.items()},
            "symptoms": {k: v.to_dict() for k, v in self.symptoms.items()},
            "procedures": {k: v.to_dict() for k, v in self.procedures.items()},
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'DiagnosticFrame':
        frame = cls(domain=data["domain"])
        for name, sdata in data.get("specs", {}).items():
            frame.specs[name] = Spec.from_dict(sdata)
        for name, sydata in data.get("symptoms", {}).items():
            frame.symptoms[name] = Symptom.from_dict(sydata)
        for name, pdata in data.get("procedures", {}).items():
            frame.procedures[name] = Procedure.from_dict(pdata)
        return frame


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTRAINT TYPE INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

# Unit patterns for each Robinson constraint type
SPATIAL_UNITS = {'mm', 'm', 'cm', 'km', 'in', 'ft', 'yd', 'mi', 'nm', 'um', 'μm'}
TEMPORAL_UNITS = {'s', 'ms', 'us', 'μs', 'ns', 'min', 'hr', 'h', 'day', 'days', 'sec', 'seconds', 'minutes', 'hours'}
MASS_UNITS = {'kg', 'g', 'mg', 'lb', 'lbs', 'oz', 't', 'ton', 'tons'}
QUANTITY_UNITS = {'count', 'pcs', 'units', 'items', '$', 'dollars', 'cents', 'psi', 'bar', 'pa', 'kpa', 'mpa', 'n', 'kn', 'nm', 'ft-lb'}

def infer_constraint_from_unit(unit: str) -> str:
    """
    Infer Robinson constraint type from measurement unit.
    
    Returns:
        'identity':       Categorical/existence measurements
        'successor':      Spatial/length measurements  
        'addition':       Temporal/duration measurements
        'multiplication': Mass/quantity measurements
    """
    if not unit:
        return "identity"
    
    unit_lower = unit.lower().strip()
    
    # Check spatial (successor)
    if unit_lower in SPATIAL_UNITS:
        return "successor"
    
    # Check temporal (addition)
    if unit_lower in TEMPORAL_UNITS:
        return "addition"
    
    # Check mass/quantity (multiplication)
    if unit_lower in MASS_UNITS or unit_lower in QUANTITY_UNITS:
        return "multiplication"
    
    # Default to identity (categorical)
    return "identity"


# ═══════════════════════════════════════════════════════════════════════════════
# ROBINSON MINIMUM - The floor
# ═══════════════════════════════════════════════════════════════════════════════

def robinson_spec(observed: float, unit: str = "", constraint_type: str = None) -> Spec:
    """
    Create minimum valid spec from single observation.
    
    Robinson floor:
    - 0 exists (we have a measurement)
    - Successor exists (we can measure again)
    - That's all we know
    
    Wide tolerance - 50% either direction, scaled by Robinson constraint.
    No heat yet - no confidence.
    
    Args:
        observed: The observed value
        unit: Measurement unit (used to infer constraint_type if not specified)
        constraint_type: Robinson constraint type (identity/successor/addition/multiplication)
    """
    # Infer constraint type from unit if not specified
    if constraint_type is None:
        constraint_type = infer_constraint_from_unit(unit)
    
    # Get Robinson factor for this constraint
    robinson_factor = get_robinson_constraint(constraint_type)
    
    # Base margin is 50%, but scale by Robinson factor
    # Successor (spatial) is most precise, multiplication (mass) has more variance
    base_margin = abs(observed) * 0.5 if observed != 0 else 1.0
    margin = base_margin * robinson_factor
    
    return Spec(
        nominal=observed,
        tolerance_low=observed - margin,
        tolerance_high=observed + margin,
        unit=unit,
        source=SpecSource.ROBINSON,
        constraint_type=constraint_type,
        heat=0.0,
    )


def tighten_spec(spec: Spec, value: float, success: bool) -> None:
    """
    Learn from outcome.
    
    Success: heat accumulates, tolerance tightens.
    Failure: no heat added (heat never subtracts), spec unchanged.
    
    Tolerance tightens as confidence grows:
        0 confidence → 50% tolerance
        1 confidence → 10% tolerance
    """
    if not success:
        spec.polarity = -1
        return
    
    # Heat flows in
    spec.heat += K
    spec.polarity = 1
    
    # Tighten tolerance based on confidence
    confidence = spec.confidence
    # At 0 confidence: 50% tolerance
    # At 1 confidence: 10% tolerance
    tightness = 0.5 - (confidence * 0.4)
    
    # Move nominal toward successful value (weighted average)
    spec.nominal = (spec.nominal + value) / 2
    
    # Tighten range
    new_margin = abs(spec.nominal) * tightness if spec.nominal != 0 else tightness
    spec.tolerance_low = spec.nominal - new_margin
    spec.tolerance_high = spec.nominal + new_margin
    
    # Graduate from Robinson to Learned
    if spec.source == SpecSource.ROBINSON:
        spec.source = SpecSource.LEARNED
